"""Persistent ChromaDB-backed vector store.

Storage is persistent: the collection is created once on disk and reused across
process restarts, so ingestion work is never repeated. Records are written with
``upsert`` using deterministic chunk IDs, which makes re-ingesting an unchanged
document idempotent instead of duplicating it.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any

from app.core.config import Settings, get_settings
from app.core.errors import VectorStoreError
from app.ingestion.models import TextChunk

logger = logging.getLogger(__name__)

# Chroma stores cosine *distance* (0 = identical) when the collection is
# created with the cosine space; we surface the equivalent similarity.
_DEFAULT_INCLUDE = ["documents", "metadatas", "distances"]


class ChromaVectorStore:
    """ChromaDB implementation of the :class:`VectorStore` protocol."""

    def __init__(self, settings: Settings | None = None) -> None:
        """Open (or create) the persistent ChromaDB collection.

        Args:
            settings: Configuration; defaults to the process settings.

        Raises:
            VectorStoreError: If ChromaDB is missing or the store cannot be
                opened (e.g. unwritable directory).
        """
        self._settings = settings or get_settings()
        self._persist_dir = Path(self._settings.chroma_persist_directory)
        self._lock = threading.RLock()
        self._client = self._create_client()
        self._collection = self._get_or_create_collection()

    # -- construction helpers ---------------------------------------------

    def _create_client(self):
        """Create the persistent Chroma client with telemetry disabled."""
        try:
            import chromadb
            from chromadb.config import Settings as ChromaSettings
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise VectorStoreError(
                "chromadb is not installed. Install the project requirements "
                "to enable the vector store."
            ) from exc

        try:
            self._persist_dir.mkdir(parents=True, exist_ok=True)
            client = chromadb.PersistentClient(
                path=str(self._persist_dir),
                settings=ChromaSettings(anonymized_telemetry=False, allow_reset=True),
            )
        except Exception as exc:
            logger.error("Could not open ChromaDB at %s: %s", self._persist_dir, exc)
            raise VectorStoreError(
                "The vector database is unavailable. Check that "
                "CHROMA_PERSIST_DIRECTORY points to a writable location."
            ) from exc

        logger.info("ChromaDB ready at %s", self._persist_dir)
        return client

    def _get_or_create_collection(self):
        """Return the configured collection, creating it only when missing."""
        name = self._settings.chroma_collection
        try:
            return self._client.get_or_create_collection(
                name=name, metadata={"hnsw:space": "cosine"}
            )
        except Exception as exc:
            logger.error("Could not open Chroma collection '%s': %s", name, exc)
            raise VectorStoreError(
                "The vector database collection could not be opened."
            ) from exc

    @property
    def collection_name(self) -> str:
        """Name of the backing Chroma collection."""
        return self._settings.chroma_collection

    # -- VectorStore protocol ---------------------------------------------

    def add_documents(
        self, chunks: Sequence[TextChunk], embeddings: Sequence[Sequence[float]]
    ) -> int:
        """Insert chunk embeddings, replacing records with the same ID.

        Args:
            chunks: Chunks to store, each carrying its own metadata.
            embeddings: Matching embedding vectors, in the same order.

        Returns:
            The number of records written.

        Raises:
            VectorStoreError: If the write fails or the inputs are misaligned.
        """
        if not chunks:
            return 0
        if len(chunks) != len(embeddings):
            raise VectorStoreError(
                "The number of chunks does not match the number of embeddings."
            )

        ids = [chunk.chunk_id for chunk in chunks]
        duplicates = {chunk_id for chunk_id in ids if ids.count(chunk_id) > 1}
        if duplicates:
            # Deterministic IDs make this impossible in practice; fail loudly
            # rather than silently dropping content.
            raise VectorStoreError(
                f"Generated duplicate chunk IDs: {sorted(duplicates)[:3]}."
            )

        try:
            with self._lock:
                self._collection.upsert(
                    ids=ids,
                    documents=[chunk.text for chunk in chunks],
                    metadatas=[chunk.to_metadata() for chunk in chunks],
                    embeddings=[list(vector) for vector in embeddings],
                )
        except VectorStoreError:
            raise
        except Exception as exc:
            logger.exception("ChromaDB upsert failed")
            raise VectorStoreError("Failed to store the document chunks.") from exc

        logger.info("Stored %s chunk(s) in '%s'.", len(chunks), self.collection_name)
        return len(chunks)

    def search(
        self, query_embedding: Sequence[float], top_k: int
    ) -> list[dict[str, Any]]:
        """Return the most similar stored chunks.

        Returns:
            A list of ``{"id", "document", "text", "score"}`` dictionaries,
            most similar first. Empty when the store is empty.
        """
        available = self.count()
        if available == 0 or top_k <= 0:
            return []

        n_results = min(top_k, available)
        try:
            with self._lock:
                raw = self._collection.query(
                    query_embeddings=[list(query_embedding)],
                    n_results=n_results,
                    include=_DEFAULT_INCLUDE,
                )
        except Exception as exc:
            logger.exception("ChromaDB query failed")
            raise VectorStoreError("Semantic search failed.") from exc

        return self._parse_results(raw)

    def delete_collection(self) -> None:
        """Delete the whole collection (used by maintenance scripts and tests)."""
        with self._lock:
            try:
                self._client.delete_collection(self._settings.chroma_collection)
            except Exception as exc:
                logger.warning("delete_collection failed: %s", exc)
            self._collection = self._get_or_create_collection()
        logger.info("Collection '%s' recreated empty.", self.collection_name)

    def delete_by_document(self, document_name: str) -> int:
        """Remove all chunks of one document, e.g. before re-ingesting it.

        Returns:
            The number of records that were removed.
        """
        with self._lock:
            before = self.count()
            try:
                self._collection.delete(where={"document_name": document_name})
            except Exception as exc:
                logger.exception("ChromaDB delete failed")
                raise VectorStoreError(
                    "Failed to remove the previous version of the document."
                ) from exc
            removed = max(0, before - self.count())
        if removed:
            logger.info(
                "Removed %s existing chunk(s) for document '%s'.",
                removed,
                document_name,
            )
        return removed

    def count(self) -> int:
        """Number of records currently stored."""
        try:
            with self._lock:
                return int(self._collection.count())
        except Exception as exc:
            logger.exception("ChromaDB count failed")
            raise VectorStoreError("Failed to read the vector store.") from exc

    def list_documents(self) -> list[str]:
        """Return the distinct document names currently indexed."""
        try:
            with self._lock:
                stored = self._collection.get(include=["metadatas"])
        except Exception as exc:
            raise VectorStoreError("Failed to list indexed documents.") from exc
        names = {
            metadata.get("document_name")
            for metadata in (stored.get("metadatas") or [])
            if metadata
        }
        return sorted(name for name in names if name)

    # -- internals ---------------------------------------------------------

    @staticmethod
    def _parse_results(raw: dict[str, Any]) -> list[dict[str, Any]]:
        """Flatten Chroma's nested query response into simple records."""
        documents = (raw.get("documents") or [[]])[0]
        metadatas = (raw.get("metadatas") or [[]])[0]
        distances = (raw.get("distances") or [[]])[0]
        ids = (raw.get("ids") or [[]])[0]

        results: list[dict[str, Any]] = []
        for index, text in enumerate(documents):
            metadata = metadatas[index] if index < len(metadatas) else {}
            metadata = metadata or {}
            distance = float(distances[index]) if index < len(distances) else 1.0
            results.append(
                {
                    "id": ids[index] if index < len(ids) else metadata.get("chunk_id", ""),
                    "document": metadata.get("document_name", "unknown"),
                    "source": metadata.get("source", "unknown"),
                    "page": int(metadata.get("page", 0) or 0),
                    "chunk_index": int(metadata.get("chunk_index", 0) or 0),
                    "text": text,
                    # Cosine distance -> similarity, clamped to [-1, 1].
                    "score": max(-1.0, min(1.0, 1.0 - distance)),
                }
            )
        return results


def create_chroma_store(settings: Settings | None = None) -> ChromaVectorStore:
    """Create a new Chroma store instance (no caching)."""
    return ChromaVectorStore(settings)


@lru_cache(maxsize=1)
def get_chroma_store() -> ChromaVectorStore:
    """Return the process-wide store singleton.

    A single instance is shared deliberately: creating several persistent
    Chroma clients against the same directory can corrupt internal state.
    """
    return ChromaVectorStore(get_settings())
