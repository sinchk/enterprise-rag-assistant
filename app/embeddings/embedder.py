"""Embedding generation.

The rest of the application depends only on the :class:`Embedder` protocol, so
swapping ``all-MiniLM-L6-v2`` for a hosted embedding API (or a heavier local
model) is a configuration change plus one new implementation class - no changes
to the ingestion, retrieval or RAG code.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from app.core.config import Settings, get_settings
from app.core.errors import AppError

logger = logging.getLogger(__name__)


class EmbeddingError(AppError):
    """Failed to generate embeddings."""

    status_code = 500
    error_code = "embedding_failed"


@runtime_checkable
class Embedder(Protocol):
    """Minimal interface required by the ingestion and retrieval layers."""

    def embed_text(self, text: str) -> list[float]:
        """Embed a single string and return one dense vector."""
        ...

    def embed_documents(self, documents: Sequence[str]) -> list[list[float]]:
        """Embed a batch of strings and return one dense vector per input."""
        ...

    @property
    def dimension(self) -> int:
        """Length of the produced embedding vectors."""
        ...


class SentenceTransformerEmbedder:
    """Local embedding provider backed by ``sentence-transformers``.

    The model is loaded lazily on first use so that the FastAPI process starts
    even when no document has been ingested yet, and so that unit tests that
    never touch embeddings do not pay the model load cost.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._model = None
        self._lock = threading.Lock()
        self._dimension: int | None = None

    # -- lazy loading ------------------------------------------------------

    def _load(self):
        """Load and cache the underlying sentence-transformers model."""
        if self._model is not None:
            return self._model
        with self._lock:
            if self._model is not None:
                return self._model
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:  # pragma: no cover - dependency guard
                raise EmbeddingError(
                    "sentence-transformers is not installed. "
                    "Install the project requirements to enable embeddings."
                ) from exc

            logger.info("Loading embedding model '%s'...", self._settings.embedding_model)
            try:
                self._model = SentenceTransformer(
                    self._settings.embedding_model,
                    device=self._settings.embedding_device,
                )
            except Exception as exc:
                logger.error("Failed to load embedding model: %s", exc)
                raise EmbeddingError(
                    f"Could not load the embedding model "
                    f"'{self._settings.embedding_model}'."
                ) from exc
            self._dimension = int(self._model.get_sentence_embedding_dimension())
            logger.info("Embedding model ready (dimension=%s).", self._dimension)
            return self._model

    # -- Embedder protocol -------------------------------------------------

    def embed_text(self, text: str) -> list[float]:
        """Embed a single string.

        Raises:
            EmbeddingError: If ``text`` is empty or encoding fails.
        """
        vectors = self.embed_documents([text])
        return vectors[0]

    def embed_documents(self, documents: Sequence[str]) -> list[list[float]]:
        """Embed a batch of strings.

        Returns one vector per input, in the same order, so the caller can zip
        vectors with chunks positionally. Blank input is encoded as a single
        space rather than dropped, because dropping a vector would shift every
        later vector out of alignment with its chunk.

        Raises:
            EmbeddingError: If encoding fails.
        """
        if not documents:
            return []
        model = self._load()
        batch = [text if text.strip() else " " for text in documents]
        try:
            encoded = model.encode(
                batch,
                batch_size=self._settings.embedding_batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        except Exception as exc:
            logger.error("Embedding failed: %s", exc)
            raise EmbeddingError("Failed to generate embeddings for the text.") from exc
        return [vector.tolist() for vector in encoded]

    @property
    def dimension(self) -> int:
        """Embedding dimensionality (triggers the lazy model load once)."""
        if self._dimension is None:
            self._load()
        return int(self._dimension or 0)


def get_embedder(settings: Settings | None = None) -> Embedder:
    """Factory returning the configured embedder implementation."""
    return SentenceTransformerEmbedder(settings)
