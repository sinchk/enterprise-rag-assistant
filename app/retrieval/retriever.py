"""Question answering over the indexed document collection."""

from __future__ import annotations

import logging

from app.core.config import Settings, get_settings
from app.core.errors import RetrievalError
from app.embeddings.embedder import Embedder, SentenceTransformerEmbedder
from app.retrieval.models import RetrievedChunk
from app.vectorstore.base import VectorStore
from app.vectorstore.chroma_store import ChromaVectorStore, get_chroma_store

logger = logging.getLogger(__name__)


class Retriever:
    """Retrieve the ``top_k`` chunks most similar to a question."""

    def __init__(
        self,
        embedder: Embedder,
        store: VectorStore,
        *,
        top_k: int = 4,
        similarity_threshold: float | None = None,
    ) -> None:
        """Create a retriever.

        Args:
            embedder: Component used to embed the question.
            store: Vector store to search.
            top_k: Default number of chunks to return.
            similarity_threshold: Optional minimum cosine similarity. Chunks
                below the threshold are dropped; an empty result then means
                "the documents do not cover this question".
        """
        self._embedder = embedder
        self._store = store
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold

    @classmethod
    def from_settings(
        cls,
        settings: Settings | None = None,
        store: VectorStore | None = None,
        embedder: Embedder | None = None,
    ) -> "Retriever":
        """Build a retriever from configuration and shared components."""
        settings = settings or get_settings()
        return cls(
            embedder=embedder or SentenceTransformerEmbedder(settings),
            store=store or ChromaVectorStore(settings),
            top_k=settings.top_k,
            similarity_threshold=settings.similarity_threshold,
        )

    def retrieve(self, question: str, top_k: int | None = None) -> list[RetrievedChunk]:
        """Return the chunks most relevant to ``question``.

        Args:
            question: Natural-language question.
            top_k: Optional per-call override of the configured ``top_k``.

        Returns:
            Ranked chunks, best match first. Empty when nothing clears the
            similarity threshold or the store is empty.

        Raises:
            RetrievalError: If the question is blank or retrieval fails.
        """
        cleaned = question.strip()
        if not cleaned:
            raise RetrievalError("A non-empty question is required.")

        limit = top_k or self.top_k
        try:
            embedding = self._embedder.embed_text(cleaned)
        except Exception as exc:
            if isinstance(exc, RetrievalError):
                raise
            logger.exception("Failed to embed the question")
            raise RetrievalError(
                "Failed to process the question with the embedding model."
            ) from exc

        try:
            records = self._store.search(embedding, limit)
        except Exception as exc:
            if isinstance(exc, RetrievalError):
                raise
            logger.exception("Vector store search failed")
            raise RetrievalError("Could not search the document collection.") from exc

        chunks: list[RetrievedChunk] = []
        for record in records:
            score = float(record.get("score", 0.0))
            if self.similarity_threshold is not None and score < self.similarity_threshold:
                logger.debug(
                    "Dropping chunk %s (score %.3f < threshold %.3f).",
                    record.get("id"),
                    score,
                    self.similarity_threshold,
                )
                continue
            chunks.append(
                RetrievedChunk(
                    chunk_id=str(record.get("id", "")),
                    text=str(record.get("text", "")),
                    document=str(record.get("document", "unknown")),
                    source=str(record.get("source", "unknown")),
                    page=int(record.get("page", 0) or 0),
                    score=score,
                    rank=len(chunks) + 1,
                )
            )

        logger.info(
            "Retrieved %s chunk(s) for question (top score %.3f).",
            len(chunks),
            chunks[0].score if chunks else 0.0,
        )
        return chunks


def get_retriever(
    settings: Settings | None = None, store: VectorStore | None = None
) -> Retriever:
    """Build a retriever using the process-wide vector store by default."""
    settings = settings or get_settings()
    return Retriever.from_settings(settings, store=store or get_chroma_store())
