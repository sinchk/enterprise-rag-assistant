"""Vector store abstraction and ChromaDB implementation.

The application talks to :class:`VectorStore`, a deliberately small protocol.
ChromaDB is the only implementation shipped, but Pinecone / Qdrant / pgvector
can be added later by implementing the same four methods.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


@runtime_checkable
class VectorStore(Protocol):
    """Minimal contract required by the ingestion and retrieval layers."""

    def add_documents(self, chunks: Sequence[Any], embeddings: Sequence[Sequence[float]]) -> int:
        """Insert or replace chunk embeddings; returns the number written."""
        ...

    def search(
        self, query_embedding: Sequence[float], top_k: int
    ) -> list[dict[str, Any]]:
        """Return the ``top_k`` most similar records for a query embedding."""
        ...

    def delete_collection(self) -> None:
        """Drop the collection and everything in it."""
        ...

    def delete_by_document(self, document_name: str) -> int:
        """Remove every record belonging to a document; returns the count."""
        ...

    def count(self) -> int:
        """Number of records currently stored."""
        ...
