"""Result objects returned by the retrieval layer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    """A chunk returned by semantic search, with its relevance score."""

    chunk_id: str
    text: str
    document: str
    source: str
    page: int
    score: float
    rank: int = 0

    @property
    def citation(self) -> str:
        """Human readable citation such as ``policy.pdf, page 7``."""
        return f"{self.document}, page {self.page}"

    @property
    def label(self) -> str:
        """Short label used inside the prompt, e.g. ``[S1]``."""
        return f"[S{self.rank}]"

    def to_source(self) -> dict[str, Any]:
        """Return the API-facing source record."""
        return {
            "document": self.document,
            "page": self.page,
            "score": round(self.score, 4),
            "chunk_id": self.chunk_id,
        }
