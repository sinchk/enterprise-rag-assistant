"""Structured document objects produced by the ingestion layer.

Frozen dataclasses are used instead of Pydantic models on purpose: these are
internal domain objects where immutability and cheap construction matter,
while Pydantic is reserved for API request/response schemas and settings.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field

_SLUG_RE = re.compile(r"[^a-z0-9]+")


def slugify(value: str, max_length: int = 60) -> str:
    """Return a filesystem/ID-safe slug for a document name."""
    slug = _SLUG_RE.sub("-", value.strip().lower()).strip("-")
    return (slug[:max_length] or "document").strip("-")


def content_hash(text: str) -> str:
    """Return a short, stable hash of ``text`` (used inside chunk IDs)."""
    normalised = " ".join(text.split())
    return hashlib.sha256(normalised.encode("utf-8")).hexdigest()[:12]


@dataclass(frozen=True, slots=True)
class PageMetadata:
    """Provenance information for a single extracted page."""

    document_name: str
    source: str
    page: int
    page_count: int = 0
    sha256: str = ""

    def to_dict(self) -> dict[str, str | int]:
        """Return a JSON-serialisable representation."""
        return {
            "document_name": self.document_name,
            "source": self.source,
            "page": self.page,
            "page_count": self.page_count,
        }


@dataclass(frozen=True, slots=True)
class PageDocument:
    """Text extracted from one PDF page, with its metadata preserved."""

    text: str
    metadata: PageMetadata

    @property
    def page(self) -> int:
        """1-based page number."""
        return self.metadata.page

    @property
    def document_name(self) -> str:
        """Original file name of the source document."""
        return self.metadata.document_name

    @property
    def word_count(self) -> int:
        """Number of whitespace-separated words in the page text."""
        return len(self.text.split())


@dataclass(frozen=True, slots=True)
class TextChunk:
    """A retrievable unit of text with full source attribution."""

    chunk_id: str
    text: str
    document_name: str
    source: str
    page: int
    chunk_index: int
    total_chunks: int = 0
    word_count: int = field(default=0, compare=False)

    @property
    def citation(self) -> str:
        """Human readable citation such as ``policy.pdf, page 7``."""
        return f"{self.document_name}, page {self.page}"

    def to_metadata(self) -> dict[str, str | int]:
        """Return ChromaDB-compatible metadata (scalars only)."""
        return {
            "chunk_id": self.chunk_id,
            "document_name": self.document_name,
            "source": self.source,
            "page": self.page,
            "chunk_index": self.chunk_index,
            "total_chunks": self.total_chunks,
            "word_count": self.word_count,
        }

    @classmethod
    def from_text(
        cls,
        text: str,
        *,
        document_name: str,
        source: str,
        page: int,
        chunk_index: int,
        total_chunks: int = 0,
        content_hash_value: str | None = None,
    ) -> "TextChunk":
        """Build a chunk with a deterministic, content-sensitive ID.

        The ID combines a slugified document name, the page, the position
        inside the page and a hash of the chunk text. Re-ingesting an
        unchanged document therefore yields identical IDs, which makes the
        vector store upsert idempotent instead of duplicating content.
        """
        digest = content_hash_value or content_hash(text)
        chunk_id = f"{slugify(document_name)}-p{page:04d}-c{chunk_index:04d}-{digest}"
        return cls(
            chunk_id=chunk_id,
            text=text,
            document_name=document_name,
            source=source,
            page=page,
            chunk_index=chunk_index,
            total_chunks=total_chunks,
            word_count=len(text.split()),
        )
