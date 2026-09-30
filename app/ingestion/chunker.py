"""Configurable, metadata-preserving text chunking.

Design decision - why a hand-written word-based splitter?
----------------------------------------------------------
The simplest reliable approach for an MVP is a sliding window over whitespace
tokens with a configurable size and overlap. It has three advantages over a
tokeniser-based or recursive-character splitter:

* **No extra dependency** and no hidden model download for token counting.
* **Explainable**: chunk boundaries are visible in the code, which matters for
  a system whose answers are cited back to a page.
* **Sentence-aware**: the window end snaps forward to the closest sentence
  boundary inside the second half of the window, so a chunk rarely cuts a
  sentence in half and embeddings of truncated sentences stay meaningful.

Words are used rather than model tokens because English prose averages roughly
1.3 tokens/word; a 600-word chunk is therefore in the 500-800 token band the
default configuration targets. The mapping is approximate and documented as a
known limitation.

Chunks are produced **per page**. A chunk never spans two pages, which keeps
page attribution exact - a requirement for the ``sources`` field of the answer.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass

from app.core.config import Settings, get_settings
from app.ingestion.models import PageDocument, TextChunk

logger = logging.getLogger(__name__)

_SENTENCE_END_RE = re.compile(r"[.!?][\"'\)\]]*$")


@dataclass(frozen=True, slots=True)
class ChunkerStats:
    """Summary of a chunking run, returned for logging and API responses."""

    pages: int
    chunks: int
    words: int

    @property
    def average_words_per_chunk(self) -> float:
        """Mean chunk length in words (0.0 when there are no chunks)."""
        return self.words / self.chunks if self.chunks else 0.0


class TextChunker:
    """Split cleaned page text into overlapping, sentence-aligned chunks."""

    def __init__(
        self,
        chunk_size: int = 600,
        chunk_overlap: int = 80,
        *,
        respect_sentences: bool = True,
    ) -> None:
        """Create a chunker.

        Args:
            chunk_size: Target chunk size in words.
            chunk_overlap: Number of words shared with the following chunk.
            respect_sentences: Snap window ends to the closest sentence
                boundary found in the latter half of the window.

        Raises:
            ValueError: If the size/overlap combination is invalid.
        """
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap must not be negative")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.respect_sentences = respect_sentences

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> "TextChunker":
        """Build a chunker from application configuration."""
        settings = settings or get_settings()
        return cls(chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap)

    # -- public API --------------------------------------------------------

    def chunk_pages(self, pages: Sequence[PageDocument]) -> list[TextChunk]:
        """Chunk every page, preserving each page's source metadata."""
        chunks: list[TextChunk] = []
        for page in pages:
            chunks.extend(self.chunk_page(page))
        logger.info(
            "Chunked %s page(s) into %s chunk(s) (avg %.0f words).",
            len(pages),
            len(chunks),
            (sum(c.word_count for c in chunks) / len(chunks)) if chunks else 0.0,
        )
        return chunks

    def chunk_page(self, page: PageDocument) -> list[TextChunk]:
        """Chunk a single page into :class:`TextChunk` objects."""
        texts = self.split_text(page.text)
        return [
            TextChunk.from_text(
                text,
                document_name=page.document_name,
                source=page.metadata.source,
                page=page.page,
                chunk_index=index,
                total_chunks=len(texts),
            )
            for index, text in enumerate(texts)
        ]

    def split_text(self, text: str) -> list[str]:
        """Split raw text into overlapping word windows.

        Returns:
            Chunk texts in document order. Blank text yields an empty list.
        """
        words = text.split()
        if not words:
            return []

        boundaries = self._sentence_boundaries(words) if self.respect_sentences else []
        min_fill = max(1, self.chunk_size // 2)

        chunks: list[str] = []
        start = 0
        total = len(words)

        while start < total:
            end = min(start + self.chunk_size, total)
            if end < total:
                snapped = self._snap_to_sentence(boundaries, start, end, min_fill)
                if snapped is not None:
                    end = snapped

            # Every window is emitted, even when two windows render identical
            # text. De-duplicating them would silently drop content, because
            # identical text usually means genuinely repetitive source text
            # (repeated clauses) rather than an artefact of the overlap.
            window = " ".join(words[start:end])
            if window:
                chunks.append(window)

            if end >= total:
                break

            # Step forward by (window - overlap) words, always making progress.
            next_start = end - self.chunk_overlap
            start = max(start + 1, next_start)

        return chunks

    def stats(self, chunks: Sequence[TextChunk], page_count: int) -> ChunkerStats:
        """Return summary statistics for a set of chunks."""
        return ChunkerStats(
            pages=page_count,
            chunks=len(chunks),
            words=sum(chunk.word_count for chunk in chunks),
        )

    # -- internals ---------------------------------------------------------

    @staticmethod
    def _sentence_boundaries(words: Sequence[str]) -> list[int]:
        """Indices just after each sentence-ending word."""
        return [
            index + 1
            for index, word in enumerate(words)
            if _SENTENCE_END_RE.search(word)
        ]

    @staticmethod
    def _snap_to_sentence(
        boundaries: Sequence[int], start: int, end: int, min_fill: int
    ) -> int | None:
        """Return the last sentence boundary inside ``(start+min_fill, end]``."""
        candidate: int | None = None
        for boundary in boundaries:
            if boundary <= start + min_fill:
                continue
            if boundary > end:
                break
            candidate = boundary
        return candidate
