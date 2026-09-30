"""Conservative text cleaning for PDF-extracted content.

The goal is to remove extraction artefacts (soft hyphenation, repeated blank
lines, control characters, stray glyph boxes, running headers/footers) while
preserving wording and ordering. The cleaner never rewrites, summarises or
paraphrases text: aggressive normalisation would make citations unreliable.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from collections import Counter
from collections.abc import Sequence

from app.ingestion.models import PageDocument

logger = logging.getLogger(__name__)

# Soft hyphenation introduced when a PDF line wraps mid-word.
_HYPHENATED_LINE_BREAK_RE = re.compile(r"(?<=[A-Za-z])-[ \t]*\n[ \t]*(?=[a-z])")
# Ligatures and typographic characters that NFKC does not fully resolve.
_CHARACTER_MAP = {
    "\u00a0": " ",  # non-breaking space
    "\u2007": " ",  # figure space
    "\u202f": " ",  # narrow no-break space
    "\u200b": "",  # zero-width space
    "\u200c": "",  # zero-width non-joiner
    "\u200d": "",  # zero-width joiner
    "\ufeff": "",  # byte order mark
    "\u00ad": "",  # soft hyphen
    "\u2018": "'",
    "\u2019": "'",
    "\u201a": "'",
    "\u201c": '"',
    "\u201d": '"',
    "\u201e": '"',
    "\u2013": "-",
    "\u2014": "-",
    "\u2212": "-",
    "\u2026": "...",
    "\ufb00": "ff",
    "\ufb01": "fi",
    "\ufb02": "fl",
}
_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_INLINE_WHITESPACE_RE = re.compile(r"[ \t]+")
_SPACE_BEFORE_PUNCT_RE = re.compile(r"[ \t]+([,.;:!?%)\]])")
_BLANK_LINES_RE = re.compile(r"\n{3,}")
# A line is considered noise when it contains no letters or digits.
_HAS_ALNUM_RE = re.compile(r"[A-Za-z0-9]")
_TRAILING_WS_RE = re.compile(r"[ \t]+$", re.MULTILINE)
_DIGIT_RUN_RE = re.compile(r"\d+")

# Running headers/footers repeat across most pages; treat a line as marginalia
# when it appears on at least this share of the first/last lines of pages.
_MARGINALIA_MIN_PAGES = 3
_MARGINALIA_RATIO = 0.6
# Page headers are commonly a title line followed by a page-number line, so
# three lines from each edge are inspected.
_MARGINALIA_EDGE_LINES = 3
# Guard against stripping legitimate short lines such as a single-word heading.
_MARGINALIA_MIN_LENGTH = 4


class TextCleaner:
    """Normalise extracted page text without altering its meaning."""

    def __init__(
        self,
        *,
        join_wrapped_lines: bool = True,
        strip_repeated_marginalia: bool = True,
        min_line_words: int = 1,
    ) -> None:
        """Create a cleaner.

        Args:
            join_wrapped_lines: Join single newlines inside a paragraph into a
                single line. PDF extraction emits one line per visual line, and
                a broken sentence embeds poorly. Paragraph breaks (blank lines)
                are always preserved.
            strip_repeated_marginalia: Remove running headers/footers that
                repeat across the majority of pages.
            min_line_words: Drop artefact lines containing fewer words than this
                and no digits (e.g. a lone bullet glyph or rule line).
        """
        self.join_wrapped_lines = join_wrapped_lines
        self.strip_repeated_marginalia = strip_repeated_marginalia
        self.min_line_words = min_line_words

    # -- public API --------------------------------------------------------

    def clean(self, text: str) -> str:
        """Clean a single block of extracted text."""
        if not text or not text.strip():
            return ""

        cleaned = self._normalise_characters(text)
        cleaned = self._join_hyphenated_words(cleaned)
        cleaned = self._drop_noise_lines(cleaned)
        cleaned = self._collapse_whitespace(cleaned)
        if self.join_wrapped_lines:
            cleaned = self._join_wrapped_lines(cleaned)
        return cleaned.strip()

    def clean_pages(self, pages: Sequence[PageDocument]) -> list[PageDocument]:
        """Clean a full document and remove repeated headers/footers.

        Marginalia detection runs on the *pre-join* representation, i.e. while
        the extractor's hard line breaks are still present. Doing it after
        line folding would merge a header, the page title and the page number
        into one line, and the result would no longer match across pages.

        Returns a new list of pages; pages that become empty are dropped.
        """
        staged: list[PageDocument] = []
        for page in pages:
            text = self._normalise_characters(page.text)
            text = self._join_hyphenated_words(text)
            text = self._collapse_whitespace(text)
            staged.append(PageDocument(text=text, metadata=page.metadata))

        if self.strip_repeated_marginalia:
            staged = self._strip_marginalia(staged)

        cleaned_pages = [
            PageDocument(text=self.clean(page.text), metadata=page.metadata)
            for page in staged
        ]
        return [page for page in cleaned_pages if page.text.strip()]

    # -- internals ---------------------------------------------------------

    @staticmethod
    def _normalise_characters(text: str) -> str:
        """Normalise unicode, drop control characters and map dash/quote noise."""
        text = unicodedata.normalize("NFKC", text)
        for bad, good in _CHARACTER_MAP.items():
            if bad in text:
                text = text.replace(bad, good)
        text = _CONTROL_CHARS_RE.sub(" ", text)
        return text.replace("\r\n", "\n").replace("\r", "\n")

    @staticmethod
    def _join_hyphenated_words(text: str) -> str:
        """Re-join words that were split across a line break by the extractor."""
        return _HYPHENATED_LINE_BREAK_RE.sub("", text)

    def _drop_noise_lines(self, text: str) -> str:
        """Remove lines that carry no textual content."""
        kept: list[str] = []
        for line in text.split("\n"):
            stripped = line.strip()
            if not stripped:
                kept.append(line)
                continue
            if not _HAS_ALNUM_RE.search(stripped):
                continue
            if (
                len(stripped.split()) < self.min_line_words
                and not any(char.isdigit() for char in stripped)
            ):
                continue
            kept.append(stripped)
        return "\n".join(kept)

    @staticmethod
    def _collapse_whitespace(text: str) -> str:
        """Collapse runs of spaces/tabs and excessive blank lines."""
        text = text.replace("\t", " ")
        text = _INLINE_WHITESPACE_RE.sub(" ", text)
        text = _SPACE_BEFORE_PUNCT_RE.sub(r"\1", text)
        text = _TRAILING_WS_RE.sub("", text)
        return _BLANK_LINES_RE.sub("\n\n", text)

    @staticmethod
    def _join_wrapped_lines(text: str) -> str:
        """Fold wrapped lines into paragraphs, keeping blank-line breaks."""
        paragraphs = text.split("\n\n")
        folded: list[str] = []
        for paragraph in paragraphs:
            lines = [line.strip() for line in paragraph.split("\n") if line.strip()]
            if not lines:
                continue
            folded.append(" ".join(lines))
        return "\n\n".join(folded)

    @staticmethod
    def _edge_lines(text: str) -> list[str]:
        """Return the first and last few non-empty lines of a page."""
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        if len(lines) <= _MARGINALIA_EDGE_LINES:
            return lines
        return lines[:_MARGINALIA_EDGE_LINES] + lines[-_MARGINALIA_EDGE_LINES:]

    def _strip_marginalia(self, pages: list[PageDocument]) -> list[PageDocument]:
        """Drop running headers/footers that repeat on most pages.

        This is a heuristic. A line qualifies when its *signature* - the line
        with all digit runs masked, lower-cased - appears within the first or
        last ``_MARGINALIA_EDGE_LINES`` lines of at least ``_MARGINALIA_RATIO``
        of all pages. Masking digits is what lets "Page 2 of 7" and "Page 3 of
        7" count as the same running footer. Qualifying lines must also be long
        enough to protect a short but legitimate heading, and must either
        contain a digit or more than one word.
        """
        if len(pages) < _MARGINALIA_MIN_PAGES:
            return pages

        counter: Counter[str] = Counter()
        for page in pages:
            counter.update({self._signature(line) for line in self._edge_lines(page.text)})

        threshold = max(_MARGINALIA_MIN_PAGES, int(len(pages) * _MARGINALIA_RATIO))
        repeated = {
            signature
            for signature, count in counter.items()
            if count >= threshold
            and len(signature) >= _MARGINALIA_MIN_LENGTH
            and (any(char.isdigit() for char in signature) or len(signature.split()) > 1)
        }
        if not repeated:
            return pages

        stripped_pages: list[PageDocument] = []
        removed = 0
        for page in pages:
            kept: list[str] = []
            for line in page.text.split("\n"):
                if self._signature(line) in repeated:
                    removed += 1
                    continue
                kept.append(line)
            stripped_pages.append(
                PageDocument(text="\n".join(kept).strip(), metadata=page.metadata)
            )
        logger.info("Removed %s repeated header/footer line(s).", removed)
        return stripped_pages

    @staticmethod
    def _signature(line: str) -> str:
        """Return a digit-masked, normalised form of a line for matching."""
        return _DIGIT_RUN_RE.sub("#", line).strip().lower()
