"""PDF text extraction built on PyMuPDF (``fitz``).

Scope note: this loader handles text-based PDFs. Scanned or image-only PDFs
contain no text layer and return empty pages; they are rejected with
:class:`~app.core.errors.EmptyDocumentError` because OCR is out of scope for
this MVP.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import fitz  # PyMuPDF

from app.core.errors import EmptyDocumentError, InvalidPdfError
from app.ingestion.models import PageDocument, PageMetadata

logger = logging.getLogger(__name__)

_PDF_SUFFIXES = {".pdf"}


class PdfLoader:
    """Extract page-by-page text from a PDF while preserving page numbers."""

    def __init__(self, max_pages: int | None = None) -> None:
        """Create a loader.

        Args:
            max_pages: Optional guardrail limiting how many pages are read,
                which protects against pathologically large documents.
        """
        self.max_pages = max_pages

    def load(self, path: str | Path) -> list[PageDocument]:
        """Load a PDF from disk and return one :class:`PageDocument` per page.

        Args:
            path: Path to a ``.pdf`` file.

        Returns:
            Extracted pages in document order.

        Raises:
            InvalidPdfError: The file is missing, has the wrong extension or
                cannot be parsed as a PDF.
            EmptyDocumentError: No page yielded any text.
        """
        pdf_path = Path(path)
        if not pdf_path.exists() or not pdf_path.is_file():
            raise InvalidPdfError("The uploaded file could not be found.")
        if pdf_path.suffix.lower() not in _PDF_SUFFIXES:
            raise InvalidPdfError(
                "Only PDF documents are supported. Scanned/image-only PDFs "
                "require OCR, which is not available in this version."
            )

        digest = _sha256(pdf_path)
        document_name = pdf_path.name

        try:
            document = fitz.open(pdf_path)
        except Exception as exc:  # PyMuPDF raises several unrelated types here.
            logger.warning("Failed to open PDF %s: %s", document_name, exc)
            raise InvalidPdfError(
                "The file is not a readable PDF document."
            ) from exc

        try:
            if document.needs_pass:
                raise InvalidPdfError(
                    "The PDF is password protected and cannot be processed."
                )

            page_count = document.page_count
            pages: list[PageDocument] = []
            for index in range(page_count):
                if self.max_pages is not None and index >= self.max_pages:
                    logger.warning(
                        "Stopping extraction of %s at %s pages (max_pages).",
                        document_name,
                        self.max_pages,
                    )
                    break
                raw_text = document.load_page(index).get_text("text") or ""
                metadata = PageMetadata(
                    document_name=document_name,
                    source=document_name,
                    page=index + 1,
                    page_count=page_count,
                    sha256=digest,
                )
                pages.append(PageDocument(text=raw_text, metadata=metadata))
        finally:
            document.close()

        if not any(page.text.strip() for page in pages):
            raise EmptyDocumentError(
                "No text could be extracted from this PDF. It is most likely a "
                "scanned or image-only document, which requires OCR support."
            )

        logger.info(
            "Extracted %s page(s) from %s (%s non-empty).",
            len(pages),
            document_name,
            sum(1 for page in pages if page.text.strip()),
        )
        return pages


def _sha256(path: Path) -> str:
    """Return the SHA-256 of a file, read in chunks to bound memory usage."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
