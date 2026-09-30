"""Shared pytest fixtures.

Every fixture is isolated: the vector store, upload directory and PDFs live in
``tmp_path`` so the suite never touches the developer's real ``chroma_db`` or
``data/uploads`` directories, and no test requires an API key or network access.
"""

from __future__ import annotations

import sys
from pathlib import Path

import fitz
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import Settings  # noqa: E402

# Keeps the sample document visible to tests without an extra fixture.
SAMPLE_PDF = Path(__file__).resolve().parents[1] / "data" / "sample" / (
    "northwind_lending_policy.pdf"
)

LENDING_TEXT = [
    "Maximum loan tenure: 20 years. Minimum applicant age: 21 years.",
    "The standard annual percentage rate for unsecured loans is 11.9 percent.",
    "Proof of identity and proof of address must be submitted with an application.",
]


def make_pdf(path: Path, pages: list[list[str]] | None = None, blank: bool = False) -> Path:
    """Create a small text-based PDF for ingestion tests.

    Text is written with ``insert_textbox`` so long lines wrap inside the page
    the way a real word processor would; writing an over-wide line with
    ``insert_text`` makes PyMuPDF extract the glyphs one per line.

    Args:
        path: Destination path (should end in ``.pdf``).
        pages: Text lines per page. Ignored when ``blank`` is set.
        blank: When True, create a single page with no text layer, mimicking a
            scanned document that OCR would be required to read.
    """
    document = fitz.open()
    if blank:
        document.new_page()
    else:
        for index, lines in enumerate(pages or [LENDING_TEXT[0]], start=1):
            if isinstance(lines, str):
                raise TypeError(
                    f"pages[{index - 1}] must be a list of lines, not a string. "
                    "Pass [[line1], [line2]] to create one page per entry."
                )
            page = document.new_page()
            # insert_textbox returns a negative value when the text does not fit,
            # and writes nothing at all. Failing loudly here prevents a silently
            # empty page from looking like a PDF without a text layer.
            remaining = page.insert_textbox(
                fitz.Rect(72, 72, 523, 770),
                "\n".join(lines),
                fontname="helv",
                fontsize=11,
                lineheight=1.4,
            )
            if remaining < 0:
                raise ValueError(
                    f"Page {index} text overflows the text box by {-remaining:.0f} "
                    f"points: {lines!r}"
                )
    document.save(path)
    document.close()
    return path


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    """Return isolated test settings."""
    return Settings(
        environment="test",
        chroma_persist_directory=tmp_path / "chroma_db",
        chroma_collection="test_documents",
        upload_directory=tmp_path / "uploads",
        embedding_model="fake-model",
        llm_provider="extractive",
        llm_model="scripted",
        chunk_size=60,
        chunk_overlap=10,
        top_k=3,
        max_upload_size_mb=2,
    )


@pytest.fixture
def sample_pdf() -> Path:
    """Path to the committed synthetic sample document."""
    if not SAMPLE_PDF.exists():
        pytest.skip(
            "Sample PDF is missing. Run: python scripts/make_sample_pdf.py"
        )
    return SAMPLE_PDF


@pytest.fixture
def lending_pdf(tmp_path: Path) -> Path:
    """A three-page PDF with one line of predictable text per page."""
    return make_pdf(tmp_path / "lending_policy.pdf", pages=[[line] for line in LENDING_TEXT])
