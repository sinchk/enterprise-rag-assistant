"""Tests for PDF loading, cleaning and the ingestion pipeline."""

from __future__ import annotations

import pytest

from app.core.errors import EmptyDocumentError, InvalidPdfError
from app.ingestion.chunker import TextChunker
from app.ingestion.cleaner import TextCleaner
from app.ingestion.models import PageDocument, PageMetadata
from app.ingestion.pdf_loader import PdfLoader
from tests.conftest import make_pdf


class TestPdfLoader:
    def test_extracts_text_with_page_numbers(self, lending_pdf) -> None:
        pages = PdfLoader().load(lending_pdf)

        assert len(pages) == 3
        assert [page.page for page in pages] == [1, 2, 3]
        assert "20 years" in pages[0].text
        assert "11.9 percent" in pages[1].text

    def test_preserves_source_metadata(self, lending_pdf) -> None:
        page = PdfLoader().load(lending_pdf)[1]
        metadata = page.metadata.to_dict()

        assert metadata["document_name"] == "lending_policy.pdf"
        assert metadata["source"] == "lending_policy.pdf"
        assert metadata["page"] == 2
        assert metadata["page_count"] == 3
        assert len(metadata["document_name"]) and page.metadata.sha256

    def test_file_hash_is_stable_and_content_dependent(self, tmp_path) -> None:
        first = make_pdf(tmp_path / "a.pdf", pages=[["Same content."]])
        second = make_pdf(tmp_path / "b.pdf", pages=[["Different content."]])

        # Re-reading the same bytes always yields the same hash, and different
        # bytes yield a different one. (Two PDFs with identical text are not
        # byte-identical: the writer embeds a timestamp and a document ID.)
        assert PdfLoader().load(first)[0].metadata.sha256 == (
            PdfLoader().load(first)[0].metadata.sha256
        )
        assert (
            PdfLoader().load(first)[0].metadata.sha256
            != PdfLoader().load(second)[0].metadata.sha256
        )

    def test_missing_file_raises_invalid_pdf(self, tmp_path) -> None:
        with pytest.raises(InvalidPdfError):
            PdfLoader().load(tmp_path / "does_not_exist.pdf")

    def test_non_pdf_extension_is_rejected(self, tmp_path) -> None:
        fake = tmp_path / "notes.txt"
        fake.write_text("plain text", encoding="utf-8")

        with pytest.raises(InvalidPdfError):
            PdfLoader().load(fake)

    def test_corrupt_file_raises_invalid_pdf(self, tmp_path) -> None:
        corrupt = tmp_path / "corrupt.pdf"
        corrupt.write_bytes(b"this is definitely not a pdf")

        with pytest.raises(InvalidPdfError):
            PdfLoader().load(corrupt)

    def test_pdf_without_text_layer_raises_empty_document(self, tmp_path) -> None:
        # A single blank page mimics a scanned document: no text, so OCR would
        # be required. This is an expected, documented failure.
        scanned = make_pdf(tmp_path / "scanned.pdf", blank=True)

        with pytest.raises(EmptyDocumentError) as excinfo:
            PdfLoader().load(scanned)
        assert "OCR" in excinfo.value.message

    def test_max_pages_limits_extraction(self, lending_pdf) -> None:
        pages = PdfLoader(max_pages=2).load(lending_pdf)
        assert len(pages) == 2

    def test_real_sample_document_loads(self, sample_pdf) -> None:
        pages = PdfLoader().load(sample_pdf)

        assert len(pages) == 7
        assert all(page.text.strip() for page in pages)
        assert all(page.page == index + 1 for index, page in enumerate(pages))


class TestTextCleaner:
    def test_collapses_excessive_whitespace(self) -> None:
        assert TextCleaner().clean("a    b\t\tc") == "a b c"

    def test_collapses_repeated_blank_lines(self) -> None:
        assert TextCleaner().clean("first\n\n\n\n\nsecond") == "first\n\nsecond"

    def test_rejoins_hyphenated_line_breaks(self) -> None:
        # Extraction artefacts: "inter-\nnational" reads as "international".
        assert TextCleaner().clean("an inter-\nnational policy") == (
            "an international policy"
        )

    def test_hyphen_repair_merges_compound_words(self) -> None:
        # Documented trade-off: without a dictionary, a compound split across a
        # line ("well-\nknown") is also merged ("wellknown"). De-hyphenation is
        # kept because split words break both embedding quality and keyword
        # search far more often than wrapped compounds do.
        assert TextCleaner().clean("a well-\nknown rule") == "a wellknown rule"

    def test_hyphen_repair_ignores_hyphens_inside_a_line(self) -> None:
        assert TextCleaner().clean("a well-known rule") == "a well-known rule"

    def test_normalises_unicode_whitespace_and_quotes(self) -> None:
        cleaned = TextCleaner().clean("The\u00a0policy\u2019s\u200b \u201cscope\u201d")
        assert cleaned == "The policy's \"scope\""

    def test_removes_control_characters(self) -> None:
        assert TextCleaner().clean("safe\x00\x07text") == "safe text"

    def test_drops_lines_without_alphanumeric_content(self) -> None:
        cleaned = TextCleaner().clean("keep this\n\u2022\n\u2588\u2588\nkeep that")

        # Artefact glyphs are gone; the two real lines survive. They are not
        # separated by a blank line, so the cleaner folds them into one
        # paragraph, which is the correct reading of consecutive PDF lines.
        assert "\u2022" not in cleaned
        assert "\u2588" not in cleaned
        assert cleaned == "keep this keep that"

    def test_drops_single_symbol_lines_but_keeps_short_text(self) -> None:
        assert TextCleaner(min_line_words=1).clean("Fig 1 shows it") == "Fig 1 shows it"

    def test_joins_wrapped_lines_but_keeps_paragraphs(self) -> None:
        cleaned = TextCleaner().clean("first line of a\nsentence wraps here\n\nnew para")
        assert cleaned == "first line of a sentence wraps here\n\nnew para"

    def test_wrapped_line_joining_can_be_disabled(self) -> None:
        cleaned = TextCleaner(join_wrapped_lines=False).clean("alpha\nbeta")
        assert cleaned == "alpha\nbeta"

    def test_removes_space_before_punctuation(self) -> None:
        assert TextCleaner().clean("a value , then b .") == "a value, then b."

    def test_blank_input_returns_empty_string(self) -> None:
        assert TextCleaner().clean("   \n  ") == ""

    def test_strips_repeated_headers_and_page_numbers(self) -> None:
        pages = [
            PageDocument(
                text=f"ACME Confidential Report\nSection {index}\nPage {index} of 4\n"
                f"Body text for section {index}.",
                metadata=PageMetadata(
                    document_name="report.pdf",
                    source="report.pdf",
                    page=index,
                    page_count=4,
                ),
            )
            for index in range(1, 5)
        ]

        cleaned = TextCleaner().clean_pages(pages)

        assert all("ACME Confidential Report" not in page.text for page in cleaned)
        assert all("Page 1 of 4" not in page.text for page in cleaned)
        assert all("Body text" in page.text for page in cleaned)
        assert all("Section" in page.text for page in cleaned)

    def test_keeps_page_numbers_when_the_document_is_short(self) -> None:
        # Below _MARGINALIA_MIN_PAGES the heuristic stays out of the way, so a
        # short document never loses content.
        pages = [
            PageDocument(
                text=f"Page {index} of 2\nBody {index}.",
                metadata=PageMetadata(
                    document_name="r.pdf", source="r.pdf", page=index, page_count=2
                ),
            )
            for index in (1, 2)
        ]
        cleaned = TextCleaner().clean_pages(pages)

        assert len(cleaned) == 2
        assert "Page 1 of 2" in cleaned[0].text
        assert "Page 2 of 2" in cleaned[1].text
        assert "Body 1." in cleaned[0].text

    def test_preserves_unique_content_when_marginalia_is_disabled(self) -> None:
        pages = [
            PageDocument(
                text="Running header\nUnique body one.",
                metadata=PageMetadata(
                    document_name="r.pdf", source="r.pdf", page=1, page_count=1
                ),
            )
        ]
        cleaned = TextCleaner(strip_repeated_marginalia=False).clean_pages(pages)
        assert "Running header" in cleaned[0].text

    def test_empty_pages_are_dropped(self) -> None:
        pages = [
            PageDocument(
                text="Real content here.",
                metadata=PageMetadata(
                    document_name="r.pdf", source="r.pdf", page=1, page_count=2
                ),
            ),
            PageDocument(
                text="\u2022\u2022",
                metadata=PageMetadata(
                    document_name="r.pdf", source="r.pdf", page=2, page_count=2
                ),
            ),
        ]
        assert len(TextCleaner().clean_pages(pages)) == 1

    def test_real_sample_document_is_cleaned(self, sample_pdf) -> None:
        pages = TextCleaner().clean_pages(PdfLoader().load(sample_pdf))
        joined = " ".join(page.text for page in pages)

        assert "Page 1 of 7" not in joined
        assert "Consumer Lending Policy (SYNTHETIC DEMO DATA)" not in joined
        assert "20 years" in joined
        assert "250,000" in joined


class TestIngestionStagesCombined:
    def test_load_clean_and_chunk_the_sample_document(self, sample_pdf) -> None:
        pages = TextCleaner().clean_pages(PdfLoader().load(sample_pdf))
        chunks = TextChunker(chunk_size=80, chunk_overlap=10).chunk_pages(pages)

        # Seven pages, each around 100-140 words, so an 80-word window splits
        # some pages into two chunks. Every page must still be represented.
        assert len(pages) == 7
        assert len(chunks) >= len(pages)
        assert {chunk.page for chunk in chunks} == set(range(1, 8))
        assert all(chunk.text.strip() for chunk in chunks)
        assert all(chunk.word_count > 0 for chunk in chunks)
        assert all(
            chunk.chunk_id.startswith("northwind-lending-policy-") for chunk in chunks
        )
