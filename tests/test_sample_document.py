"""Tests for the sample document generator and its coupling to the eval set.

``data/sample/northwind_lending_policy.pdf`` is the ground truth for
``evaluation/questions.json``. If the generator and the labelled question set
drift apart, the reported metrics silently become meaningless, so these tests
check both the generator itself and the agreement between the two artefacts.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.ingestion.cleaner import TextCleaner
from app.ingestion.pdf_loader import PdfLoader
from scripts import make_sample_pdf
from scripts.evaluate import DEFAULT_QUESTIONS

SAMPLE_PDF = Path("data/sample/northwind_lending_policy.pdf")


def clean(text: str) -> str:
    """Clean raw page text the same way the ingestion pipeline does."""
    return TextCleaner().clean(text)


class TestBuildPdf:
    @pytest.fixture(scope="class")
    def built(self, tmp_path_factory) -> Path:
        return make_sample_pdf.build_pdf(tmp_path_factory.mktemp("sample") / "generated.pdf")

    def test_creates_the_requested_path(self, built: Path) -> None:
        assert built.exists()
        assert built.stat().st_size > 0

    def test_has_the_expected_page_count(self, built: Path) -> None:
        assert len(make_sample_pdf.PAGES) == 7
        assert len(PdfLoader().load(built)) == 7

    def test_metadata_marks_the_document_as_synthetic(self, built: Path) -> None:
        import pymupdf

        with pymupdf.open(built) as document:
            metadata = document.metadata

        assert "synthetic" in metadata["title"].lower()
        assert "synthetic" in metadata["keywords"].lower()

    def test_is_valid_pdf_bytes(self, built: Path) -> None:
        assert built.read_bytes().startswith(b"%PDF-")

    def test_regeneration_is_reproducible_in_text(self, built: Path, tmp_path) -> None:
        second = make_sample_pdf.build_pdf(tmp_path / "again.pdf")

        first_text = [page.text for page in PdfLoader().load(built)]
        second_text = [page.text for page in PdfLoader().load(second)]

        assert first_text == second_text

    def test_every_page_carries_a_heading_and_page_number(self, built: Path) -> None:
        for index, page in enumerate(PdfLoader().load(built), start=1):
            text = page.text
            assert f"Page {index} of 7" in text

    def test_known_facts_are_present_on_the_expected_pages(self, built: Path) -> None:
        pages = {page.page: clean(page.text) for page in PdfLoader().load(built)}

        # Eligibility criteria (page 2) and the interest rate on page 4.
        assert "20 years" in pages[2]
        assert "11.9" in pages[4]
        assert "processing" in pages[4].lower()

    def test_committed_sample_matches_the_generator(self, tmp_path) -> None:
        """The committed PDF must not be stale relative to the script."""
        regenerated = make_sample_pdf.build_pdf(tmp_path / "regenerated.pdf")
        committed = [page.text for page in PdfLoader().load(SAMPLE_PDF)]
        fresh = [page.text for page in PdfLoader().load(regenerated)]

        assert committed == fresh

    def test_main_reports_the_output(self, monkeypatch, capsys) -> None:
        monkeypatch.setattr(make_sample_pdf, "OUTPUT_PATH", Path("data/sample/x.pdf"))
        assert make_sample_pdf.main() == 0
        assert "Wrote" in capsys.readouterr().out
        Path("data/sample/x.pdf").unlink(missing_ok=True)


class TestEvaluationSetAgreement:
    """The labelled questions must still match the sample document."""

    @pytest.fixture(scope="class")
    def questions(self) -> list[dict]:
        return json.loads(DEFAULT_QUESTIONS.read_text(encoding="utf-8"))["questions"]

    def test_set_names_the_committed_sample(self) -> None:
        payload = json.loads(DEFAULT_QUESTIONS.read_text(encoding="utf-8"))
        assert payload["document"] == SAMPLE_PDF.name

    def test_every_expected_keyword_appears_on_the_expected_page(
        self, questions: list[dict]
    ) -> None:
        pages = {page.page: clean(page.text).lower() for page in PdfLoader().load(SAMPLE_PDF)}

        for item in questions:
            page_number = item.get("expected_page")
            if page_number is None:
                continue
            haystack = pages[page_number]
            missing = [kw for kw in item["expected_keywords"] if kw.lower() not in haystack]
            assert not missing, f"{item['id']}: {missing} not on page {page_number}"

    def test_unanswerable_questions_have_no_expected_page(
        self, questions: list[dict]
    ) -> None:
        for item in questions:
            if not item["answerable"]:
                assert item.get("expected_page") is None, item["id"]
                assert not item.get("expected_keywords"), item["id"]
