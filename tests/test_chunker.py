"""Tests for the text chunker."""

from __future__ import annotations

import pytest

from app.ingestion.chunker import TextChunker
from app.ingestion.models import PageDocument, PageMetadata


def make_page(text: str, page: int = 1, document: str = "policy.pdf") -> PageDocument:
    """Build a PageDocument for chunker tests."""
    return PageDocument(
        text=text,
        metadata=PageMetadata(
            document_name=document, source=document, page=page, page_count=10
        ),
    )


class TestSplitText:
    def test_short_text_produces_a_single_chunk(self) -> None:
        chunker = TextChunker(chunk_size=100, chunk_overlap=10)
        chunks = chunker.split_text("The maximum loan tenure is 20 years.")
        assert chunks == ["The maximum loan tenure is 20 years."]

    def test_text_is_split_into_windows_of_the_configured_size(self) -> None:
        text = " ".join(f"word{index}" for index in range(250))
        chunker = TextChunker(chunk_size=100, chunk_overlap=0)
        chunks = chunker.split_text(text)

        assert len(chunks) == 3
        assert all(len(chunk.split()) <= 100 for chunk in chunks)
        # No words are lost: the concatenation covers the whole input.
        assert len(" ".join(chunks).split()) == 250

    def test_overlap_repeats_words_between_neighbouring_chunks(self) -> None:
        text = " ".join(f"word{index}" for index in range(200))
        chunker = TextChunker(chunk_size=100, chunk_overlap=25)
        chunks = chunker.split_text(text)

        assert len(chunks) >= 2
        first = chunks[0].split()
        second = chunks[1].split()
        assert first[-25:] == second[:25], "expected 25 shared words"

    def test_zero_overlap_produces_disjoint_windows(self) -> None:
        text = " ".join(f"word{index}" for index in range(100))
        chunker = TextChunker(chunk_size=40, chunk_overlap=0)
        chunks = chunker.split_text(text)

        assert chunks[0].split()[-1] != chunks[1].split()[0]

    def test_blank_text_yields_no_chunks(self) -> None:
        chunker = TextChunker(chunk_size=50, chunk_overlap=5)
        assert chunker.split_text("   \n\n  ") == []

    def test_sentence_boundary_is_preferred_over_hard_cut(self) -> None:
        text = (
            "Sentence one has ten words here. "
            "Sentence two has ten words here. "
            "Sentence three has ten words here. "
            "Sentence four has ten words here."
        )
        chunker = TextChunker(chunk_size=18, chunk_overlap=0)
        chunks = chunker.split_text(text)

        # The first window snaps to the end of a sentence rather than cutting
        # mid-sentence at exactly 18 words.
        assert chunks[0].endswith(".")

    def test_respect_sentences_can_be_disabled(self) -> None:
        text = " ".join(f"word{index}" for index in range(60))
        chunker = TextChunker(chunk_size=20, chunk_overlap=0, respect_sentences=False)
        chunks = chunker.split_text(text)
        assert len(chunks) == 3

    def test_repetitive_text_is_fully_covered(self) -> None:
        # Regression test: de-duplicating identical windows used to drop
        # content when the source text itself was repetitive.
        text = "word " * 500
        chunker = TextChunker(chunk_size=50, chunk_overlap=5)
        chunks = chunker.split_text(text)

        assert len(chunks) > 1
        assert all(len(chunk.split()) <= 50 for chunk in chunks)
        # Total emitted words must cover the input, overlap included.
        assert sum(len(chunk.split()) for chunk in chunks) >= 500

    def test_very_long_single_sentence_is_still_split(self) -> None:
        text = "word " * 500
        chunker = TextChunker(chunk_size=50, chunk_overlap=5)
        chunks = chunker.split_text(text)
        assert len(chunks) > 1
        assert all(len(chunk.split()) <= 50 for chunk in chunks)

    @pytest.mark.parametrize(
        ("size", "overlap"),
        [(0, 0), (10, 10), (10, 20), (10, -1)],
    )
    def test_invalid_configuration_is_rejected(self, size: int, overlap: int) -> None:
        with pytest.raises(ValueError):
            TextChunker(chunk_size=size, chunk_overlap=overlap)


class TestChunkMetadata:
    def test_chunk_preserves_document_and_page(self) -> None:
        chunker = TextChunker(chunk_size=20, chunk_overlap=5)
        page = make_page(" ".join(f"w{i}" for i in range(50)), page=4)
        chunks = chunker.chunk_page(page)

        assert chunks, "expected at least one chunk"
        for index, chunk in enumerate(chunks):
            assert chunk.page == 4
            assert chunk.document_name == "policy.pdf"
            assert chunk.source == "policy.pdf"
            assert chunk.chunk_index == index
            assert chunk.total_chunks == len(chunks)

    def test_chunk_ids_are_deterministic_across_runs(self) -> None:
        chunker = TextChunker(chunk_size=20, chunk_overlap=5)
        page = make_page("Alpha beta gamma delta epsilon zeta eta theta.", page=2)

        first = [chunk.chunk_id for chunk in chunker.chunk_page(page)]
        second = [chunk.chunk_id for chunk in chunker.chunk_page(page)]

        assert first == second
        assert all("-p0002-" in chunk_id for chunk_id in first)

    def test_chunk_ids_change_when_content_changes(self) -> None:
        chunker = TextChunker(chunk_size=20, chunk_overlap=5)
        before = chunker.chunk_page(make_page("One two three four five six.", page=1))
        after = chunker.chunk_page(make_page("Completely different words here.", page=1))

        assert before[0].chunk_id != after[0].chunk_id

    def test_chunks_never_span_two_pages(self) -> None:
        chunker = TextChunker(chunk_size=30, chunk_overlap=5)
        pages = [
            make_page(" ".join(f"page{page}w{i}" for i in range(40)), page=page)
            for page in (1, 2, 3)
        ]
        chunks = chunker.chunk_pages(pages)

        assert {chunk.page for chunk in chunks} == {1, 2, 3}
        for chunk in chunks:
            assert chunk.text.split()[0].startswith(f"page{chunk.page}")

    def test_metadata_is_serialisable_for_the_vector_store(self) -> None:
        chunker = TextChunker(chunk_size=20, chunk_overlap=5)
        chunk = chunker.chunk_page(make_page("Some text for the chunk."))[0]
        metadata = chunk.to_metadata()

        assert set(metadata) == {
            "chunk_id",
            "document_name",
            "source",
            "page",
            "chunk_index",
            "total_chunks",
            "word_count",
        }
        assert all(
            isinstance(value, (str, int, float, bool)) for value in metadata.values()
        )

    def test_citation_is_human_readable(self) -> None:
        chunker = TextChunker(chunk_size=20, chunk_overlap=5)
        chunk = chunker.chunk_page(make_page("Text here.", page=7))[0]
        assert chunk.citation == "policy.pdf, page 7"

    def test_stats_reports_average_chunk_length(self) -> None:
        chunker = TextChunker(chunk_size=20, chunk_overlap=0)
        page = make_page(" ".join(f"w{i}" for i in range(40)))
        stats = chunker.stats(chunker.chunk_page(page), page_count=1)

        assert stats.chunks == 2
        assert stats.words == 40
        assert stats.average_words_per_chunk == 20.0


class TestChunkerFromSettings:
    def test_values_come_from_configuration(self, settings) -> None:
        chunker = TextChunker.from_settings(settings)
        assert chunker.chunk_size == settings.chunk_size
        assert chunker.chunk_overlap == settings.chunk_overlap

    def test_default_configuration_is_within_the_documented_band(self) -> None:
        from app.core.config import Settings as RealSettings

        defaults = RealSettings()
        assert 500 <= defaults.chunk_size <= 800
        assert 50 <= defaults.chunk_overlap <= 100
        assert defaults.chunk_overlap < defaults.chunk_size
