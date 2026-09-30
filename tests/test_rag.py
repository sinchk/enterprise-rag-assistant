"""Tests for the RAG pipeline and the grounding prompt.

The embedder and LLM are doubles, so these tests verify the pipeline's own
logic (ingestion wiring, the refusal short-circuit, source attribution) without
any network access or API key.
"""

from __future__ import annotations

import pytest

from app.ingestion.chunker import TextChunker
from app.ingestion.cleaner import TextCleaner
from app.ingestion.pdf_loader import PdfLoader
from app.llm.client import (
    ExtractiveBaselineClient,
    NO_CONTEXT_ANSWER,
    get_llm_client,
)
from app.core.errors import EmptyDocumentError, InvalidPdfError, LLMConfigurationError
from app.rag.pipeline import RAGPipeline
from app.rag.prompts import (
    NO_ANSWER_MESSAGE,
    SYSTEM_PROMPT,
    build_context,
    build_messages,
    referenced_sources,
)
from app.retrieval.models import RetrievedChunk
from app.retrieval.retriever import Retriever
from app.vectorstore.chroma_store import create_chroma_store
from tests.fakes import ExplodingLLM, HashingEmbedder, ScriptedLLM, make_chunk

TENURE_TEXT = (
    "Loan eligibility. The maximum loan tenure is 20 years and the minimum is "
    "1 year. Applicants must be at least 21 years old."
)
APR_TEXT = (
    "Interest and fees. The standard annual percentage rate for unsecured "
    "loans is 11.9 percent, charged on the reducing balance."
)


@pytest.fixture
def pipeline(settings, tmp_path) -> RAGPipeline:
    """A fully wired pipeline with a real store and a scripted LLM."""
    store = create_chroma_store(settings)
    embedder = HashingEmbedder()
    return RAGPipeline(
        settings=settings,
        store=store,
        embedder=embedder,
        retriever=Retriever(embedder, store, top_k=settings.top_k),
        llm=ScriptedLLM(default="The maximum loan tenure is 20 years [S1]."),
        chunker=TextChunker.from_settings(settings),
    )


@pytest.fixture
def sample_doc(tmp_path) -> "object":
    """A one-page PDF with the tenure and APR facts on separate pages."""
    from tests.conftest import make_pdf

    return make_pdf(
        tmp_path / "policy.pdf",
        pages=[[TENURE_TEXT], [APR_TEXT]],
    )


class TestIngestion:
    def test_ingest_returns_useful_statistics(self, pipeline, sample_doc) -> None:
        result = pipeline.ingest_document(sample_doc)

        assert result.document == "policy.pdf"
        assert result.pages_processed == 2
        assert result.chunks_created == 2
        assert result.total_words > 0
        assert result.average_chunk_words > 0
        assert result.duration_ms >= 0

    def test_ingested_chunks_are_retrievable(self, pipeline, sample_doc) -> None:
        pipeline.ingest_document(sample_doc)

        results = pipeline.retriever.retrieve("What is the maximum loan tenure?")
        assert results
        assert results[0].page == 1
        assert "20 years" in results[0].text

    def test_reingesting_the_same_document_does_not_duplicate(
        self, pipeline, sample_doc
    ) -> None:
        first = pipeline.ingest_document(sample_doc)
        second = pipeline.ingest_document(sample_doc)

        assert first.chunks_created == second.chunks_created
        assert second.chunks_replaced == first.chunks_created
        assert pipeline.store.count() == first.chunks_created

    def test_reingesting_a_changed_document_replaces_stale_chunks(
        self, pipeline, tmp_path
    ) -> None:
        from tests.conftest import make_pdf

        original = make_pdf(tmp_path / "policy.pdf", pages=[[TENURE_TEXT]])
        pipeline.ingest_document(original)

        revised = make_pdf(
            tmp_path / "policy.pdf",
            pages=[["This version was rewritten and no longer mentions tenure."]],
        )
        pipeline.ingest_document(revised)

        assert pipeline.store.count() == 1
        assert "20 years" not in pipeline.store.search(
            HashingEmbedder().embed_text("loan tenure"), 1
        )[0]["text"]

    def test_indexed_documents_are_listed(self, pipeline, sample_doc) -> None:
        pipeline.ingest_document(sample_doc)
        assert pipeline.indexed_documents() == ["policy.pdf"]

    def test_invalid_document_propagates_a_domain_error(self, pipeline, tmp_path) -> None:
        broken = tmp_path / "broken.pdf"
        broken.write_bytes(b"not really a pdf")

        with pytest.raises(InvalidPdfError):
            pipeline.ingest_document(broken)

    def test_scanned_document_propagates_empty_document(self, pipeline, tmp_path) -> None:
        from tests.conftest import make_pdf

        scanned = make_pdf(tmp_path / "scanned.pdf", blank=True)

        with pytest.raises(EmptyDocumentError):
            pipeline.ingest_document(scanned)

    def test_document_name_is_used_for_citations(self, pipeline, tmp_path) -> None:
        from tests.conftest import make_pdf

        path = make_pdf(tmp_path / "annual_report.pdf", pages=[[TENURE_TEXT]])
        pipeline.ingest_document(path)

        result = pipeline.ask("What is the maximum loan tenure?")
        assert result.sources[0]["document"] == "annual_report.pdf"


class TestQuestionAnswering:
    def test_answer_includes_the_cited_source_page(self, pipeline, sample_doc) -> None:
        pipeline.ingest_document(sample_doc)
        result = pipeline.ask("What is the maximum loan tenure?")

        assert "20 years" in result.answer
        assert result.sources, "expected at least one source"
        assert result.sources[0]["document"] == "policy.pdf"
        assert result.sources[0]["page"] == 1
        assert result.chunks_used > 0
        assert result.top_score is not None
        assert result.no_answer is False

    def test_prompt_contains_context_labels_and_the_question(
        self, pipeline, sample_doc
    ) -> None:
        pipeline.ingest_document(sample_doc)
        pipeline.ask("What is the maximum loan tenure?")

        prompt = pipeline.llm.last_user_prompt
        assert "[S1] document: policy.pdf | page: 1" in prompt
        assert "20 years" in prompt
        assert "What is the maximum loan tenure?" in prompt

    def test_no_context_short_circuits_without_calling_the_llm(
        self, settings, sample_doc
    ) -> None:
        # An empty store means there is nothing to ground an answer in.
        store = create_chroma_store(settings)
        llm = ScriptedLLM(default="should never be used")
        embedder = HashingEmbedder()
        pipeline = RAGPipeline(
            settings=settings,
            store=store,
            embedder=embedder,
            retriever=Retriever(embedder, store, top_k=3),
            llm=llm,
        )

        result = pipeline.ask("What is the maximum loan tenure?")

        assert result.answer == NO_ANSWER_MESSAGE
        assert result.no_answer is True
        assert result.sources == []
        assert result.chunks_used == 0
        assert llm.prompts == [], "the LLM must not be called without context"

    def test_model_refusal_is_flagged(self, pipeline, sample_doc) -> None:
        pipeline.ingest_document(sample_doc)
        pipeline.llm._responses = [NO_ANSWER_MESSAGE]

        result = pipeline.ask("How many branches does Northwind operate?")

        assert result.no_answer is True
        assert result.answer == NO_ANSWER_MESSAGE

    def test_llm_failure_propagates(self, settings, sample_doc) -> None:
        store = create_chroma_store(settings)
        embedder = HashingEmbedder()
        pipeline = RAGPipeline(
            settings=settings,
            store=store,
            embedder=embedder,
            retriever=Retriever(embedder, store, top_k=3),
            llm=ExplodingLLM(),
        )
        pipeline.ingest_document(sample_doc)

        with pytest.raises(RuntimeError):
            pipeline.ask("What is the maximum loan tenure?")

    def test_blank_question_is_rejected(self, pipeline, sample_doc) -> None:
        pipeline.ingest_document(sample_doc)
        with pytest.raises(Exception):
            pipeline.ask("   ")

    def test_top_k_override_is_passed_through(self, pipeline, sample_doc) -> None:
        pipeline.ingest_document(sample_doc)
        assert pipeline.ask("What is the maximum loan tenure?", top_k=1).chunks_used == 1


class TestPrompts:
    def test_system_prompt_states_the_grounding_contract(self) -> None:
        assert "ONLY the information in the CONTEXT" in SYSTEM_PROMPT
        assert "Never use prior knowledge" in SYSTEM_PROMPT
        assert NO_ANSWER_MESSAGE in SYSTEM_PROMPT
        assert "Source:" in SYSTEM_PROMPT

    def test_build_context_labels_every_chunk_with_its_page(self) -> None:
        chunks = [make_chunk(4, "First fact.", rank=1), make_chunk(9, "Second fact.", rank=2)]
        context = build_context(chunks)

        assert "[S1] document: test.pdf | page: 4" in context
        assert "[S2] document: test.pdf | page: 9" in context
        assert "First fact." in context and "Second fact." in context

    def test_build_context_respects_the_character_budget(self) -> None:
        chunks = [make_chunk(index, "x" * 400, rank=index) for index in range(1, 6)]
        context = build_context(chunks, max_chars=900)

        assert len(context) < 2000
        # The highest ranked chunk is always kept.
        assert "[S1]" in context

    def test_build_context_with_no_chunks_is_empty(self) -> None:
        assert build_context([]) == ""

    def test_messages_have_a_system_and_a_user_turn(self) -> None:
        messages = build_messages("What is the tenure?", [make_chunk(2, "Fact.")])

        assert [m.role for m in messages] == ["system", "user"]
        assert messages[0].content == SYSTEM_PROMPT
        assert "What is the tenure?" in messages[1].content

    def test_referenced_sources_follows_citation_tags(self) -> None:
        chunks = [make_chunk(1, "A", rank=1), make_chunk(2, "B", rank=2)]

        assert [c.page for c in referenced_sources("Fact from [S2].", chunks)] == [2]
        assert [c.page for c in referenced_sources("Both [S1] and [S2].", chunks)] == [1, 2]

    def test_referenced_sources_falls_back_to_all_chunks(self) -> None:
        chunks = [make_chunk(1, "A", rank=1), make_chunk(2, "B", rank=2)]
        assert referenced_sources("No tags here.", chunks) == chunks

    def test_unknown_citation_tags_are_ignored(self) -> None:
        chunks = [make_chunk(1, "A", rank=1)]
        assert [c.page for c in referenced_sources("Citing [S9].", chunks)] == [1]


class TestLlmProviders:
    def test_missing_api_key_fails_with_a_clear_message(self, settings) -> None:
        from app.core.config import Settings

        unconfigured = Settings(llm_provider="openai", openai_api_key=None)
        with pytest.raises(LLMConfigurationError) as excinfo:
            get_llm_client(unconfigured)

        assert "OPENAI_API_KEY" in excinfo.value.message
        assert excinfo.value.status_code == 503

    def test_extractive_provider_is_selected_by_configuration(self, settings) -> None:
        assert isinstance(get_llm_client(settings), ExtractiveBaselineClient)

    def test_extractive_provider_only_copies_context_sentences(self) -> None:
        client = ExtractiveBaselineClient()
        messages = build_messages(
            "What is the maximum loan tenure?",
            [make_chunk(2, "The maximum loan tenure is 20 years. Other text.")],
        )
        answer = client.complete(messages)

        # Only the sentence with the highest overlap is returned, and it is
        # copied verbatim: no question text, no invented wording.
        assert answer == "The maximum loan tenure is 20 years."
        assert "?" not in answer
        assert "Other text." not in answer

    def test_extractive_provider_can_return_several_tied_sentences(self) -> None:
        client = ExtractiveBaselineClient()
        messages = build_messages(
            "What is the maximum loan tenure?",
            [
                make_chunk(
                    2,
                    "The maximum loan tenure is 20 years. "
                    "The maximum loan tenure cannot be extended. "
                    "Something unrelated.",
                )
            ],
        )
        answer = client.complete(messages)

        assert "20 years" in answer
        assert "cannot be extended" in answer
        assert "unrelated" not in answer

    def test_extractive_provider_abstains_without_matching_evidence(self) -> None:
        client = ExtractiveBaselineClient()
        messages = build_messages(
            "Which stadium do the zebras play in?",
            [make_chunk(2, "The maximum loan tenure is 20 years. Other text.")],
        )

        assert client.complete(messages) == NO_CONTEXT_ANSWER

    def test_extractive_provider_reports_its_non_llm_identity(self) -> None:
        assert "non-LLM" in ExtractiveBaselineClient().model_name

    def test_pipeline_model_label_works_without_an_api_key(self, settings) -> None:
        from app.core.config import Settings

        pipeline = RAGPipeline(
            settings=Settings(llm_provider="openai", openai_api_key=None),
            store=create_chroma_store(settings),
            embedder=HashingEmbedder(),
        )
        assert pipeline.model_label == "gpt-4o-mini"
        # Touching .llm raises, but model_label never does.
        with pytest.raises(LLMConfigurationError):
            _ = pipeline.llm


class TestDefaultComponents:
    def test_pipeline_uses_configured_defaults(self, settings) -> None:
        from app.core.config import Settings

        strict = Settings(
            chroma_persist_directory=settings.chroma_persist_directory,
            chroma_collection=settings.chroma_collection,
            chunk_size=123,
            chunk_overlap=45,
        )
        pipeline = RAGPipeline(
            settings=strict,
            store=create_chroma_store(settings),
            embedder=HashingEmbedder(),
            llm=ScriptedLLM(),
        )

        assert pipeline.chunker.chunk_size == 123
        assert pipeline.chunker.chunk_overlap == 45
        assert isinstance(pipeline.loader, PdfLoader)
        assert isinstance(pipeline.cleaner, TextCleaner)

    def test_retrieved_chunk_source_shape(self) -> None:
        chunk = make_chunk(6, "Fact.")
        source = chunk.to_source()

        assert set(source) == {"document", "page", "score", "chunk_id"}
        assert isinstance(source["score"], float)


def test_retrieved_chunk_is_immutable() -> None:
    chunk: RetrievedChunk = make_chunk(1, "Fact.")
    with pytest.raises(Exception):
        chunk.page = 2  # type: ignore[misc]
