"""End-to-end checks against the real sentence-transformer model.

The other suites use deterministic doubles so they run in milliseconds. These
tests use the real embedding model, so they are slower and require the model
files to be available locally. They are skipped when the model cannot be
loaded, which keeps the default suite usable offline.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.core.config import Settings
from app.embeddings.embedder import SentenceTransformerEmbedder
from app.rag.pipeline import RAGPipeline
from app.retrieval.retriever import Retriever
from app.vectorstore.chroma_store import create_chroma_store
from tests.fakes import ScriptedLLM

MODEL = "sentence-transformers/all-MiniLM-L6-v2"
SAMPLE_PDF = "data/sample/northwind_lending_policy.pdf"


@pytest.fixture(scope="module")
def model_settings() -> Settings:
    """Settings pinned to the real model and the CPU device."""
    return Settings(
        llm_provider="extractive",
        embedding_model=MODEL,
        embedding_device="cpu",
        embedding_batch_size=8,
    )


@pytest.fixture(scope="module")
def embedder(model_settings) -> SentenceTransformerEmbedder:
    """Load the real model once, or skip if it is unavailable."""
    embedder = SentenceTransformerEmbedder(model_settings)
    try:
        embedder.dimension
    except Exception as exc:  # pragma: no cover - depends on the environment
        pytest.skip(f"embedding model unavailable: {exc}")
    return embedder


@pytest.fixture
def settings(tmp_path, model_settings) -> Settings:
    """Model settings redirected at a temporary store."""
    return model_settings.model_copy(
        update={
            "chroma_persist_directory": tmp_path / "chroma",
            "upload_directory": tmp_path / "uploads",
        }
    )


class TestRealEmbedder:
    def test_embedding_shape_and_normalisation(self, embedder) -> None:
        vectors = embedder.embed_documents(
            ["maximum loan tenure", "annual percentage rate"]
        )

        assert len(vectors) == 2
        assert all(len(vector) == embedder.dimension for vector in vectors)
        # Sentence-transformers returns L2-normalised vectors, which is what makes
        # the cosine similarity in Chroma equal to a dot product.
        for vector in vectors:
            assert sum(component * component for component in vector) == pytest.approx(
                1.0, abs=1e-4
            )

    def test_dimension_matches_the_model(self, embedder) -> None:
        assert embedder.dimension == 384

    def test_similar_text_is_closer_than_unrelated_text(self, embedder) -> None:
        query, near, far = embedder.embed_documents(
            ["maximum loan tenure", "The maximum tenure is 20 years.", "Banana bread recipe"]
        )

        def cosine(a, b) -> float:
            return sum(x * y for x, y in zip(a, b))

        assert cosine(query, near) > cosine(query, far)

    def test_empty_input_returns_no_vectors(self, embedder) -> None:
        assert embedder.embed_documents([]) == []

    def test_one_vector_per_input_keeps_alignment(self, embedder) -> None:
        vectors = embedder.embed_documents(["first fact", "   ", "third fact"])
        assert len(vectors) == 3


class TestRealPipelineEndToEnd:
    def test_ingest_retrieve_and_answer(self, settings, embedder) -> None:
        store = create_chroma_store(settings)
        pipeline = RAGPipeline(
            settings=settings,
            store=store,
            embedder=embedder,
            retriever=Retriever(embedder, store, top_k=settings.top_k),
            llm=ScriptedLLM(default="The maximum loan tenure is 20 years [S1]."),
        )

        ingestion = pipeline.ingest_document(SAMPLE_PDF)
        assert ingestion.pages_processed == 7
        assert ingestion.chunks_created > 0
        assert store.count() == ingestion.chunks_created

        answer = pipeline.ask("What is the maximum loan tenure?")

        assert "20 years" in answer.answer
        assert answer.sources, "a grounded answer must cite at least one source"
        assert answer.top_score is not None and answer.top_score > 0
        # The tenure fact lives on the eligibility page of the sample document.
        assert answer.sources[0]["page"] == 2
        assert answer.sources[0]["document"] == Path(SAMPLE_PDF).name

    def test_reingest_is_idempotent(self, settings, embedder) -> None:
        store = create_chroma_store(settings)
        pipeline = RAGPipeline(
            settings=settings,
            store=store,
            embedder=embedder,
            retriever=Retriever(embedder, store, top_k=settings.top_k),
            llm=ScriptedLLM(default="Answer [S1]."),
        )

        first = pipeline.ingest_document(SAMPLE_PDF)
        second = pipeline.ingest_document(SAMPLE_PDF)

        assert second.chunks_created == first.chunks_created
        assert second.chunks_replaced == first.chunks_created
        assert store.count() == first.chunks_created

    def test_unrelated_question_finds_nothing(self, settings, embedder) -> None:
        store = create_chroma_store(settings)
        pipeline = RAGPipeline(
            settings=settings,
            store=store,
            embedder=embedder,
            retriever=Retriever(embedder, store, top_k=settings.top_k),
            llm=ScriptedLLM(default="Answer [S1]."),
        )
        pipeline.ingest_document(SAMPLE_PDF)

        chunks = pipeline.retriever.retrieve("How do I bake sourdough bread at altitude?")

        # A lending policy is not about bread, so every similarity must be low.
        assert all(chunk.score < 0.5 for chunk in chunks)

    def test_store_survives_a_restart(self, settings, embedder) -> None:
        first_store = create_chroma_store(settings)
        RAGPipeline(
            settings=settings,
            store=first_store,
            embedder=embedder,
            retriever=Retriever(embedder, first_store, top_k=4),
            llm=ScriptedLLM(default="Answer [S1]."),
        ).ingest_document(SAMPLE_PDF)
        chunks_written = first_store.count()
        # A brand new store object must read the data written by the first one.
        reopened = create_chroma_store(settings)

        assert reopened.count() == chunks_written
        # Documents are keyed by file name, not by the path they were read from.
        assert reopened.list_documents() == [Path(SAMPLE_PDF).name]
