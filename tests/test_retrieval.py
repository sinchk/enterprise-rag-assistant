"""Tests for semantic retrieval, the vector store and the embedder contract.

The real ChromaDB store is exercised (it is fast and needs no model download)
with a deterministic hashing embedder, so metadata round-trips, persistence and
idempotent upserts are genuinely verified.
"""

from __future__ import annotations

import pytest

from app.core.errors import RetrievalError, VectorStoreError
from app.embeddings.embedder import Embedder, SentenceTransformerEmbedder
from app.ingestion.chunker import TextChunker
from app.ingestion.models import PageDocument, PageMetadata
from app.retrieval.retriever import Retriever
from app.vectorstore.base import VectorStore
from app.vectorstore.chroma_store import ChromaVectorStore, create_chroma_store
from tests.fakes import ExplodingEmbedder, HashingEmbedder

TENURE_TEXT = (
    "Loan eligibility. The maximum loan tenure is 20 years and the minimum is "
    "1 year. Applicants must be at least 21 years old."
)
APR_TEXT = (
    "Interest and fees. The standard annual percentage rate for unsecured "
    "loans is 11.9 percent, charged on the reducing balance."
)
INSURANCE_TEXT = (
    "Insurance. A property insurance policy is required for the full duration "
    "of a secured loan, naming the lender as loss payee."
)


@pytest.fixture
def store(settings) -> ChromaVectorStore:
    """A real ChromaDB store rooted in the temporary directory."""
    return create_chroma_store(settings)


@pytest.fixture
def populated_store(store, settings) -> ChromaVectorStore:
    """A store containing three clearly distinct policy chunks."""
    embedder = HashingEmbedder()
    pages = [
        PageDocument(
            text=TENURE_TEXT,
            metadata=PageMetadata(
                document_name="policy.pdf", source="policy.pdf", page=2, page_count=3
            ),
        ),
        PageDocument(
            text=APR_TEXT,
            metadata=PageMetadata(
                document_name="policy.pdf", source="policy.pdf", page=3, page_count=3
            ),
        ),
        PageDocument(
            text=INSURANCE_TEXT,
            metadata=PageMetadata(
                document_name="other.pdf", source="other.pdf", page=5, page_count=6
            ),
        ),
    ]
    chunks = TextChunker(chunk_size=200, chunk_overlap=10).chunk_pages(pages)
    store.add_documents(chunks, embedder.embed_documents([c.text for c in chunks]))
    return store


class TestChromaVectorStore:
    def test_add_and_count(self, store) -> None:
        chunk = TextChunker(chunk_size=50, chunk_overlap=5).chunk_page(
            PageDocument(
                text="Some indexable text about loan tenure limits.",
                metadata=PageMetadata(
                    document_name="a.pdf", source="a.pdf", page=1, page_count=1
                ),
            )
        )[0]
        written = store.add_documents([chunk], HashingEmbedder().embed_documents([chunk.text]))

        assert written == 1
        assert store.count() == 1

    def test_empty_add_is_a_no_op(self, store) -> None:
        assert store.add_documents([], []) == 0
        assert store.count() == 0

    def test_mismatched_embeddings_are_rejected(self, store) -> None:
        chunk = TextChunker(chunk_size=50, chunk_overlap=5).chunk_page(
            PageDocument(
                text="Text.",
                metadata=PageMetadata(
                    document_name="a.pdf", source="a.pdf", page=1, page_count=1
                ),
            )
        )[0]

        with pytest.raises(VectorStoreError):
            store.add_documents([chunk], [])

    def test_metadata_survives_a_round_trip(self, populated_store) -> None:
        records = populated_store.search(
            HashingEmbedder().embed_text(TENURE_TEXT), top_k=3
        )
        top = records[0]

        assert top["document"] == "policy.pdf"
        assert top["page"] == 2
        assert top["source"] == "policy.pdf"
        assert top["chunk_index"] == 0
        assert "20 years" in top["text"]
        # The slug of "policy.pdf" is "policy-pdf", hence the chunk id prefix.
        assert top["id"].startswith("policy-pdf-p0002-c0000-")
        assert 0.0 < top["score"] <= 1.0

    def test_search_returns_the_closest_chunk_first(self, populated_store) -> None:
        records = populated_store.search(
            HashingEmbedder().embed_text("What is the maximum loan tenure?"), top_k=3
        )

        assert records[0]["page"] == 2
        assert records[0]["score"] >= records[1]["score"]

    def test_top_k_limits_results(self, populated_store) -> None:
        embedder = HashingEmbedder()
        assert len(populated_store.search(embedder.embed_text("loan"), 1)) == 1
        assert len(populated_store.search(embedder.embed_text("loan"), 2)) == 2

    def test_top_k_larger_than_store_returns_everything(self, populated_store) -> None:
        records = populated_store.search(HashingEmbedder().embed_text("loan"), 50)
        assert len(records) == 3

    def test_search_on_empty_store_returns_nothing(self, store) -> None:
        assert store.search(HashingEmbedder().embed_text("anything"), 4) == []

    def test_duplicate_ids_upsert_instead_of_duplicating(
        self, store, settings
    ) -> None:
        embedder = HashingEmbedder()
        pages = [
            PageDocument(
                text=TENURE_TEXT,
                metadata=PageMetadata(
                    document_name="policy.pdf", source="policy.pdf", page=2, page_count=3
                ),
            )
        ]
        chunks = TextChunker(chunk_size=200, chunk_overlap=10).chunk_pages(pages)
        vectors = embedder.embed_documents([c.text for c in chunks])

        store.add_documents(chunks, vectors)
        store.add_documents(chunks, vectors)

        assert store.count() == len(chunks)

    def test_delete_by_document_removes_only_that_document(
        self, populated_store
    ) -> None:
        removed = populated_store.delete_by_document("other.pdf")

        assert removed == 1
        assert populated_store.count() == 2
        assert populated_store.list_documents() == ["policy.pdf"]

    def test_delete_by_unknown_document_is_harmless(self, populated_store) -> None:
        assert populated_store.delete_by_document("missing.pdf") == 0
        assert populated_store.count() == 3

    def test_delete_collection_empties_the_store(self, populated_store) -> None:
        populated_store.delete_collection()

        assert populated_store.count() == 0
        assert populated_store.list_documents() == []

    def test_data_persists_across_store_instances(self, settings) -> None:
        embedder = HashingEmbedder()
        page = PageDocument(
            text=TENURE_TEXT,
            metadata=PageMetadata(
                document_name="policy.pdf", source="policy.pdf", page=2, page_count=3
            ),
        )
        chunks = TextChunker(chunk_size=200, chunk_overlap=10).chunk_pages([page])

        first = create_chroma_store(settings)
        first.add_documents(chunks, embedder.embed_documents([c.text for c in chunks]))

        # A brand new instance, as a restarted process would create.
        second = create_chroma_store(settings)
        assert second.count() == 1
        assert second.search(embedder.embed_text("loan tenure"), 1)[0]["page"] == 2

    def test_persist_directory_is_created(self, settings) -> None:
        assert not settings.chroma_persist_directory.exists()
        create_chroma_store(settings)
        assert settings.chroma_persist_directory.exists()

    def test_store_satisfies_the_protocol(self, populated_store) -> None:
        assert isinstance(populated_store, VectorStore)


class TestRetriever:
    def test_returns_ranked_chunks_with_metadata(self, populated_store) -> None:
        retriever = Retriever(
            HashingEmbedder(), populated_store, top_k=3
        )
        results = retriever.retrieve("What is the maximum loan tenure?")

        assert results, "expected at least one result"
        assert results[0].page == 2
        assert results[0].document == "policy.pdf"
        assert results[0].score > 0
        assert [chunk.rank for chunk in results] == list(range(1, len(results) + 1))
        assert results[0].label == "[S1]"
        assert results[0].citation == "policy.pdf, page 2"

    def test_top_k_can_be_overridden_per_call(self, populated_store) -> None:
        retriever = Retriever(HashingEmbedder(), populated_store, top_k=1)

        assert len(retriever.retrieve("loan", top_k=3)) == 3
        assert len(retriever.retrieve("loan", top_k=1)) == 1

    def test_similarity_threshold_filters_weak_matches(self, populated_store) -> None:
        permissive = Retriever(HashingEmbedder(), populated_store, top_k=3)
        strict = Retriever(
            HashingEmbedder(), populated_store, top_k=3, similarity_threshold=0.99
        )

        assert permissive.retrieve("How many branches are there?")
        assert strict.retrieve("How many branches are there?") == []

    def test_blank_question_is_rejected(self, populated_store) -> None:
        retriever = Retriever(HashingEmbedder(), populated_store, top_k=3)

        with pytest.raises(RetrievalError):
            retriever.retrieve("   ")

    def test_embedder_failure_is_wrapped(self, populated_store) -> None:
        retriever = Retriever(ExplodingEmbedder(), populated_store, top_k=3)

        with pytest.raises(RetrievalError) as excinfo:
            retriever.retrieve("anything")
        # The client sees a safe message, not the internal traceback.
        assert "embedding model" in excinfo.value.message

    def test_empty_store_returns_no_results(self, store) -> None:
        retriever = Retriever(HashingEmbedder(), store, top_k=3)
        assert retriever.retrieve("What is the maximum loan tenure?") == []

    def test_relevant_chunk_is_found_for_its_own_topic(self, populated_store) -> None:
        retriever = Retriever(HashingEmbedder(), populated_store, top_k=1)

        assert retriever.retrieve("annual percentage rate unsecured")[0].page == 3
        assert retriever.retrieve("property insurance loss payee")[0].page == 5

    def test_from_settings_uses_configured_top_k(self, settings, store) -> None:
        retriever = Retriever.from_settings(settings, store=store, embedder=HashingEmbedder())
        assert retriever.top_k == settings.top_k


class TestEmbedderContract:
    def test_hashing_embedder_satisfies_the_protocol(self) -> None:
        assert isinstance(HashingEmbedder(), Embedder)

    def test_real_embedder_satisfies_the_protocol(self, settings) -> None:
        assert isinstance(SentenceTransformerEmbedder(settings), Embedder)

    def test_batch_and_single_embedding_agree(self) -> None:
        embedder = HashingEmbedder()
        texts = ["loan tenure is 20 years", "the annual percentage rate is 11.9"]

        assert embedder.embed_text(texts[0]) == embedder.embed_documents(texts)[0]
        assert len(embedder.embed_documents(texts)) == 2

    def test_empty_batch_returns_empty_list(self) -> None:
        assert HashingEmbedder().embed_documents([]) == []

    def test_blank_text_does_not_crash(self) -> None:
        assert len(HashingEmbedder().embed_text("   ")) == 64

    def test_sentence_transformer_embedder_is_lazy(self, settings) -> None:
        # Constructing the embedder must not download or load the model.
        embedder = SentenceTransformerEmbedder(settings)
        assert embedder._model is None
