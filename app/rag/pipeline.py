"""The RAG pipeline orchestrating ingestion, retrieval and generation.

    PDF -> parse -> clean -> chunk -> embed -> ChromaDB
    question -> embed -> search -> top-k chunks -> prompt -> LLM -> answer

The class owns no framework abstraction of its own: it receives a loader,
cleaner, chunker, embedder, vector store, retriever and LLM client, so each
piece can be replaced or mocked independently (which is how the test suite
runs without a paid API).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.core.config import Settings, get_settings
from app.core.errors import RetrievalError
from app.embeddings.embedder import Embedder, SentenceTransformerEmbedder
from app.ingestion.chunker import TextChunker
from app.ingestion.cleaner import TextCleaner
from app.ingestion.pdf_loader import PdfLoader
from app.llm.client import LLMClient, describe_provider, get_llm_client
from app.rag.prompts import NO_ANSWER_MESSAGE, build_messages, referenced_sources
from app.retrieval.retriever import Retriever
from app.vectorstore.base import VectorStore
from app.vectorstore.chroma_store import get_chroma_store

logger = logging.getLogger(__name__)

# Substring of the "not found" reply produced by NO_ANSWER_MESSAGE.
NO_ANSWER_PHRASE = "could not find this information"


@dataclass(frozen=True, slots=True)
class IngestionResult:
    """Outcome of ingesting a single document."""

    document: str
    pages_processed: int
    chunks_created: int
    chunks_replaced: int = 0
    total_words: int = 0
    average_chunk_words: float = 0.0
    duration_ms: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable summary for the API response."""
        return {
            "document": self.document,
            "pages_processed": self.pages_processed,
            "chunks_created": self.chunks_created,
            "chunks_replaced": self.chunks_replaced,
            "total_words": self.total_words,
            "average_chunk_words": round(self.average_chunk_words, 1),
            "duration_ms": self.duration_ms,
        }


@dataclass(frozen=True, slots=True)
class AnswerResult:
    """A grounded answer plus the evidence it came from."""

    answer: str
    sources: list[dict[str, Any]] = field(default_factory=list)
    no_answer: bool = False
    chunks_used: int = 0
    model: str = ""
    top_score: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view of the answer."""
        return {
            "answer": self.answer,
            "sources": self.sources,
            "no_answer": self.no_answer,
            "chunks_used": self.chunks_used,
            "model": self.model,
            "top_score": None if self.top_score is None else round(self.top_score, 4),
        }


class RAGPipeline:
    """Compose the ingestion and question-answering stages."""

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        loader: PdfLoader | None = None,
        cleaner: TextCleaner | None = None,
        chunker: TextChunker | None = None,
        embedder: Embedder | None = None,
        store: VectorStore | None = None,
        retriever: Retriever | None = None,
        llm: LLMClient | None = None,
    ) -> None:
        """Wire the pipeline, defaulting every stage from configuration."""
        self.settings = settings or get_settings()
        self.loader = loader or PdfLoader()
        self.cleaner = cleaner or TextCleaner()
        self.chunker = chunker or TextChunker.from_settings(self.settings)
        self.embedder = embedder or SentenceTransformerEmbedder(self.settings)
        self.store = store or get_chroma_store()
        self.retriever = retriever or Retriever.from_settings(
            self.settings, store=self.store, embedder=self.embedder
        )
        # The LLM client is built on first use: /health and /ingest must work
        # even when no API key is configured, while /ask reports the missing
        # configuration as a 503 through LLMConfigurationError.
        self._llm = llm

    @property
    def llm(self) -> LLMClient:
        """The LLM client, created on first access.

        Raises:
            LLMConfigurationError: If the provider is unconfigured (for example
                a missing ``OPENAI_API_KEY``).
        """
        if self._llm is None:
            self._llm = get_llm_client(self.settings)
        return self._llm

    @property
    def model_label(self) -> str:
        """Model name for responses, available without an API key."""
        if self._llm is not None:
            return self._llm.model_name
        return describe_provider(self.settings)

    # -- ingestion ---------------------------------------------------------

    def ingest_document(self, pdf_path: str | Path) -> IngestionResult:
        """Run the full ingestion pipeline for a single PDF.

        The document is parsed, cleaned, chunked, embedded and stored. Chunks
        previously stored for the same document name are removed first, so
        re-ingesting an updated file does not leave stale content behind.

        Raises:
            InvalidPdfError, EmptyDocumentError: Propagated from the loader.
            VectorStoreError: If the chunks cannot be persisted.
        """
        started = time.perf_counter()
        path = Path(pdf_path)

        pages = self.loader.load(path)
        cleaned_pages = self.cleaner.clean_pages(pages)
        chunks = self.chunker.chunk_pages(cleaned_pages)
        if not chunks:
            raise ValueError(
                f"No indexable text was produced from '{path.name}' after cleaning."
            )

        embeddings = self.embedder.embed_documents([chunk.text for chunk in chunks])
        replaced = self.store.delete_by_document(path.name)
        created = self.store.add_documents(chunks, embeddings)

        stats = self.chunker.stats(chunks, page_count=len(cleaned_pages))
        result = IngestionResult(
            document=path.name,
            pages_processed=len(cleaned_pages),
            chunks_created=created,
            chunks_replaced=replaced,
            total_words=stats.words,
            average_chunk_words=stats.average_words_per_chunk,
            duration_ms=int((time.perf_counter() - started) * 1000),
        )
        logger.info(
            "Ingested '%s': %s page(s), %s chunk(s) in %s ms.",
            result.document,
            result.pages_processed,
            result.chunks_created,
            result.duration_ms,
        )
        return result

    # -- question answering ------------------------------------------------

    def ask(self, question: str, top_k: int | None = None) -> AnswerResult:
        """Answer a question strictly from the indexed documents.

        When retrieval returns nothing (empty store, or every chunk below the
        similarity threshold) the model is not called at all and a fixed
        "not found" answer is returned. This is the strongest guarantee in the
        system that missing information is never invented.

        Raises:
            RetrievalError: If the question is blank or retrieval fails.
            LLMConfigurationError / LLMProviderError: From the LLM client.
        """
        chunks = self.retriever.retrieve(question, top_k=top_k)

        if not chunks:
            logger.info(
                "No supporting context found for question; returning a "
                "'not found' answer without calling the LLM."
            )
            return AnswerResult(
                answer=NO_ANSWER_MESSAGE,
                sources=[],
                no_answer=True,
                chunks_used=0,
                model=self.model_label,
                top_score=None,
            )

        messages = build_messages(
            question, chunks, max_context_chars=self.settings.llm_max_context_chars
        )
        answer = self.llm.complete(messages)
        sources = referenced_sources(answer, chunks)

        return AnswerResult(
            answer=answer,
            sources=[chunk.to_source() for chunk in sources],
            no_answer=self._states_no_answer(answer),
            chunks_used=len(chunks),
            model=self.model_label,
            top_score=chunks[0].score,
        )

    # -- maintenance -------------------------------------------------------

    def indexed_documents(self) -> list[str]:
        """Return the names of the documents currently indexed."""
        return self.store.list_documents()

    @staticmethod
    def _states_no_answer(answer: str) -> bool:
        """Whether the reply states the information was not found."""
        normalised = " ".join(answer.lower().split())
        return NO_ANSWER_PHRASE in normalised


def get_pipeline(settings: Settings | None = None) -> RAGPipeline:
    """Build a pipeline using process-wide configuration and the shared store."""
    return RAGPipeline(settings=settings or get_settings())


# Re-exported so callers can catch pipeline errors without importing the module.
__all__ = ["AnswerResult", "IngestionResult", "RAGPipeline", "RetrievalError", "get_pipeline"]
