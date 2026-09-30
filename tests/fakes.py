"""Test doubles used across the suite.

The doubles let the tests exercise real ingestion, real chunking and the real
ChromaDB store without downloading an embedding model or calling a paid LLM.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Sequence
from typing import Any

from app.llm.client import LLMMessage
from app.retrieval.models import RetrievedChunk

_TOKEN_RE = re.compile(r"[a-z0-9]+")


class HashingEmbedder:
    """Deterministic bag-of-words embedder.

    Each token is hashed into a bucket and the counts are L2-normalised, so two
    sentences that share words have a higher cosine similarity than unrelated
    ones. This is not a semantic model, but it behaves predictably, needs no
    download and keeps the test suite fast and offline.
    """

    def __init__(self, dimension: int = 64) -> None:
        self._dimension = dimension
        self.calls: list[Sequence[str]] = []

    def embed_text(self, text: str) -> list[float]:
        """Embed a single string."""
        return self.embed_documents([text])[0]

    def embed_documents(self, documents: Sequence[str]) -> list[list[float]]:
        """Embed a batch of strings."""
        self.calls.append(list(documents))
        return [self._vector(text) for text in documents]

    def _vector(self, text: str) -> list[float]:
        vector = [0.0] * self._dimension
        for token in _TOKEN_RE.findall(text.lower()):
            index = int(hashlib.sha1(token.encode()).hexdigest(), 16) % self._dimension
            vector[index] += 1.0
        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            return vector
        return [value / norm for value in vector]

    @property
    def dimension(self) -> int:
        """Dimensionality of the produced vectors."""
        return self._dimension


class ScriptedLLM:
    """LLM double that records prompts and replays canned answers."""

    def __init__(self, responses: Sequence[str] | None = None, default: str = "OK [S1]") -> None:
        self._responses = list(responses or [])
        self._default = default
        self.prompts: list[list[LLMMessage]] = []

    def complete(self, messages: list[LLMMessage]) -> str:
        """Return the next canned response."""
        self.prompts.append(list(messages))
        if self._responses:
            return self._responses.pop(0)
        return self._default

    @property
    def last_user_prompt(self) -> str:
        """The most recent user message, for prompt assertions."""
        return self.prompts[-1][-1].content

    @property
    def model_name(self) -> str:
        """Reported model name."""
        return "scripted-llm"


class ExplodingEmbedder:
    """Embedder double that always fails, to test error handling."""

    def embed_text(self, text: str) -> list[float]:
        raise RuntimeError("embedding backend unavailable")

    def embed_documents(self, documents: Sequence[str]) -> list[list[float]]:
        raise RuntimeError("embedding backend unavailable")

    @property
    def dimension(self) -> int:
        """Reported dimensionality."""
        return 0


class ExplodingLLM:
    """LLM double that always fails, to test error handling."""

    def complete(self, messages: list[LLMMessage]) -> str:
        raise RuntimeError("provider timeout")

    @property
    def model_name(self) -> str:
        """Reported model name."""
        return "exploding-llm"


def make_chunk(page: int, text: str, document: str = "test.pdf", rank: int = 1) -> RetrievedChunk:
    """Build a :class:`RetrievedChunk` for prompt and pipeline tests."""
    return RetrievedChunk(
        chunk_id=f"{document}-p{page:04d}-c0000-deadbeef",
        text=text,
        document=document,
        source=document,
        page=page,
        score=0.9 - 0.1 * (rank - 1),
        rank=rank,
    )


class FakePipeline:
    """Duck-typed stand-in for :class:`~app.rag.pipeline.RAGPipeline`."""

    def __init__(
        self,
        answer: dict[str, Any] | None = None,
        ingestion: dict[str, Any] | None = None,
    ) -> None:
        self._answer = answer or {
            "answer": "The maximum loan tenure is 20 years [S1].",
            "sources": [{"document": "policy.pdf", "page": 7, "score": 0.8, "chunk_id": "c1"}],
            "no_answer": False,
            "chunks_used": 1,
            "model": "scripted-llm",
            "top_score": 0.8,
        }
        self._ingestion = ingestion
        self.store = _FakeStore()
        self.asked: list[str] = []
        self.ingested: list[str] = []

    def ask(self, question: str, top_k: int | None = None) -> Any:
        """Return the canned answer result."""
        self.asked.append(question)
        return _Result(self._answer)

    def ingest_document(self, path: Any) -> Any:
        """Return the canned ingestion result."""
        self.ingested.append(str(path))
        payload = self._ingestion or {
            "document": str(path).split("/")[-1],
            "pages_processed": 3,
            "chunks_created": 5,
            "chunks_replaced": 0,
            "total_words": 1200,
            "average_chunk_words": 240.0,
            "duration_ms": 120,
        }
        return _Result(payload)

    def indexed_documents(self) -> list[str]:
        """Return a fixed document list."""
        return ["policy.pdf"]


class _Result:
    """Minimal object exposing ``to_dict()`` like the real result dataclasses."""

    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def to_dict(self) -> dict[str, Any]:
        return self._payload


class _FakeStore:
    """Store double exposing only the count API used by the routes."""

    def __init__(self, chunks: int = 7) -> None:
        self._chunks = chunks

    def count(self) -> int:
        return self._chunks
