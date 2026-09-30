"""Semantic retrieval over the vector store.

Retrieval is intentionally independent of the LLM: it only turns a question
into ranked chunks with provenance. The RAG pipeline decides what to do with
them.
"""

from __future__ import annotations

"""Retrieval layer: turning a question into ranked, attributed chunks."""

from app.retrieval.models import RetrievedChunk
from app.retrieval.retriever import Retriever, get_retriever

__all__ = ["RetrievedChunk", "Retriever", "get_retriever"]
