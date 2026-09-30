"""Pydantic request/response schemas for the REST API.

These are the only wire-format types; internal domain objects stay in their own
modules. Every endpoint has an explicit ``response_model`` so the generated
OpenAPI documentation (served at ``/docs``) is accurate.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app import __version__

EXAMPLE_QUESTION = "What is the maximum loan tenure?"


class Source(BaseModel):
    """A single piece of evidence backing an answer."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "document": "northwind_lending_policy.pdf",
                "page": 7,
                "score": 0.8123,
                "chunk_id": "northwind-lending-policy-p0007-c0000-a1b2c3d4e5f6",
            }
        }
    )

    document: str = Field(description="File name of the source document.")
    page: int = Field(description="1-based page number the text came from.")
    score: float = Field(description="Cosine similarity between question and chunk.")
    chunk_id: str = Field(description="Stable identifier of the retrieved chunk.")


class AskRequest(BaseModel):
    """Question submitted by the user."""

    model_config = ConfigDict(
        json_schema_extra={"example": {"question": EXAMPLE_QUESTION}},
    )

    question: str = Field(
        min_length=1,
        max_length=2000,
        description="Natural-language question about the ingested documents.",
        examples=[EXAMPLE_QUESTION],
    )
    top_k: int | None = Field(
        default=None,
        ge=1,
        le=20,
        description="Override the number of retrieved chunks for this request.",
    )

    @field_validator("question")
    @classmethod
    def _reject_blank(cls, value: str) -> str:
        """Reject questions that contain only whitespace."""
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("question must contain at least one non-space character")
        return cleaned


class AskResponse(BaseModel):
    """Grounded answer plus attribution."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "answer": (
                    "The maximum loan tenure is 20 years [S1].\n\n"
                    "Source: northwind_lending_policy.pdf, page 7."
                ),
                "sources": [
                    {
                        "document": "northwind_lending_policy.pdf",
                        "page": 7,
                        "score": 0.8123,
                        "chunk_id": "northwind-lending-policy-p0007-c0000-a1b2c3d4e5f6",
                    }
                ],
                "no_answer": False,
                "chunks_used": 4,
                "model": "gpt-4o-mini",
                "top_score": 0.8123,
            }
        }
    )

    answer: str = Field(
        description=(
            "The generated answer. When the documents do not contain the "
            "information, this is an explicit 'could not find' statement."
        )
    )
    sources: list[Source] = Field(
        default_factory=list,
        description="Documents and pages the answer was derived from.",
    )
    no_answer: bool = Field(
        description="True when the system states the information was not found.",
    )
    chunks_used: int = Field(description="Number of chunks supplied to the model.")
    model: str = Field(description="Model that produced the answer.")
    top_score: float | None = Field(
        default=None, description="Similarity of the best matching chunk."
    )


class IngestResponse(BaseModel):
    """Summary returned after a document has been indexed."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "message": "Document ingested successfully",
                "document": "northwind_lending_policy.pdf",
                "pages_processed": 12,
                "chunks_created": 18,
                "chunks_replaced": 0,
                "total_words": 7412,
                "average_chunk_words": 411.8,
                "duration_ms": 4210,
            }
        }
    )

    message: str = Field(description="Human readable status message.")
    document: str = Field(description="File name of the ingested document.")
    pages_processed: int = Field(description="Pages that yielded text.")
    chunks_created: int = Field(description="Chunks written to the vector store.")
    chunks_replaced: int = Field(
        description="Previous chunks deleted for this document before writing."
    )
    total_words: int = Field(description="Total words indexed.")
    average_chunk_words: float = Field(description="Mean chunk length in words.")
    duration_ms: int = Field(description="Server-side ingestion time in milliseconds.")


class VectorStoreStatus(BaseModel):
    """Health of the vector database."""

    status: str = Field(description="'ok' or 'unavailable'.")
    chunks: int = Field(description="Number of chunks currently indexed.")


class LLMStatus(BaseModel):
    """Configuration status of the LLM provider."""

    provider: str = Field(description="Configured provider name.")
    model: str = Field(description="Configured model name.")
    configured: bool = Field(
        description="Whether the provider has everything it needs to answer."
    )


class HealthResponse(BaseModel):
    """Response of the health endpoint."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": "ok",
                "version": __version__,
                "environment": "local",
                "vector_store": {"status": "ok", "chunks": 18},
                "llm": {
                    "provider": "openai",
                    "model": "gpt-4o-mini",
                    "configured": True,
                },
            }
        }
    )

    status: str = Field(description="'ok' when every dependency is usable.")
    version: str = Field(description="Application version.")
    environment: str = Field(description="Deployment environment name.")
    vector_store: VectorStoreStatus = Field(description="Vector database health.")
    llm: LLMStatus = Field(description="LLM provider health.")


class DocumentListResponse(BaseModel):
    """Documents currently indexed in the vector store."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "documents": ["northwind_lending_policy.pdf"],
                "chunks": 18,
            }
        }
    )

    documents: list[str] = Field(description="Names of the indexed documents.")
    chunks: int = Field(description="Total number of stored chunks.")


class ErrorDetail(BaseModel):
    """Body of every error response."""

    code: str = Field(description="Stable machine-readable error code.")
    message: str = Field(description="Explanation safe to show to API users.")


class ErrorResponse(BaseModel):
    """Envelope used for all non-2xx responses."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "error": {
                    "code": "empty_document",
                    "message": (
                        "No text could be extracted from this PDF. It is most "
                        "likely a scanned or image-only document, which requires "
                        "OCR support."
                    ),
                }
            }
        }
    )

    error: ErrorDetail
