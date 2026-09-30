"""HTTP routes for the RAG assistant."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, File, Request, UploadFile, status

from app import __version__
from app.api.schemas import (
    AskRequest,
    AskResponse,
    DocumentListResponse,
    HealthResponse,
    IngestResponse,
    LLMStatus,
    VectorStoreStatus,
)
from app.api.uploads import read_limited, store_upload, validate_upload
from app.core.config import Settings, get_settings
from app.rag.pipeline import RAGPipeline

logger = logging.getLogger(__name__)

router = APIRouter()

ERROR_RESPONSES: dict[int | str, dict] = {
    400: {"description": "Invalid document or request."},
    413: {"description": "Upload exceeds the configured size limit."},
    415: {"description": "Unsupported file type."},
    422: {"description": "No extractable text in the document."},
    500: {"description": "Unexpected server error."},
    502: {"description": "The LLM provider failed."},
    503: {"description": "A dependency is unavailable or unconfigured."},
}


def get_pipeline(request: Request) -> RAGPipeline:
    """FastAPI dependency returning the application-wide RAG pipeline."""
    pipeline: RAGPipeline = request.app.state.pipeline
    return pipeline


def get_app_settings(request: Request) -> Settings:
    """FastAPI dependency returning the application settings."""
    settings: Settings | None = getattr(request.app.state, "settings", None)
    return settings or get_settings()


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Service health",
    description=(
        "Reports whether the vector database and the LLM provider are usable. "
        "Returns 200 when everything is healthy and 503 when a dependency is "
        "unavailable, which makes the endpoint suitable for container probes."
    ),
    tags=["system"],
)
def health(
    pipeline: RAGPipeline = Depends(get_pipeline),
    settings: Settings = Depends(get_app_settings),
) -> HealthResponse:
    """Return the health of the service and its dependencies."""
    store_status: VectorStoreStatus
    chunk_count = 0
    try:
        chunk_count = pipeline.store.count()
        store_status = VectorStoreStatus(status="ok", chunks=chunk_count)
    except Exception as exc:  # Reported as degraded, never as a 500.
        logger.warning("Vector store health check failed: %s", exc)
        store_status = VectorStoreStatus(status="unavailable", chunks=0)

    llm_status = LLMStatus(
        provider=settings.llm_provider,
        model=settings.llm_model,
        configured=(
            settings.llm_provider == "extractive" or settings.has_openai_credentials
        ),
    )
    healthy = store_status.status == "ok" and llm_status.configured
    return HealthResponse(
        status="ok" if healthy else "degraded",
        version=__version__,
        environment=settings.environment,
        vector_store=store_status,
        llm=llm_status,
    )


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------


@router.post(
    "/ingest",
    response_model=IngestResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Ingest a PDF document",
    description=(
        "Uploads a text-based PDF. The document is parsed with PyMuPDF, "
        "cleaned, split into overlapping chunks, embedded with the configured "
        "sentence-transformers model and stored in the persistent ChromaDB "
        "collection.\n\n"
        "Re-uploading the same file name replaces its previous chunks, so "
        "ingestion is idempotent. Scanned or image-only PDFs are rejected "
        "because OCR is not implemented in this version."
    ),
    responses=ERROR_RESPONSES,
    tags=["documents"],
)
def ingest_document(
    file: UploadFile = File(..., description="PDF file to index."),
    pipeline: RAGPipeline = Depends(get_pipeline),
    settings: Settings = Depends(get_app_settings),
) -> IngestResponse:
    """Ingest an uploaded PDF and return indexing statistics."""
    # Read at most one byte beyond the limit so an oversized body is detected
    # without buffering the whole request.
    data = read_limited(file.file, settings)
    validated = validate_upload(
        filename=file.filename or "",
        content_type=file.content_type,
        data=data,
        settings=settings,
    )
    path = store_upload(validated, data, settings)

    result = pipeline.ingest_document(path)
    return IngestResponse(
        message="Document ingested successfully", **result.to_dict()
    )


@router.get(
    "/documents",
    response_model=DocumentListResponse,
    summary="List indexed documents",
    description="Returns the documents currently available for question answering.",
    tags=["documents"],
)
def list_documents(
    pipeline: RAGPipeline = Depends(get_pipeline),
) -> DocumentListResponse:
    """List the documents present in the vector store."""
    return DocumentListResponse(
        documents=pipeline.indexed_documents(), chunks=pipeline.store.count()
    )


# ---------------------------------------------------------------------------
# Question answering
# ---------------------------------------------------------------------------


@router.post(
    "/ask",
    response_model=AskResponse,
    summary="Ask a question about the documents",
    description=(
        "Embeds the question, retrieves the most similar chunks from ChromaDB "
        "and asks the language model to answer using only that context.\n\n"
        "The response includes the document and page of every source used. If "
        "the documents do not contain the answer, the system replies that the "
        "information could not be found instead of guessing."
    ),
    responses=ERROR_RESPONSES,
    tags=["qa"],
)
def ask_question(
    payload: AskRequest,
    pipeline: RAGPipeline = Depends(get_pipeline),
) -> AskResponse:
    """Answer a question with citations back to the source pages."""
    result = pipeline.ask(payload.question, top_k=payload.top_k)
    return AskResponse(**result.to_dict())


@router.get(
    "/",
    include_in_schema=False,
)
def root() -> dict[str, str]:
    """Point users at the interactive documentation."""
    return {
        "service": "Enterprise Document RAG Assistant",
        "docs": "/docs",
        "openapi": "/openapi.json",
    }
