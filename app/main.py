"""FastAPI application factory, lifespan management and error handling.

Run locally with::

    uvicorn app.main:app --reload
"""

from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app import __version__
from app.api.routes import router
from app.core.config import Settings, configure_logging, get_settings
from app.core.errors import AppError
from app.llm.client import describe_provider
from app.rag.pipeline import RAGPipeline

logger = logging.getLogger(__name__)

DESCRIPTION = """\
Retrieval-Augmented Generation service for enterprise PDF documents.

Typical flow:

1. `POST /ingest` uploads a PDF, which is parsed, cleaned, chunked, embedded \
and stored in a persistent ChromaDB collection.
2. `POST /ask` embeds a question, retrieves the most similar chunks and asks \
the language model to answer using only that context.
3. Every answer is returned with the document and page it came from, and the \
model is instructed to say so when the documents do not contain the answer.

Scanned/image-only PDFs are not supported: they have no text layer and OCR is
outside the scope of this version.
"""

TAGS_METADATA = [
    {"name": "system", "description": "Service health and metadata."},
    {"name": "documents", "description": "Upload and inspect indexed documents."},
    {"name": "qa", "description": "Grounded question answering with citations."},
]


def _error_body(code: str, message: str) -> dict[str, Any]:
    """Build the standard error envelope."""
    return {"error": {"code": code, "message": message}}


def register_exception_handlers(app: FastAPI) -> None:
    """Map domain and framework exceptions to safe JSON responses.

    Internal details (tracebacks, library messages) are logged server-side only;
    clients receive a stable error code and a human readable message.
    """

    @app.exception_handler(AppError)
    async def handle_app_error(request: Request, exc: AppError) -> JSONResponse:
        logger.warning(
            "%s %s -> %s (%s)", request.method, request.url.path, exc.error_code, exc.message
        )
        return JSONResponse(
            status_code=exc.status_code,
            content=_error_body(exc.error_code, exc.message),
        )

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        problems = [
            f"{'.'.join(str(part) for part in error['loc'][1:]) or 'body'}: {error['msg']}"
            for error in exc.errors()
        ]
        logger.info(
            "Validation error on %s %s: %s", request.method, request.url.path, problems
        )
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=_error_body(
                "validation_error",
                "The request body is invalid. "
                + "; ".join(problems[:5])
                + ("; ..." if len(problems) > 5 else ""),
            ),
        )

    @app.exception_handler(StarletteHTTPException)
    async def handle_http_error(
        request: Request, exc: StarletteHTTPException
    ) -> JSONResponse:
        if exc.status_code < 500:
            return JSONResponse(
                status_code=exc.status_code,
                content=_error_body("http_error", str(exc.detail)),
            )
        logger.error(
            "HTTP %s on %s %s", exc.status_code, request.method, request.url.path
        )
        return JSONResponse(
            status_code=exc.status_code,
            content=_error_body(
                "internal_error", "The server could not complete the request."
            ),
        )

    @app.exception_handler(Exception)
    async def handle_unexpected_error(request: Request, exc: Exception) -> JSONResponse:
        logger.exception(
            "Unhandled error on %s %s", request.method, request.url.path, exc_info=exc
        )
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=_error_body(
                "internal_error",
                "An unexpected error occurred. Please check the server logs.",
            ),
        )


def create_app(
    settings: Settings | None = None, pipeline: RAGPipeline | None = None
) -> FastAPI:
    """Build the FastAPI application.

    Args:
        settings: Configuration to use; defaults to the process settings.
            Passing settings explicitly also means the caller owns logging
            configuration, so this factory does not reconfigure it.
        pipeline: Pre-built pipeline. Tests inject a fake here; when omitted
            the real pipeline is created during application startup so that
            importing the module never loads models or opens the database.
    """
    if settings is None:
        settings = get_settings()
        configure_logging(settings.log_level)

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        if application.state.pipeline is None:
            application.state.pipeline = RAGPipeline(settings=settings)
        logger.info(
            "RAG assistant ready: %s",
            json.dumps(describe_configuration(settings), sort_keys=True),
        )
        yield
        application.state.pipeline = None

    app = FastAPI(
        title=settings.app_name,
        version=__version__,
        description=DESCRIPTION,
        openapi_tags=TAGS_METADATA,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # An explicitly injected pipeline is published immediately so the app is
    # usable without running startup (tests and embedding use this). When no
    # pipeline is injected, the real one is built during startup, which keeps
    # module import free of model loading and database access.
    app.state.settings = settings
    app.state.pipeline = pipeline

    register_exception_handlers(app)
    app.include_router(router)
    return app


app = create_app()


def describe_configuration(settings: Settings | None = None) -> dict[str, Any]:
    """Return a loggable summary of the active configuration (no secrets)."""
    settings = settings or get_settings()
    return {
        "environment": settings.environment,
        "embedding_model": settings.embedding_model,
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "top_k": settings.top_k,
        "chroma_persist_directory": str(settings.chroma_persist_directory),
        "llm_provider": settings.llm_provider,
        "llm_model": describe_provider(settings),
    }
