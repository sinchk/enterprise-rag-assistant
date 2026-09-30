"""Domain exceptions.

The API layer maps these to HTTP responses, which keeps internal error details
(tracebacks, library messages) out of client-facing payloads.
"""

from __future__ import annotations


class AppError(Exception):
    """Base class for expected, user-facing failures.

    ``message`` is safe to return to an API client; the original cause is
    preserved in ``__cause__`` for server-side logs only.
    """

    status_code: int = 500
    error_code: str = "internal_error"

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


# --- Ingestion ------------------------------------------------------------


class DocumentError(AppError):
    """Base class for document ingestion failures."""

    status_code = 400
    error_code = "document_error"


class UnsupportedFileTypeError(DocumentError):
    """The uploaded file is not an accepted document type."""

    status_code = 415
    error_code = "unsupported_file_type"


class InvalidPdfError(DocumentError):
    """The file is not a readable PDF."""

    status_code = 400
    error_code = "invalid_pdf"


class EmptyDocumentError(DocumentError):
    """The PDF contains no extractable text (e.g. a scanned document)."""

    status_code = 422
    error_code = "empty_document"


class PayloadTooLargeError(DocumentError):
    """The upload exceeds the configured size limit."""

    status_code = 413
    error_code = "payload_too_large"


# --- Services -------------------------------------------------------------


class VectorStoreError(AppError):
    """The vector database is unavailable or rejected an operation."""

    status_code = 503
    error_code = "vector_store_unavailable"


class LLMConfigurationError(AppError):
    """A required LLM setting (typically the API key) is missing."""

    status_code = 503
    error_code = "llm_not_configured"


class LLMProviderError(AppError):
    """The LLM provider failed while generating a completion."""

    status_code = 502
    error_code = "llm_provider_error"


class RetrievalError(AppError):
    """Semantic retrieval failed."""

    status_code = 500
    error_code = "retrieval_failed"
