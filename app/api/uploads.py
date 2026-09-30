"""Secure handling of uploaded documents.

Uploaded files are treated as untrusted: the extension is checked, the MIME
type is sanity-checked, the PDF magic bytes are verified, the size is capped and
the file name is sanitised before it touches the filesystem. Uploaded content is
only ever parsed as data - it is never executed or imported.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

from app.core.config import Settings
from app.core.errors import (
    EmptyDocumentError,
    InvalidPdfError,
    PayloadTooLargeError,
    UnsupportedFileTypeError,
)

logger = logging.getLogger(__name__)

# "%PDF-" is the mandatory header of every PDF file.
PDF_MAGIC = b"%PDF-"

# Browsers and clients are inconsistent about PDF MIME types, so a small set of
# observed values is accepted; the magic bytes remain the authoritative check.
_PDF_CONTENT_TYPES = {
    "application/pdf",
    "application/x-pdf",
    "application/octet-stream",
    "binary/octet-stream",
    "",
}

_UNSAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._ -]+")
_MULTI_DOT_RE = re.compile(r"\.{2,}")


@dataclass(frozen=True, slots=True)
class StoredUpload:
    """A validated upload, ready to be written to disk.

    ``path`` holds the *sanitised* name, which is also the name used for
    citations, so the original client-supplied name is not carried further.
    """

    path: Path
    size_bytes: int


def sanitize_filename(filename: str) -> str:
    """Return a safe, traversal-free file name.

    Directory components are discarded, unsafe characters are replaced and the
    length is bounded. ``"../../etc/passwd"`` therefore becomes ``"passwd"``.
    """
    base = Path(filename or "").name.strip()
    base = _MULTI_DOT_RE.sub(".", base)
    base = _UNSAFE_NAME_RE.sub("_", base).strip(" ._")
    if not base:
        raise InvalidPdfError("The uploaded file has no usable file name.")
    return base[:120]


def validate_upload(
    filename: str, content_type: str | None, data: bytes, settings: Settings
) -> StoredUpload:
    """Validate an upload and return the file name to store it under.

    Args:
        filename: Name reported by the client.
        content_type: MIME type reported by the client (untrusted).
        data: The complete file body.
        settings: Provides the size limit and accepted extensions.

    Returns:
        A :class:`StoredUpload` describing where the file should be written.

    Raises:
        UnsupportedFileTypeError: Wrong extension or MIME type.
        PayloadTooLargeError: The body exceeds ``max_upload_size_mb``.
        InvalidPdfError: Empty body or missing PDF header.
    """
    if not data:
        raise InvalidPdfError("The uploaded file is empty.")

    safe_name = sanitize_filename(filename)
    extension = Path(safe_name).suffix.lower()
    if extension not in settings.allowed_upload_extensions:
        allowed = ", ".join(settings.allowed_upload_extensions)
        raise UnsupportedFileTypeError(
            f"Unsupported file type '{extension or 'unknown'}'. Allowed: {allowed}."
        )

    if (content_type or "").split(";")[0].strip().lower() not in _PDF_CONTENT_TYPES:
        raise UnsupportedFileTypeError(
            f"Unsupported content type '{content_type}'. Upload a PDF file."
        )

    if len(data) > settings.max_upload_size_bytes:
        raise PayloadTooLargeError(
            f"The file exceeds the maximum upload size of "
            f"{settings.max_upload_size_mb} MB."
        )

    if not data.startswith(PDF_MAGIC):
        raise InvalidPdfError(
            "The uploaded file does not look like a PDF (missing %PDF- header)."
        )

    return StoredUpload(
        path=Path(settings.upload_directory) / safe_name, size_bytes=len(data)
    )


def read_limited(stream, settings: Settings) -> bytes:
    """Read at most ``max_upload_size_bytes + 1`` bytes from an upload stream.

    Reading one byte beyond the limit lets the caller distinguish "exactly at
    the limit" from "too large" without buffering an unbounded request body.
    """
    limit = settings.max_upload_size_bytes
    return stream.read(limit + 1)


def store_upload(upload: StoredUpload, data: bytes, settings: Settings) -> Path:
    """Write validated bytes to the upload directory and return the path."""
    directory = Path(settings.upload_directory)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / upload.path.name
    try:
        target.write_bytes(data)
    except OSError as exc:
        logger.error("Could not write upload to %s: %s", target, exc)
        raise EmptyDocumentError("The uploaded file could not be stored.") from exc
    logger.info("Stored upload '%s' (%s bytes).", target.name, upload.size_bytes)
    return target
