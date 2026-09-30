"""Application configuration loaded from environment variables / ``.env``.

All tunable values live here so that no magic numbers are scattered through the
code base. Values are read once per process via :func:`get_settings`, which is
cached; tests can call :func:`reset_settings_cache` after mutating the
environment.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Repository root, i.e. the directory that contains ``app/``.
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

# Directories that are always available, regardless of the working directory.
DEFAULT_CHROMA_DIR: Path = PROJECT_ROOT / "chroma_db"
DEFAULT_UPLOAD_DIR: Path = PROJECT_ROOT / "data" / "uploads"
SAMPLE_DIR: Path = PROJECT_ROOT / "data" / "sample"

LlmProvider = Literal["openai", "extractive"]


class Settings(BaseSettings):
    """Runtime settings for the RAG service.

    Attributes are read from the process environment first and then from a
    ``.env`` file in the current working directory. Secrets are represented by
    :class:`~pydantic.SecretStr` so they are never logged or echoed.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- Service metadata -------------------------------------------------
    app_name: str = "Enterprise Document RAG Assistant"
    app_version: str = "1.0.0"
    environment: Literal["local", "docker", "test", "production"] = "local"
    log_level: str = "INFO"

    # --- Vector store -----------------------------------------------------
    chroma_persist_directory: Path = Field(
        default=DEFAULT_CHROMA_DIR,
        description="Directory where ChromaDB persists its files. Must be a "
        "writable path that survives restarts.",
    )
    chroma_collection: str = Field(
        default="enterprise_documents",
        description="Name of the ChromaDB collection holding document chunks.",
    )

    # --- Embeddings -------------------------------------------------------
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="Sentence-Transformers model id used to embed text.",
    )
    embedding_device: str | None = Field(
        default=None,
        description="Torch device for embeddings (e.g. 'cpu' or 'cuda'). "
        "Defaults to None, letting sentence-transformers auto-detect.",
    )
    embedding_batch_size: int = Field(
        default=32, ge=1, le=512, description="Batch size for embedding calls."
    )

    # --- Chunking ---------------------------------------------------------
    chunk_size: int = Field(
        default=600,
        ge=50,
        le=4000,
        description="Target chunk size in words.",
    )
    chunk_overlap: int = Field(
        default=80,
        ge=0,
        le=1000,
        description="Overlap between consecutive chunks, in words.",
    )

    # --- Retrieval --------------------------------------------------------
    top_k: int = Field(
        default=4, ge=1, le=50, description="Number of chunks returned per query."
    )
    similarity_threshold: float | None = Field(
        default=None,
        ge=-1.0,
        le=1.0,
        description="Optional minimum cosine similarity. Chunks scoring below "
        "this value are discarded, and an empty result set means 'no evidence'.",
    )

    # --- LLM --------------------------------------------------------------
    llm_provider: LlmProvider = Field(
        default="openai",
        description="'openai' for a hosted chat model, 'extractive' for the "
        "offline non-LLM baseline used in tests and keyless demos.",
    )
    llm_model: str = Field(
        default="gpt-4o-mini", description="Chat completion model identifier."
    )
    llm_temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    llm_max_tokens: int = Field(default=700, ge=32, le=8192)
    llm_timeout_seconds: float = Field(default=60.0, gt=0)
    llm_max_context_chars: int = Field(
        default=12000,
        ge=500,
        le=500000,
        description="Hard cap on the prompt context to bound token cost.",
    )
    openai_api_key: SecretStr | None = Field(
        default=None, description="Required when llm_provider='openai'."
    )
    openai_base_url: str | None = Field(
        default=None,
        description="Optional override, e.g. for an OpenAI-compatible gateway.",
    )

    # --- Uploads / security ----------------------------------------------
    upload_directory: Path = Field(
        default=DEFAULT_UPLOAD_DIR, description="Where uploaded PDFs are stored."
    )
    max_upload_size_mb: int = Field(
        default=25, ge=1, le=500, description="Maximum accepted upload size."
    )
    allowed_upload_extensions: tuple[str, ...] = Field(
        default=(".pdf",), description="Accepted upload file extensions."
    )

    @field_validator("allowed_upload_extensions", mode="before")
    @classmethod
    def _normalise_extensions(cls, value: object) -> object:
        """Accept comma-separated strings from the environment."""
        if isinstance(value, str):
            parts = [p.strip().lower() for p in value.split(",") if p.strip()]
            if not parts:
                raise ValueError("allowed_upload_extensions must not be empty")
            return tuple(parts)
        return value

    @field_validator("log_level")
    @classmethod
    def _validate_log_level(cls, value: str) -> str:
        level = value.upper()
        if level not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
            raise ValueError(f"Unsupported log level: {value}")
        return level

    @model_validator(mode="after")
    def _validate_consistency(self) -> "Settings":
        """Reject contradictory chunking configuration early."""
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                "chunk_overlap must be smaller than chunk_size "
                f"(got overlap={self.chunk_overlap}, size={self.chunk_size})"
            )
        return self

    @property
    def max_upload_size_bytes(self) -> int:
        """Maximum accepted upload size expressed in bytes."""
        return self.max_upload_size_mb * 1024 * 1024

    @property
    def has_openai_credentials(self) -> bool:
        """Whether an OpenAI API key is configured."""
        return self.openai_api_key is not None and bool(
            self.openai_api_key.get_secret_value().strip()
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings singleton."""
    return Settings()


def reset_settings_cache() -> None:
    """Clear the cached settings (used by tests that patch the environment)."""
    get_settings.cache_clear()


def configure_logging(level: str | None = None) -> None:
    """Install a single, non-duplicating logging configuration."""
    logging.basicConfig(
        level=getattr(logging, (level or get_settings().log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)-8s %(name)s | %(message)s",
        force=True,
    )
    # ChromaDB telemetry is disabled in ChromaVectorStore. Silence the logger so
    # version-skew noise from the telemetry client does not pollute the output.
    logging.getLogger("chromadb.telemetry").setLevel(logging.CRITICAL)
    logging.getLogger("httpx").setLevel(logging.WARNING)
