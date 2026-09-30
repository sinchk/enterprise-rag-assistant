# syntax=docker/dockerfile:1
#
# Enterprise Document RAG Assistant - API image
#
# Multi-stage build: dependencies (which need a compiler for a few wheels) are
# installed into /opt/venv and copied into a slim runtime image, so the final
# image carries no build toolchain.

# --- Stage 1: build the dependency environment -----------------------------
FROM python:3.12-slim-bookworm AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

# Build essentials are required by a small number of transitive wheels. They
# live in this stage only and never reach the runtime image.
RUN apt-get update \
    && apt-get install --no-install-recommends -y build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /build
COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel \
    && pip install -r requirements.txt

# --- Stage 2: runtime -------------------------------------------------------
FROM python:3.12-slim-bookworm AS runtime

LABEL org.opencontainers.image.title="Enterprise Document RAG Assistant" \
      org.opencontainers.image.description="PDF ingestion, semantic retrieval and grounded answers over enterprise documents." \
      org.opencontainers.image.licenses="MIT"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    CHROMA_PERSIST_DIRECTORY=/data/chroma \
    UPLOAD_DIRECTORY=/data/uploads \
    LOG_LEVEL=INFO

# curl is used by the compose healthcheck; nothing else is added to the image.
RUN apt-get update \
    && apt-get install --no-install-recommends -y curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 rag

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
COPY --chown=rag:rag app ./app
COPY --chown=rag:rag scripts ./scripts
COPY --chown=rag:rag evaluation ./evaluation
COPY --chown=rag:rag data/sample ./data/sample
COPY --chown=rag:rag requirements.txt requirements-dev.txt README.md LICENSE ./

# The vector store and uploads live on a volume so they survive container
# replacement; /app stays read-only from the application's point of view.
RUN mkdir -p /data/chroma /data/uploads && chown -R rag:rag /data

USER rag
VOLUME ["/data"]
EXPOSE 8000

# /health is served without touching the model, so the check stays fast.
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

# One worker by default: the embedding model is loaded per process, so extra
# workers mean extra copies of the model in memory. Override for throughput.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
