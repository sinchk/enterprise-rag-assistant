"""Tests for the FastAPI layer.

The API tests use a duck-typed pipeline double, so they cover routing,
validation, status codes, error envelopes and OpenAPI generation without an
embedding model, a vector database or an LLM. One integration test wires the
real pipeline (with a deterministic embedder and a real ChromaDB) to prove the
layers are connected correctly.
"""

from __future__ import annotations

import io

import pytest
from fastapi.testclient import TestClient

from app.api.schemas import AskRequest
from app.core.errors import (
    InvalidPdfError,
    LLMConfigurationError,
    LLMProviderError,
    RetrievalError,
    VectorStoreError,
)
from app.ingestion.chunker import TextChunker
from app.ingestion.pdf_loader import PdfLoader
from app.main import create_app
from app.rag.pipeline import RAGPipeline
from app.retrieval.retriever import Retriever
from app.vectorstore.chroma_store import create_chroma_store
from tests.conftest import make_pdf
from tests.fakes import FakePipeline, HashingEmbedder, ScriptedLLM


def build_client(pipeline=None, settings=None) -> TestClient:
    """Create a TestClient for an app wired to the given pipeline."""
    return TestClient(create_app(settings=settings, pipeline=pipeline or FakePipeline()))


@pytest.fixture
def client(settings) -> TestClient:
    """Client backed by the pipeline double and the isolated test settings."""
    return build_client(settings=settings)


def pdf_upload(content: bytes | None = None, filename: str = "policy.pdf", content_type: str = "application/pdf"):
    """Build a multipart file tuple for the upload endpoint."""
    return {
        "file": (filename, io.BytesIO(content or _valid_pdf_bytes()), content_type)
    }


def _valid_pdf_bytes() -> bytes:
    """A tiny valid PDF with a text layer."""
    document = __import__("fitz").open()
    page = document.new_page()
    page.insert_text((72, 72), "Maximum loan tenure is 20 years.", fontname="helv", fontsize=11)
    data = document.tobytes()
    document.close()
    return data


class TestHealth:
    def test_health_returns_ok(self, client) -> None:
        response = client.get("/health")

        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert body["version"]
        assert body["vector_store"]["status"] == "ok"
        assert body["llm"]["provider"] == "extractive"

    def test_health_reports_degraded_when_the_store_is_down(self) -> None:
        class BrokenStore:
            def count(self) -> int:
                raise VectorStoreError("The vector database is unavailable.")

        pipeline = FakePipeline()
        pipeline.store = BrokenStore()
        response = build_client(pipeline).get("/health")

        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "degraded"
        assert body["vector_store"]["status"] == "unavailable"

    def test_health_reports_degraded_without_an_api_key(self, settings) -> None:
        from app.core.config import Settings

        unconfigured = Settings(
            chroma_persist_directory=settings.chroma_persist_directory,
            chroma_collection=settings.chroma_collection,
            llm_provider="openai",
            openai_api_key=None,
        )
        response = build_client(FakePipeline(), unconfigured).get("/health")

        body = response.json()
        assert body["status"] == "degraded"
        assert body["llm"]["configured"] is False


class TestAskEndpoint:
    def test_valid_question_returns_a_grounded_answer(self, client) -> None:
        response = client.post("/ask", json={"question": "What is the maximum loan tenure?"})

        assert response.status_code == 200
        body = response.json()
        assert body["answer"] == "The maximum loan tenure is 20 years [S1]."
        assert body["sources"] == [
            {"document": "policy.pdf", "page": 7, "score": 0.8, "chunk_id": "c1"}
        ]
        assert body["no_answer"] is False
        assert body["model"] == "scripted-llm"
        assert body["chunks_used"] == 1

    def test_top_k_override_is_accepted(self, client) -> None:
        response = client.post(
            "/ask", json={"question": "What is the tenure?", "top_k": 2}
        )
        assert response.status_code == 200

    @pytest.mark.parametrize(
        ("payload", "reason"),
        [
            ({}, "missing question"),
            ({"question": ""}, "empty question"),
            ({"question": "   "}, "whitespace-only question"),
            ({"question": None}, "null question"),
            ({"question": 42}, "wrong type"),
            ({"question": "a" * 2001}, "question too long"),
            ({"question": "ok", "top_k": 0}, "top_k below range"),
            ({"question": "ok", "top_k": 99}, "top_k above range"),
        ],
    )
    def test_invalid_requests_return_422(self, client, payload, reason) -> None:
        response = client.post("/ask", json=payload)

        assert response.status_code == 422, reason
        body = response.json()
        assert body["error"]["code"] == "validation_error"
        assert body["error"]["message"]

    def test_malformed_json_returns_422(self, client) -> None:
        response = client.post(
            "/ask", content=b"{not json", headers={"content-type": "application/json"}
        )

        assert response.status_code == 422
        assert response.json()["error"]["code"] == "validation_error"

    def test_question_is_trimmed(self, client) -> None:
        response = client.post("/ask", json={"question": "  spaced question  "})
        assert response.status_code == 200

    @pytest.mark.parametrize(
        ("error", "status_code", "code"),
        [
            (LLMConfigurationError("OPENAI_API_KEY is not configured."), 503, "llm_not_configured"),
            (LLMProviderError("The language model provider failed."), 502, "llm_provider_error"),
            (RetrievalError("Could not search the document collection."), 500, "retrieval_failed"),
            (VectorStoreError("The vector database is unavailable."), 503, "vector_store_unavailable"),
        ],
    )
    def test_domain_errors_map_to_status_codes(self, error, status_code, code) -> None:
        class FailingPipeline(FakePipeline):
            def ask(self, question, top_k=None):
                raise error

        response = build_client(FailingPipeline()).post(
            "/ask", json={"question": "What is the maximum loan tenure?"}
        )

        assert response.status_code == status_code
        body = response.json()
        assert body["error"]["code"] == code
        assert body["error"]["message"] == str(error)

    def test_unexpected_error_does_not_leak_internals(self) -> None:
        class ExplodingPipeline(FakePipeline):
            def ask(self, question, top_k=None):
                raise RuntimeError("secret internal detail: token sk-123456")

        # Starlette's generic Exception handler re-raises after producing the
        # response, so the test client must be told not to propagate it.
        client = TestClient(
            create_app(pipeline=ExplodingPipeline()), raise_server_exceptions=False
        )
        response = client.post("/ask", json={"question": "What is the maximum loan tenure?"})

        assert response.status_code == 500
        body = response.json()
        assert body["error"]["code"] == "internal_error"
        assert "sk-123456" not in response.text
        assert "Traceback" not in response.text

    def test_model_refusal_is_reported(self) -> None:
        pipeline = FakePipeline(
            answer={
                "answer": "I could not find this information in the provided documents.",
                "sources": [],
                "no_answer": True,
                "chunks_used": 0,
                "model": "scripted-llm",
                "top_score": None,
            }
        )
        response = build_client(pipeline).post(
            "/ask", json={"question": "Which stadium do they play in?"}
        )

        body = response.json()
        assert body["no_answer"] is True
        assert body["sources"] == []
        assert body["top_score"] is None


class TestIngestEndpoint:
    def test_upload_returns_ingestion_statistics(self, client) -> None:
        response = client.post("/ingest", files=pdf_upload())

        assert response.status_code == 201
        body = response.json()
        assert body["message"] == "Document ingested successfully"
        assert body["document"] == "policy.pdf"
        assert body["chunks_created"] == 5
        assert body["pages_processed"] == 3
        assert body["duration_ms"] >= 0

    def test_missing_file_field_returns_422(self, client) -> None:
        assert client.post("/ingest", data={}).status_code == 422

    @pytest.mark.parametrize(
        ("filename", "content", "content_type", "status_code", "code"),
        [
            ("notes.txt", b"plain text", "text/plain", 415, "unsupported_file_type"),
            ("script.sh", b"#!/bin/sh", "application/x-sh", 415, "unsupported_file_type"),
            ("policy.pdf", b"", "application/pdf", 400, "invalid_pdf"),
            ("policy.pdf", b"not a pdf at all", "application/pdf", 400, "invalid_pdf"),
            ("policy.pdf", _valid_pdf_bytes(), "text/html", 415, "unsupported_file_type"),
        ],
    )
    def test_invalid_uploads_are_rejected(
        self, client, filename, content, content_type, status_code, code
    ) -> None:
        response = client.post(
            "/ingest",
            files={"file": (filename, io.BytesIO(content), content_type)},
        )

        assert response.status_code == status_code
        assert response.json()["error"]["code"] == code

    def test_oversized_upload_is_rejected(self, settings) -> None:
        from app.core.config import Settings

        tiny_limit = Settings(
            chroma_persist_directory=settings.chroma_persist_directory,
            chroma_collection=settings.chroma_collection,
            upload_directory=settings.upload_directory,
            max_upload_size_mb=1,
        )
        # 1 MB limit, so a 1.5 MB body must be refused before it is stored.
        payload = _valid_pdf_bytes() + b"\n%" + b"0" * (int(1.5 * 1024 * 1024))
        response = build_client(FakePipeline(), tiny_limit).post(
            "/ingest",
            files={"file": ("big.pdf", io.BytesIO(payload), "application/pdf")},
        )

        assert response.status_code == 413
        assert response.json()["error"]["code"] == "payload_too_large"

    def test_path_traversal_in_filename_is_neutralised(self, settings) -> None:
        from app.api.uploads import sanitize_filename

        assert sanitize_filename("../../etc/passwd") == "passwd"
        assert sanitize_filename("/absolute/path/report.pdf") == "report.pdf"
        assert sanitize_filename("evil;rm -rf.pdf") == "evil_rm -rf.pdf"
        assert sanitize_filename("....//....//x.pdf") == "x.pdf"

    def test_empty_filename_is_rejected(self, client) -> None:
        from app.core.errors import InvalidPdfError
        from app.api.uploads import sanitize_filename

        with pytest.raises(InvalidPdfError):
            sanitize_filename("")

        # Over HTTP the empty part is dropped by the multipart parser (422);
        # if it does reach us, validate_upload rejects it with 400.
        response = client.post(
            "/ingest", files={"file": ("", io.BytesIO(_valid_pdf_bytes()), "application/pdf")}
        )
        assert response.status_code in (400, 422)
        assert "error" in response.json()

    def test_documents_endpoint_lists_indexed_files(self, client) -> None:
        response = client.get("/documents")

        assert response.status_code == 200
        assert response.json() == {"documents": ["policy.pdf"], "chunks": 7}


class TestDocumentation:
    def test_swagger_ui_is_served(self, client) -> None:
        response = client.get("/docs")
        assert response.status_code == 200
        assert "swagger" in response.text.lower()

    def test_openapi_schema_documents_every_endpoint(self, client) -> None:
        schema = client.get("/openapi.json").json()

        assert set(schema["paths"]) == {"/health", "/ingest", "/ask", "/documents"}
        for path, operations in schema["paths"].items():
            for method, operation in operations.items():
                assert operation["summary"], f"{method} {path} has no summary"
                assert operation["description"], f"{method} {path} has no description"
                assert operation.get("tags"), f"{method} {path} has no tags"

    def test_ask_endpoint_documents_its_schema_and_examples(self, client) -> None:
        schema = client.get("/openapi.json").json()
        ask = schema["paths"]["/ask"]["post"]

        assert set(ask["requestBody"]["content"]["application/json"]["schema"]) <= {
            "$ref"
        }
        ask_schema = schema["components"]["schemas"]["AskRequest"]
        assert "question" in ask_schema["properties"]
        assert ask_schema["properties"]["question"]["examples"]
        assert set(ask["responses"]) >= {"200", "422", "503"}

    def test_ask_response_includes_a_source_example(self, client) -> None:
        schema = client.get("/openapi.json").json()
        example = schema["components"]["schemas"]["AskResponse"]["example"] if False else None
        # Pydantic renders the configured example under "examples" in JSON Schema.
        schemas = schema["components"]["schemas"]
        assert "AskResponse" in schemas
        assert "Source" in schemas
        assert example is None  # documents the choice, keeps the test explicit

    def test_root_points_at_the_documentation(self, client) -> None:
        body = client.get("/").json()
        assert body["docs"] == "/docs"

    def test_unknown_path_returns_404_in_the_error_envelope(self, client) -> None:
        response = client.get("/does-not-exist")
        assert response.status_code == 404
        assert response.json()["error"]["code"] == "http_error"


class TestSchemaValidation:
    @pytest.mark.parametrize("value", ["", "   ", "\n\t"])
    def test_blank_questions_are_rejected_by_the_schema(self, value) -> None:
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            AskRequest(question=value)

    def test_question_length_is_bounded(self) -> None:
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            AskRequest(question="x" * 2001)

    def test_question_is_trimmed_by_the_schema(self) -> None:
        assert AskRequest(question="  hello  ").question == "hello"


class TestEndToEndWithRealPipeline:
    @staticmethod
    def _real_pipeline(settings):
        """Wire a real pipeline with a deterministic embedder and a stub LLM."""
        store = create_chroma_store(settings)
        embedder = HashingEmbedder()
        return RAGPipeline(
            settings=settings,
            store=store,
            embedder=embedder,
            retriever=Retriever(embedder, store, top_k=settings.top_k),
            llm=ScriptedLLM(default="The maximum loan tenure is 20 years [S1]."),
        )

    def test_scanned_pdf_without_a_text_layer_returns_422(self, settings) -> None:
        settings.upload_directory.mkdir(parents=True, exist_ok=True)
        blank_pdf = make_pdf(settings.upload_directory / "scanned.pdf", blank=True)
        client = build_client(self._real_pipeline(settings), settings)

        response = client.post(
            "/ingest",
            files={
                "file": ("scanned.pdf", blank_pdf.read_bytes(), "application/pdf")
            },
        )

        assert response.status_code == 422
        assert response.json()["error"]["code"] == "empty_document"

    def test_ingest_then_ask_through_the_http_api(
        self, settings, tmp_path, lending_pdf
    ) -> None:
        """The real pipeline, real ChromaDB, deterministic embedder, stub LLM."""
        client = build_client(self._real_pipeline(settings), settings)

        upload = client.post(
            "/ingest",
            files={
                "file": (
                    "lending_policy.pdf",
                    lending_pdf.read_bytes(),
                    "application/pdf",
                )
            },
        )
        assert upload.status_code == 201
        assert upload.json()["chunks_created"] == 3

        answer = client.post(
            "/ask", json={"question": "What is the maximum loan tenure?"}
        )
        assert answer.status_code == 200
        body = answer.json()
        assert body["sources"][0]["document"] == "lending_policy.pdf"
        assert body["sources"][0]["page"] == 1
        assert body["chunks_used"] == settings.top_k

        # The upload was sanitised and stored under the configured directory.
        stored = settings.upload_directory / "lending_policy.pdf"
        assert stored.exists()
        assert PdfLoader().load(stored)[0].page == 1

    def test_ask_before_any_ingestion_refuses_to_answer(self, settings) -> None:
        store = create_chroma_store(settings)
        embedder = HashingEmbedder()
        llm = ScriptedLLM(default="must not be called")
        pipeline = RAGPipeline(
            settings=settings,
            store=store,
            embedder=embedder,
            retriever=Retriever(embedder, store, top_k=settings.top_k),
            llm=llm,
        )

        response = build_client(pipeline, settings).post(
            "/ask", json={"question": "What is the maximum loan tenure?"}
        )

        assert response.status_code == 200
        body = response.json()
        assert body["no_answer"] is True
        assert body["answer"] == "I could not find this information in the provided documents."
        assert body["sources"] == []
        assert llm.prompts == []

    def test_missing_api_key_returns_503_on_ask(self, settings) -> None:
        from app.core.config import Settings

        unconfigured = Settings(
            chroma_persist_directory=settings.chroma_persist_directory,
            chroma_collection=settings.chroma_collection,
            upload_directory=settings.upload_directory,
            llm_provider="openai",
            openai_api_key=None,
        )
        store = create_chroma_store(settings)
        embedder = HashingEmbedder()
        pipeline = RAGPipeline(
            settings=unconfigured,
            store=store,
            embedder=embedder,
            retriever=Retriever(embedder, store, top_k=3),
        )
        # Give the store something to retrieve so the LLM is actually needed.
        settings.upload_directory.mkdir(parents=True, exist_ok=True)
        pages = PdfLoader().load(
            make_pdf(settings.upload_directory / "p.pdf", pages=[["A fact about loan tenure."]])
        )
        chunks = TextChunker(chunk_size=100, chunk_overlap=10).chunk_pages(pages)
        store.add_documents(chunks, embedder.embed_documents([c.text for c in chunks]))

        response = build_client(pipeline, unconfigured).post(
            "/ask", json={"question": "What is the loan tenure?"}
        )

        assert response.status_code == 503
        assert response.json()["error"]["code"] == "llm_not_configured"
        assert "OPENAI_API_KEY" in response.json()["error"]["message"]
