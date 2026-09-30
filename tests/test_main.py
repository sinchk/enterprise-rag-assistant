"""Tests for the application factory, lifespan and configuration reporting."""

from __future__ import annotations

import logging

from fastapi.testclient import TestClient

from app.core.config import Settings
from app.main import create_app, describe_configuration
from tests.fakes import FakePipeline


class TestCreateApp:
    def test_uses_the_injected_settings(self, settings) -> None:
        app = create_app(settings=settings, pipeline=FakePipeline())

        assert app.title == settings.app_name
        assert app.state.settings is settings

    def test_injected_pipeline_is_available_without_startup(self, settings) -> None:
        pipeline = FakePipeline()
        app = create_app(settings=settings, pipeline=pipeline)

        assert app.state.pipeline is pipeline
        with TestClient(app) as client:
            assert client.get("/health").status_code == 200

    def test_startup_builds_the_real_pipeline_when_none_is_injected(
        self, settings, monkeypatch
    ) -> None:
        built: list[object] = []

        def _factory(settings=None):
            pipeline = FakePipeline()
            built.append(pipeline)
            return pipeline

        monkeypatch.setattr("app.main.RAGPipeline", _factory)
        app = create_app(settings=settings)

        assert built == [], "the real pipeline must not be built at import time"

        with TestClient(app) as client:
            assert built == [app.state.pipeline]
            assert client.get("/health").json()["status"] == "ok"

    def test_shutdown_releases_the_pipeline(self, settings) -> None:
        app = create_app(settings=settings, pipeline=FakePipeline())

        with TestClient(app):
            assert app.state.pipeline is not None

        assert app.state.pipeline is None

    def test_injected_pipeline_is_kept_across_startup(self, settings) -> None:
        pipeline = FakePipeline()
        app = create_app(settings=settings, pipeline=pipeline)

        with TestClient(app):
            assert app.state.pipeline is pipeline

    def test_openapi_document_is_served_and_described(self, settings) -> None:
        app = create_app(settings=settings, pipeline=FakePipeline())

        with TestClient(app) as client:
            schema = client.get("/openapi.json").json()
            docs = client.get("/docs")

        assert set(schema["paths"]) >= {"/health", "/ingest", "/ask", "/documents"}
        assert docs.status_code == 200
        assert schema["info"]["title"] == settings.app_name

    def test_root_points_at_the_documentation(self, settings) -> None:
        app = create_app(settings=settings, pipeline=FakePipeline())

        with TestClient(app) as client:
            body = client.get("/").json()

        assert body["service"] == settings.app_name
        assert body["docs"] == "/docs"
        assert body["openapi"] == "/openapi.json"


class TestDescribeConfiguration:
    def test_reports_the_effective_settings(self) -> None:
        settings = Settings(
            llm_provider="openai",
            openai_api_key="sk-super-secret-value",
            embedding_model="sentence-transformers/all-MiniLM-L6-v2",
            chunk_size=512,
            chunk_overlap=64,
            top_k=5,
        )

        summary = describe_configuration(settings)

        assert summary["embedding_model"] == "sentence-transformers/all-MiniLM-L6-v2"
        assert summary["chunk_size"] == 512
        assert summary["chunk_overlap"] == 64
        assert summary["top_k"] == 5
        assert summary["llm_provider"] == "openai"
        assert summary["llm_model"] == settings.llm_model

    def test_never_exposes_the_api_key(self) -> None:
        settings = Settings(
            llm_provider="openai", openai_api_key="sk-super-secret-value"
        )

        rendered = repr(describe_configuration(settings))

        assert "sk-super-secret-value" not in rendered
        assert "openai_api_key" not in rendered

    def test_is_json_serialisable(self) -> None:
        import json

        assert json.loads(json.dumps(describe_configuration(Settings())))["top_k"] >= 1

    def test_startup_logs_the_configuration(self, settings, caplog) -> None:
        app = create_app(settings=settings, pipeline=FakePipeline())

        with caplog.at_level(logging.INFO, logger="app.main"):
            with TestClient(app):
                pass

        assert "RAG assistant ready" in caplog.text
        assert settings.embedding_model in caplog.text

    def test_injected_settings_do_not_reconfigure_logging(self, settings) -> None:
        """An embedding caller owns its own logging setup."""
        root_handlers = list(logging.getLogger().handlers)
        create_app(settings=settings, pipeline=FakePipeline())
        assert logging.getLogger().handlers == root_handlers
