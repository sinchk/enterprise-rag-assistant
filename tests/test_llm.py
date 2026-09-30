"""Tests for the OpenAI LLM client.

The OpenAI SDK is replaced with a stub, so the request shape, the error mapping
and the handling of an empty completion are verified without a network call, an
API key or any cost.
"""

from __future__ import annotations

import logging

import pytest

from app.core.config import Settings
from app.core.errors import LLMConfigurationError, LLMProviderError
from app.llm.client import OpenAIChatClient, describe_provider
from app.llm import client as client_module
from app.llm.client import LLMMessage


class _FakeMessage:
    def __init__(self, content: str | None) -> None:
        self.content = content


class _FakeChoice:
    def __init__(self, content: str | None) -> None:
        self.message = _FakeMessage(content)


class _FakeResponse:
    def __init__(self, content: str | None) -> None:
        self.choices = [_FakeChoice(content)]


class _FakeCompletions:
    def __init__(self, response: _FakeResponse | None = None, error: Exception | None = None) -> None:
        self._response = response
        self._error = error
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self._error is not None:
            raise self._error
        return self._response


class _FakeOpenAI:
    """Stand-in for ``openai.OpenAI`` recording how it was constructed."""

    completions: _FakeCompletions
    init_kwargs: dict

    def __init__(self, **kwargs) -> None:
        type(self).init_kwargs = kwargs
        self.chat = type("Chat", (), {"completions": type(self).completions})()


@pytest.fixture
def openai_settings() -> Settings:
    """Settings with a dummy key so client construction succeeds."""
    return Settings(
        llm_provider="openai",
        openai_api_key="sk-test-not-a-real-key",
        llm_model="gpt-4o-mini",
        llm_temperature=0.0,
        llm_max_tokens=256,
        llm_timeout_seconds=12.5,
    )


@pytest.fixture
def patched_openai(monkeypatch):
    """Install the fake SDK and return the completions double."""
    import openai

    monkeypatch.setattr(openai, "OpenAI", _FakeOpenAI)
    return _FakeOpenAI


class TestConfiguration:
    def test_missing_api_key_raises_a_clear_error(self) -> None:
        with pytest.raises(LLMConfigurationError) as excinfo:
            OpenAIChatClient(Settings(llm_provider="openai", openai_api_key=None))

        message = excinfo.value.message
        assert "OPENAI_API_KEY" in message
        assert ".env.example" in message
        assert excinfo.value.status_code == 503

    def test_blank_api_key_is_treated_as_missing(self) -> None:
        with pytest.raises(LLMConfigurationError):
            OpenAIChatClient(
                Settings(llm_provider="openai", openai_api_key="   ")
            )

    def test_client_receives_key_timeout_and_model(self, openai_settings, patched_openai) -> None:
        patched_openai.completions = _FakeCompletions(_FakeResponse("Grounded answer."))
        OpenAIChatClient(openai_settings)

        assert patched_openai.init_kwargs["api_key"] == "sk-test-not-a-real-key"
        assert patched_openai.init_kwargs["timeout"] == 12.5
        assert "base_url" not in patched_openai.init_kwargs

    def test_base_url_override_is_forwarded(self, openai_settings, patched_openai) -> None:
        patched_openai.completions = _FakeCompletions(_FakeResponse("ok"))
        settings = openai_settings.model_copy(
            update={"openai_base_url": "http://localhost:11434/v1"}
        )

        OpenAIChatClient(settings)
        assert patched_openai.init_kwargs["base_url"] == "http://localhost:11434/v1"

    def test_api_key_is_never_logged(self, openai_settings, patched_openai, caplog) -> None:
        patched_openai.completions = _FakeCompletions(_FakeResponse("ok"))

        with caplog.at_level(logging.DEBUG):
            OpenAIChatClient(openai_settings)

        assert "sk-test-not-a-real-key" not in caplog.text

    def test_describe_provider_reports_the_model(self, openai_settings) -> None:
        assert describe_provider(openai_settings) == "gpt-4o-mini"

    def test_describe_provider_reports_the_baseline(self) -> None:
        assert "non-LLM" in describe_provider(Settings(llm_provider="extractive"))


class TestCompletion:
    def test_returns_the_assistant_text(self, openai_settings, patched_openai) -> None:
        patched_openai.completions = _FakeCompletions(_FakeResponse("  The answer is 20 years.  "))
        client = OpenAIChatClient(openai_settings)

        assert client.complete([LLMMessage(role="user", content="Q?")]) == (
            "The answer is 20 years."
        )

    def test_request_carries_the_model_and_grounding_messages(
        self, openai_settings, patched_openai
    ) -> None:
        patched_openai.completions = _FakeCompletions(_FakeResponse("answer"))
        client = OpenAIChatClient(openai_settings)

        client.complete(
            [
                LLMMessage(role="system", content="Use only the CONTEXT."),
                LLMMessage(role="user", content="What is the tenure?"),
            ]
        )

        call = patched_openai.completions.calls[0]
        assert call["model"] == "gpt-4o-mini"
        assert call["temperature"] == 0.0
        assert call["max_tokens"] == 256
        assert call["messages"] == [
            {"role": "system", "content": "Use only the CONTEXT."},
            {"role": "user", "content": "What is the tenure?"},
        ]

    def test_provider_failure_is_wrapped(self, openai_settings, patched_openai) -> None:
        patched_openai.completions = _FakeCompletions(
            error=RuntimeError("Connection reset by peer")
        )
        client = OpenAIChatClient(openai_settings)

        with pytest.raises(LLMProviderError) as excinfo:
            client.complete([LLMMessage(role="user", content="Q?")])

        assert excinfo.value.status_code == 502
        # The raw transport error must not reach the API client.
        assert "Connection reset by peer" not in excinfo.value.message

    def test_rate_limit_error_is_wrapped(self, openai_settings, patched_openai) -> None:
        patched_openai.completions = _FakeCompletions(
            error=RuntimeError("Rate limit reached for gpt-4o-mini")
        )
        client = OpenAIChatClient(openai_settings)

        with pytest.raises(LLMProviderError):
            client.complete([LLMMessage(role="user", content="Q?")])

    def test_empty_completion_is_an_error(self, openai_settings, patched_openai) -> None:
        patched_openai.completions = _FakeCompletions(_FakeResponse("   "))
        client = OpenAIChatClient(openai_settings)

        with pytest.raises(LLMProviderError) as excinfo:
            client.complete([LLMMessage(role="user", content="Q?")])
        assert "empty response" in excinfo.value.message

    def test_missing_choices_is_an_error(self, openai_settings, patched_openai) -> None:
        empty = _FakeResponse("x")
        empty.choices = []
        patched_openai.completions = _FakeCompletions(empty)
        client = OpenAIChatClient(openai_settings)

        with pytest.raises(LLMProviderError):
            client.complete([LLMMessage(role="user", content="Q?")])

    def test_model_name_is_reported(self, openai_settings, patched_openai) -> None:
        patched_openai.completions = _FakeCompletions(_FakeResponse("ok"))
        assert OpenAIChatClient(openai_settings).model_name == "gpt-4o-mini"

    def test_client_satisfies_the_llm_protocol(self, openai_settings, patched_openai) -> None:
        from app.llm.client import LLMClient

        patched_openai.completions = _FakeCompletions(_FakeResponse("ok"))
        assert isinstance(OpenAIChatClient(openai_settings), LLMClient)

    def test_unknown_provider_is_rejected(self) -> None:
        with pytest.raises(LLMConfigurationError) as excinfo:
            client_module.get_llm_client(
                Settings.model_construct(llm_provider="mystery", llm_model="x")
            )
        assert "mystery" in excinfo.value.message
