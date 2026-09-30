"""Pluggable LLM clients.

Two providers are shipped:

``openai``
    A hosted chat-completion model. This is the default and the provider used
    for real answers.

``extractive``
    A **non-LLM** deterministic baseline that returns the sentences from the
    supplied context with the highest lexical overlap with the question. It
    exists so the API, the tests and the evaluation harness can run with no API
    key and no network access. It is not a substitute for an LLM and its
    answers must not be reported as model quality.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Literal, Protocol, runtime_checkable

from app.core.config import Settings, get_settings
from app.core.errors import LLMConfigurationError, LLMProviderError

logger = logging.getLogger(__name__)

SentenceSplitter = re.compile(r"(?<=[.!?])\s+")
# Prompt scaffolding emitted by app.rag.prompts, not part of the document text.
_PROMPT_SCAFFOLD_RE = re.compile(
    r"^(CONTEXT:|\[S\d+\]\s*document:)", re.IGNORECASE
)


@dataclass(frozen=True, slots=True)
class LLMMessage:
    """A single chat message."""

    role: Literal["system", "user", "assistant"]
    content: str


@runtime_checkable
class LLMClient(Protocol):
    """Minimal contract required by the RAG pipeline."""

    def complete(self, messages: list[LLMMessage]) -> str:
        """Return the assistant's reply for the given conversation."""
        ...

    @property
    def model_name(self) -> str:
        """Identifier of the model answering, surfaced in API responses."""
        ...


class OpenAIChatClient:
    """Chat-completion client for the OpenAI API (or a compatible endpoint)."""

    def __init__(self, settings: Settings | None = None) -> None:
        """Create the client.

        Raises:
            LLMConfigurationError: If ``OPENAI_API_KEY`` is missing, or the
                ``openai`` package is not installed.
        """
        self._settings = settings or get_settings()
        if not self._settings.has_openai_credentials:
            raise LLMConfigurationError(
                "OPENAI_API_KEY is not configured. Add it to your .env file (see "
                ".env.example) or set LLM_PROVIDER=extractive to run without an "
                "LLM."
            )
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise LLMConfigurationError(
                "The openai package is not installed."
            ) from exc

        kwargs: dict[str, Any] = {
            "api_key": self._settings.openai_api_key.get_secret_value(),
            "timeout": self._settings.llm_timeout_seconds,
        }
        if self._settings.openai_base_url:
            kwargs["base_url"] = self._settings.openai_base_url

        self._client = OpenAI(**kwargs)
        logger.info("LLM provider initialised (model=%s).", self.model_name)

    def complete(self, messages: list[LLMMessage]) -> str:
        """Send a chat completion request and return the assistant text.

        Raises:
            LLMProviderError: If the API call fails or returns no content.
        """
        try:
            response = self._client.chat.completions.create(
                model=self._settings.llm_model,
                messages=[
                    {"role": message.role, "content": message.content}
                    for message in messages
                ],
                temperature=self._settings.llm_temperature,
                max_tokens=self._settings.llm_max_tokens,
            )
        except Exception as exc:
            logger.error("LLM request failed: %s", exc)
            raise LLMProviderError(
                "The language model provider could not be reached. Please try "
                "again later."
            ) from exc

        content = response.choices[0].message.content if response.choices else None
        if not content or not content.strip():
            raise LLMProviderError(
                "The language model returned an empty response."
            )
        return content.strip()

    @property
    def model_name(self) -> str:
        """Configured chat model identifier."""
        return self._settings.llm_model


class ExtractiveBaselineClient:
    """Offline, non-LLM baseline that echoes the most relevant context lines.

    Intended for smoke tests, CI and keyless demos. It never invents text: it
    only ever returns sentences copied verbatim from the provided context, so
    the grounding guarantee still holds.

    Two transparent lexical rules replace the judgement a real LLM provides:

    * **Selection** - keep the sentences sharing the most non-stopword terms
      with the question.
    * **Abstention** - reply that the information was not found unless the
      selected sentences cover at least ``_MIN_COVERAGE`` of the question's
      content words. This is a bag-of-words heuristic, not semantic
      understanding, and it is a known limitation of this provider.
    """

    _STOPWORDS = frozenset(
        """a an the of to for in on at by is are was were be been being and or
        if what which who whom whose how when where why does do did done
        can could should would will shall may might must i we you they it this
        that these those there here with from as not no any all our your their
        his her its about into over under between during per have has had
        please tell me give explain much many long list name number get""".split()
    )
    #: Share of the question's content words that must appear in the answer.
    _MIN_COVERAGE = 0.6

    def __init__(self, max_sentences: int = 3) -> None:
        self._max_sentences = max_sentences

    def complete(self, messages: list[LLMMessage]) -> str:
        """Return the context sentences with the highest lexical overlap."""
        user_prompt = messages[-1].content if messages else ""
        context, question = _split_prompt(user_prompt)
        if not context:
            return NO_CONTEXT_ANSWER

        question_terms = self._terms(question)
        if not question_terms:
            return NO_CONTEXT_ANSWER

        sentences = [
            sentence.strip()
            for sentence in SentenceSplitter.split(context)
            if sentence.strip()
        ]
        if not sentences:
            return NO_CONTEXT_ANSWER

        scored = [
            (self._overlap(self._terms(sentence), question_terms), index, sentence)
            for index, sentence in enumerate(sentences)
        ]
        best_score = max(score for score, _, _ in scored)
        if best_score <= 0:
            return NO_CONTEXT_ANSWER

        selected = sorted(
            (item for item in scored if item[0] >= best_score),
            key=lambda item: (-item[0], item[1]),
        )[: self._max_sentences]
        selected.sort(key=lambda item: item[1])
        evidence = " ".join(sentence for _, _, sentence in selected)

        coverage = len(self._terms(evidence) & question_terms) / len(question_terms)
        if coverage < self._MIN_COVERAGE:
            logger.info(
                "Extractive baseline abstaining: question coverage %.2f < %.2f.",
                coverage,
                self._MIN_COVERAGE,
            )
            return NO_CONTEXT_ANSWER
        return evidence

    def _terms(self, text: str) -> set[str]:
        """Lowercase content words of a piece of text."""
        return {
            word
            for word in re.findall(r"[a-z0-9]+", text.lower())
            if word not in self._STOPWORDS and len(word) > 2
        }

    def _overlap(self, sentence_terms: set[str], question_terms: set[str]) -> int:
        """Count of shared content words, a transparent similarity proxy."""
        return len(sentence_terms & question_terms)

    @property
    def model_name(self) -> str:
        """Name reported for this provider."""
        return "extractive-baseline (non-LLM)"


NO_CONTEXT_ANSWER = (
    "I could not find this information in the provided documents."
)


def _split_prompt(user_prompt: str) -> tuple[str, str]:
    """Split the pipeline prompt into its context and question sections.

    The question is the first paragraph after the ``Question:`` marker, which
    is how :data:`app.rag.prompts.USER_PROMPT_TEMPLATE` is laid out; the
    trailing instruction paragraph and the ``CONTEXT:`` / ``[S#]`` scaffolding
    are discarded so only document text and the user's words are reasoned over.
    """
    context, _, remainder = user_prompt.partition("Question:")
    if not remainder.strip():
        return context.strip(), user_prompt.strip()

    question = remainder.strip().split("\n\n", maxsplit=1)[0].strip()
    kept = [
        line
        for line in context.splitlines()
        if line.strip() and not _PROMPT_SCAFFOLD_RE.match(line.strip())
    ]
    return "\n".join(kept).strip(), question


def get_llm_client(settings: Settings | None = None) -> LLMClient:
    """Instantiate the configured LLM provider.

    Raises:
        LLMConfigurationError: If the provider is unknown or misconfigured.
    """
    settings = settings or get_settings()
    if settings.llm_provider == "openai":
        return OpenAIChatClient(settings)
    if settings.llm_provider == "extractive":
        logger.warning(
            "LLM_PROVIDER=extractive: answers come from a non-LLM baseline "
            "that copies context sentences. Configure OPENAI_API_KEY for real "
            "grounded generation."
        )
        return ExtractiveBaselineClient()
    raise LLMConfigurationError(
        f"Unknown LLM provider '{settings.llm_provider}'. "
        "Use 'openai' or 'extractive'."
    )


def describe_provider(settings: Settings | None = None) -> str:
    """Return the model name to report without instantiating a client.

    Used for health checks and for the "not found" reply, which must work even
    when no API key is configured.
    """
    settings = settings or get_settings()
    if settings.llm_provider == "extractive":
        return "extractive-baseline (non-LLM)"
    return settings.llm_model

