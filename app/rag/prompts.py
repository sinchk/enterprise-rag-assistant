"""Prompt templates and context construction.

Grounding is enforced by three mechanisms working together:

1. The system prompt states the grounding contract explicitly and forbids
   using prior knowledge.
2. Only the retrieved chunks are included as context, each tagged with a stable
   ``[S#]`` label plus document and page.
3. The pipeline treats an empty retrieval result as a definitive "not found"
   answer and never calls the model at all, so the model cannot improvise.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence

from app.llm.client import LLMMessage
from app.retrieval.models import RetrievedChunk

logger = logging.getLogger(__name__)

_CITATION_RE = re.compile(r"\[S(\d+)\]")

SYSTEM_PROMPT = """\
You are a document question-answering assistant for an enterprise knowledge \
base. You answer strictly from the CONTEXT supplied by the user.

Rules you must follow:
1. Use ONLY the information in the CONTEXT. Treat it as the single source of \
truth for this conversation.
2. Never use prior knowledge, assumptions, or general facts that are not \
stated in the CONTEXT, even if you are confident they are correct.
3. If the CONTEXT does not contain the answer, reply exactly: \
"I could not find this information in the provided documents." Do not guess, \
do not speculate, and do not offer partial or adjacent information.
4. If the CONTEXT only partially answers the question, state precisely what is \
covered and what is missing.
5. Be concise and factual: short paragraphs or bullet points, no filler, no \
restating the question.
6. Cite your evidence: use the [S#] tags inline for specific facts and end \
your reply with a line of the form "Source: <document>, page <n>". Only cite \
the tags you actually used.
7. If the context contains conflicting statements, say so and cite both.
"""

USER_PROMPT_TEMPLATE = """\
CONTEXT:
{context}

Question: {question}

Answer using only the CONTEXT above, and finish with a "Source:" line.\
"""

NO_ANSWER_MESSAGE = "I could not find this information in the provided documents."


def build_context(chunks: Sequence[RetrievedChunk], max_chars: int = 12000) -> str:
    """Render retrieved chunks as a labelled, page-attributed context block.

    Args:
        chunks: Retrieved chunks, best match first.
        max_chars: Hard cap on the rendered context so prompt size (and cost)
            stay bounded. Chunks are added in rank order until the cap is
            reached; the highest ranked chunk is always included.

    Returns:
        A string with one ``[S#]``-tagged block per chunk, or an empty string
        when no chunks were retrieved.
    """
    if not chunks:
        return ""

    blocks: list[str] = []
    used = 0
    for chunk in chunks:
        block = (
            f"{chunk.label} document: {chunk.document} | page: {chunk.page}\n"
            f"{chunk.text.strip()}"
        )
        if blocks and used + len(block) > max_chars:
            logger.warning(
                "Context truncated after %s chunk(s) (%s characters).",
                len(blocks),
                used,
            )
            break
        blocks.append(block)
        used += len(block)
    return "\n\n".join(blocks)


def build_messages(
    question: str, chunks: Sequence[RetrievedChunk], max_context_chars: int = 12000
) -> list[LLMMessage]:
    """Build the chat messages for a grounded question."""
    context = build_context(chunks, max_chars=max_context_chars)
    user_prompt = USER_PROMPT_TEMPLATE.format(context=context, question=question.strip())
    return [
        LLMMessage(role="system", content=SYSTEM_PROMPT),
        LLMMessage(role="user", content=user_prompt),
    ]


def referenced_sources(
    answer: str, chunks: Sequence[RetrievedChunk]
) -> list[RetrievedChunk]:
    """Return the chunks the model cited, falling back to all of them.

    The answer is inspected for ``[S#]`` tags. When the model cites nothing,
    every chunk supplied as context is reported as a source, because those
    chunks are what the answer was derived from.
    """
    cited: list[RetrievedChunk] = []
    seen: set[str] = set()
    for token in _CITATION_RE.findall(answer):
        label = f"[S{token}]"
        for chunk in chunks:
            if chunk.label == label and chunk.chunk_id not in seen:
                seen.add(chunk.chunk_id)
                cited.append(chunk)
                break
    return cited or list(chunks)
