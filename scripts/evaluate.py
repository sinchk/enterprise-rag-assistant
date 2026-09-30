"""Evaluate retrieval and answer quality against a labelled question set.

Usage::

    # Index the sample document first, then evaluate both stages
    python scripts/ingest.py data/sample/northwind_lending_policy.pdf
    python scripts/evaluate.py

    # Retrieval only, no LLM and no API key required
    python scripts/evaluate.py --retrieval-only

    # Ingest on the fly and write a machine-readable report
    python scripts/evaluate.py --document data/sample/northwind_lending_policy.pdf \
        --json-out evaluation/report.json

Methodology and limitations are documented in the README. In short: this is a
transparent lexical baseline, not a rigorous benchmark. There is no LLM judge
and no human annotation, so the numbers describe behaviour on 16 hand-written
questions and must not be presented as statistically meaningful.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import mean
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import configure_logging, get_settings  # noqa: E402
from app.core.errors import AppError  # noqa: E402
from app.llm.client import NO_CONTEXT_ANSWER  # noqa: E402
from app.rag.pipeline import RAGPipeline  # noqa: E402

logger = logging.getLogger("evaluate")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QUESTIONS = PROJECT_ROOT / "evaluation" / "questions.json"
_PUNCTUATION_RE = re.compile(r"[^\w\s%/-]")
_WHITESPACE_RE = re.compile(r"\s+")
_DECIMAL_RE = re.compile(r"(?<=\d)\.(?=\d)")
# Stands in for a decimal point while punctuation is stripped. It must consist
# of word characters, otherwise the punctuation filter would remove it too.
# The sequence is deliberately unlikely to occur in prose.
_DECIMAL_GUARD = "qzqzqzq"


# ---------------------------------------------------------------------------
# Question set
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class EvalQuestion:
    """One labelled evaluation question."""

    id: str
    question: str
    expected_answer: str
    expected_keywords: tuple[str, ...]
    expected_page: int | None
    answerable: bool

    @property
    def is_unanswerable(self) -> bool:
        """Whether the document is expected not to contain the answer."""
        return not self.answerable


@dataclass(frozen=True, slots=True)
class EvalSet:
    """A parsed evaluation set."""

    document: str
    questions: list[EvalQuestion]

    @property
    def answerable(self) -> list[EvalQuestion]:
        """Questions whose answer is in the document."""
        return [q for q in self.questions if q.answerable]

    @property
    def unanswerable(self) -> list[EvalQuestion]:
        """Questions whose answer is absent from the document."""
        return [q for q in self.questions if not q.answerable]


def load_questions(path: Path) -> EvalSet:
    """Load and validate the evaluation set from JSON.

    Raises:
        SystemExit: If the file is missing, malformed or internally inconsistent
            (for example an answerable question without keywords).
    """
    if not path.exists():
        raise SystemExit(f"Evaluation file not found: {path}")

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Evaluation file is not valid JSON: {exc}") from exc

    raw_questions = payload.get("questions")
    if not isinstance(raw_questions, list) or not raw_questions:
        raise SystemExit("Evaluation file contains no 'questions' list.")

    questions: list[EvalQuestion] = []
    for index, raw in enumerate(raw_questions, start=1):
        missing = {"id", "question", "expected_answer", "answerable"} - set(raw)
        if missing:
            raise SystemExit(
                f"Question #{index} is missing field(s): {sorted(missing)}"
            )
        answerable = bool(raw["answerable"])
        keywords = tuple(str(k).lower() for k in raw.get("expected_keywords", []))
        if answerable and not keywords:
            raise SystemExit(
                f"Question '{raw['id']}' is answerable but has no expected_keywords, "
                "so its answer quality cannot be measured."
            )
        if answerable and raw.get("expected_page") is None:
            raise SystemExit(
                f"Question '{raw['id']}' is answerable but has no expected_page, "
                "so its retrieval quality cannot be measured."
            )
        questions.append(
            EvalQuestion(
                id=str(raw["id"]),
                question=str(raw["question"]),
                expected_answer=str(raw["expected_answer"]),
                expected_keywords=keywords,
                expected_page=raw.get("expected_page"),
                answerable=answerable,
            )
        )

    return EvalSet(document=str(payload.get("document", "")), questions=questions)


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------


def normalise(text: str) -> str:
    """Lowercase and normalise text for whole-word keyword matching.

    Two details matter for numeric answers:

    * A decimal point between two digits is preserved, so "11.9" stays a single
      token instead of being split into "11 9", which would make it
      unmatchable against an answer that contains the same figure.
    * "%" is rewritten to the word "percent", so an answer phrased as "11.9%"
      matches a keyword "percent" and vice versa.

    Punctuation is replaced by a space rather than deleted, so "20 years." still
    contains the whole-word token "years" instead of being glued to the next
    word. Whitespace is then collapsed to single spaces.
    """
    lowered = text.lower().replace("\u2019", "'")
    protected = _DECIMAL_RE.sub(_DECIMAL_GUARD, lowered)
    stripped = _PUNCTUATION_RE.sub(" ", protected)
    folded = stripped.replace("%", " percent ")
    collapsed = _WHITESPACE_RE.sub(" ", folded)
    return collapsed.replace(_DECIMAL_GUARD, ".").strip()


def keyword_recall(answer: str, keywords: tuple[str, ...]) -> tuple[float, list[str]]:
    """Fraction of expected keywords present in the answer.

    Matching is whole-word based on normalised text, so "accept" does not match
    "unacceptable" while "tenure" does match "the loan tenure limit". Returns the
    score and the list of missing keywords.
    """
    if not keywords:
        return 0.0, []
    haystack = f" {normalise(answer)} "
    missing = [kw for kw in keywords if f" {normalise(kw)} " not in haystack]
    return (len(keywords) - len(missing)) / len(keywords), missing


def is_abstention(answer: str, no_answer_flag: bool | None) -> bool:
    """Whether an answer states that the information was not found.

    ``no_answer_flag`` is the pipeline's own signal; the text is also checked so
    that a free-form refusal from a model is still counted. Both the reference
    answer and the candidate are normalised, so a dash in the canonical message
    cannot prevent a match.
    """
    if no_answer_flag:
        return True
    normalised = normalise(answer)
    return (
        normalise(NO_CONTEXT_ANSWER) in normalised
        or "not stated in the document" in normalised
        or ("does not" in normalised and "document" in normalised)
    )


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


@dataclass
class QuestionResult:
    """Per-question outcome."""

    id: str
    question: str
    answerable: bool
    expected_page: int | None
    retrieved_pages: list[int] = field(default_factory=list)
    top_score: float | None = None
    rank_of_expected_page: int | None = None
    hit_at_k: bool = False
    answer: str = ""
    keyword_recall: float = 0.0
    missing_keywords: list[str] = field(default_factory=list)
    source_pages: list[int] = field(default_factory=list)
    source_page_match: bool = False
    abstained: bool = False
    duration_ms: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view (answer truncated for readability)."""
        data = asdict(self)
        data["answer"] = self.answer[:400]
        return data


def evaluate_question(
    pipeline: RAGPipeline, item: EvalQuestion, top_k: int, generate: bool
) -> QuestionResult:
    """Run one question through the pipeline and score it."""
    import time

    started = time.perf_counter()
    chunks = pipeline.retriever.retrieve(item.question, top_k=top_k)

    result = QuestionResult(
        id=item.id,
        question=item.question,
        answerable=item.answerable,
        expected_page=item.expected_page,
        retrieved_pages=[chunk.page for chunk in chunks],
        top_score=chunks[0].score if chunks else None,
    )

    if item.expected_page is not None:
        if item.expected_page in result.retrieved_pages:
            result.rank_of_expected_page = result.retrieved_pages.index(
                item.expected_page
            ) + 1
        result.hit_at_k = result.rank_of_expected_page is not None

    if not generate:
        result.duration_ms = int((time.perf_counter() - started) * 1000)
        return result

    answer = pipeline.ask(item.question, top_k=top_k)
    result.answer = answer.answer
    result.source_pages = [int(source["page"]) for source in answer.sources]
    result.abstained = is_abstention(answer.answer, answer.no_answer)
    if item.answerable:
        result.keyword_recall, result.missing_keywords = keyword_recall(
            answer.answer, item.expected_keywords
        )
        result.source_page_match = item.expected_page in result.source_pages
    result.duration_ms = int((time.perf_counter() - started) * 1000)
    return result


def summarise(results: list[QuestionResult], top_k: int) -> dict[str, Any]:
    """Aggregate per-question results into a metric summary.

    Only metrics that are meaningful for the available questions are reported:
    retrieval metrics use answerable questions, abstention uses unanswerable
    ones, and answer metrics require a generation step.
    """
    answerable = [r for r in results if r.answerable]
    unanswerable = [r for r in results if not r.answerable]
    scored_answers = [r for r in answerable if r.answer]
    generated = any(r.answer for r in results)

    summary: dict[str, Any] = {
        "questions_total": len(results),
        "questions_answerable": len(answerable),
        "questions_unanswerable": len(unanswerable),
        "generated_answers": generated,
    }

    if answerable:
        ranks = [
            r.rank_of_expected_page
            for r in answerable
            if r.rank_of_expected_page is not None
        ]
        summary["retrieval"] = {
            "top_k": top_k,
            "hit_rate_at_k": round(mean([r.hit_at_k for r in answerable]), 4),
            "mrr": round(mean([1.0 / rank for rank in ranks]), 4) if ranks else 0.0,
            "hit_at_rank_1": round(
                mean([r.rank_of_expected_page == 1 for r in answerable]), 4
            ),
            "questions_missed": [
                {"id": r.id, "retrieved_pages": r.retrieved_pages}
                for r in answerable
                if not r.hit_at_k
            ],
        }

    if unanswerable and generated:
        # Abstention is only observable when the model was actually called.
        summary["abstention"] = {
            "abstention_rate": round(mean([r.abstained for r in unanswerable]), 4),
            "questions_not_abstained": [
                {"id": r.id, "answer": r.answer[:200]}
                for r in unanswerable
                if not r.abstained
            ],
        }

    if scored_answers:
        summary["answer_quality"] = {
            "mean_keyword_recall": round(
                mean([r.keyword_recall for r in scored_answers]), 4
            ),
            "all_keywords_present_rate": round(
                mean([r.keyword_recall == 1.0 for r in scored_answers]), 4
            ),
            "source_page_match_rate": round(
                mean([r.source_page_match for r in scored_answers]), 4
            ),
            "questions_below_full_recall": [
                {"id": r.id, "recall": r.keyword_recall, "missing": r.missing_keywords}
                for r in scored_answers
                if r.keyword_recall < 1.0
            ],
        }

    return summary


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _print_table(results: list[QuestionResult]) -> None:
    """Print one line per question."""
    header = f"{'id':<5} {'pages':<12} {'hit':<4} {'rec':<5} {'src':<4} {'flag':<6} question"
    print("\nPer-question results")
    print("-" * len(header))
    print(header)
    print("-" * len(header))
    for result in results:
        pages = ",".join(str(page) for page in result.retrieved_pages) or "-"
        hit = "yes" if result.hit_at_k else ("n/a" if result.expected_page is None else "NO")
        recall = f"{result.keyword_recall:.2f}" if result.answer else "-"
        src = "ok" if result.source_page_match else ("-" if not result.answer else "NO")
        flag = "skip" if result.expected_page is None else ("abstain" if result.abstained else "answer")
        print(
            f"{result.id:<5} {pages:<12} {hit:<4} {recall:<5} {src:<4} {flag:<6} "
            f"{result.question[:58]}"
        )


def _print_summary(summary: dict[str, Any], generate: bool) -> None:
    """Print the aggregate metrics with an explanation of each one."""
    print("\nSummary")
    print("=" * 60)
    retrieval = summary.get("retrieval")
    if retrieval:
        print(
            f"Retrieval   hit@{retrieval['top_k']}={retrieval['hit_rate_at_k']:.2f}  "
            f"MRR={retrieval['mrr']:.2f}  "
            f"top-1={retrieval['hit_at_rank_1']:.2f}  "
            f"(share of answerable questions whose expected page is retrieved)"
        )
        for missed in retrieval["questions_missed"]:
            print(f"            missed: {missed['id']} -> pages {missed['retrieved_pages']}")
    abstention = summary.get("abstention")
    if abstention:
        print(
            f"Abstention  rate={abstention['abstention_rate']:.2f}  "
            f"(share of unanswerable questions correctly refused)"
        )
        for item in abstention["questions_not_abstained"]:
            print(f"            not refused: {item['id']} -> {item['answer'][:100]}")
    quality = summary.get("answer_quality")
    if quality:
        print(
            f"Answers     keyword_recall={quality['mean_keyword_recall']:.2f}  "
            f"all_keywords={quality['all_keywords_present_rate']:.2f}  "
            f"source_page_match={quality['source_page_match_rate']:.2f}"
        )
        for item in quality["questions_below_full_recall"]:
            print(
                f"            partial: {item['id']} recall={item['recall']:.2f} "
                f"missing={item['missing']}"
            )
    print("=" * 60)
    if not generate:
        print("Retrieval-only mode: answer quality was not measured (no LLM call).")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """Run the evaluation CLI and return a process exit code."""
    settings = get_settings()
    parser = argparse.ArgumentParser(
        description="Evaluate retrieval and answer quality of the RAG assistant.",
    )
    parser.add_argument(
        "--questions", type=Path, default=DEFAULT_QUESTIONS, help="Question set JSON."
    )
    parser.add_argument(
        "--document",
        type=Path,
        default=None,
        help="PDF to ingest before evaluating (defaults to the document named in "
        "the question set, resolved under data/sample).",
    )
    parser.add_argument(
        "--top-k", type=int, default=None, help="Chunks retrieved per question."
    )
    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help="Skip the LLM and report retrieval metrics only.",
    )
    parser.add_argument(
        "--json-out", type=Path, default=None, help="Write the full report as JSON."
    )
    args = parser.parse_args(argv)

    configure_logging(settings.log_level)
    eval_set = load_questions(args.questions)
    top_k = args.top_k or settings.top_k
    generate = not args.retrieval_only

    pipeline = RAGPipeline(settings=settings)

    document = args.document
    if document is None and eval_set.document:
        document = PROJECT_ROOT / "data" / "sample" / eval_set.document
    if document is not None and document.exists():
        result = pipeline.ingest_document(document)
        print(
            f"Ingested {result.document}: {result.chunks_created} chunk(s) "
            f"from {result.pages_processed} page(s)."
        )
    elif pipeline.store.count() == 0:
        raise SystemExit(
            "No documents are indexed and none could be loaded. Run\n"
            "  python scripts/ingest.py data/sample/northwind_lending_policy.pdf"
        )

    if generate and settings.llm_provider == "extractive":
        print(
            "\nWARNING: LLM_PROVIDER=extractive. Answers are produced by the "
            "non-LLM lexical baseline, so the answer metrics below are a smoke "
            "test of the plumbing, not a measure of model quality."
        )
    if generate and not settings.has_openai_credentials and settings.llm_provider == "openai":
        raise SystemExit(
            "OPENAI_API_KEY is not set. Configure it, or re-run with "
            "--retrieval-only for a keyless retrieval evaluation."
        )

    print(f"\nEvaluating {len(eval_set.questions)} question(s), top_k={top_k}")
    results: list[QuestionResult] = []
    for item in eval_set.questions:
        try:
            results.append(evaluate_question(pipeline, item, top_k, generate))
        except AppError as exc:
            logger.error("Question %s failed: %s", item.id, exc.message)
            results.append(
                QuestionResult(
                    id=item.id,
                    question=item.question,
                    answerable=item.answerable,
                    expected_page=item.expected_page,
                    answer=f"ERROR: {exc.message}",
                )
            )

    summary = summarise(results, top_k=top_k)
    _print_table(results)
    _print_summary(summary, generate)

    if args.json_out:
        report = {
            "document": str(document) if document else eval_set.document,
            "llm_provider": settings.llm_provider,
            "llm_model": settings.llm_model,
            "embedding_model": settings.embedding_model,
            "chunk_size": settings.chunk_size,
            "chunk_overlap": settings.chunk_overlap,
            "top_k": top_k,
            "summary": summary,
            "results": [r.to_dict() for r in results],
        }
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nWrote the full report to {args.json_out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
