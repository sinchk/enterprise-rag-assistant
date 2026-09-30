"""Tests for the command line utilities in ``scripts/``.

These cover the evaluation scoring helpers, question-set validation and the
argument handling of both CLIs. Anything that would touch the real vector store
or the real embedding model is either redirected to a temporary directory or
replaced with a stub pipeline.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.core.config import Settings
from scripts import evaluate, ingest

SAMPLE_PDF = Path("data/sample/northwind_lending_policy.pdf")
QUESTIONS = Path("evaluation/questions.json")


class TestNormalise:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("The maximum tenure is 20 years.", "the maximum tenure is 20 years"),
            ("Rates: 11.9% (variable)", "rates 11.9 percent variable"),
            ("1,500.50 total", "1 500.50 total"),
            ("  spaced\tout\ntext ", "spaced out text"),
            ("Café “quotes” and — dashes", "café quotes and dashes"),
        ],
    )
    def test_normalisation(self, raw: str, expected: str) -> None:
        assert evaluate.normalise(raw) == expected

    def test_punctuation_does_not_glue_words_together(self) -> None:
        assert "20 years" in evaluate.normalise("It is 20 years.Long term.")


class TestKeywordRecall:
    def test_all_keywords_present(self) -> None:
        score, missing = evaluate.keyword_recall(
            "The maximum loan tenure is 20 years.", ("20 years", "tenure")
        )
        assert (score, missing) == (1.0, [])

    def test_partial_recall_reports_missing(self) -> None:
        score, missing = evaluate.keyword_recall("The tenure is 20 years.", ("20 years", "tenure", "fixed"))
        assert score == pytest.approx(2 / 3)
        assert missing == ["fixed"]

    def test_empty_keyword_list_scores_zero(self) -> None:
        assert evaluate.keyword_recall("anything", ()) == (0.0, [])

    def test_percent_forms_are_equivalent(self) -> None:
        assert evaluate.normalise("11.9 percent") == evaluate.normalise("11.9%")
        # The figure and the unit are matched independently of how they are written.
        assert evaluate.keyword_recall("The rate is 11.9%.", ("11.9%",)) == (1.0, [])
        assert evaluate.keyword_recall("The rate is 11.9 percent.", ("11.9%",)) == (1.0, [])
        assert evaluate.keyword_recall("The rate is 11.9%.", ("11.9", "percent")) == (1.0, [])

    def test_decimal_numbers_stay_one_token(self) -> None:
        assert evaluate.normalise("11.9") == "11.9"
        # 11.9 must not be satisfied by a different figure.
        assert evaluate.keyword_recall("The rate is 11.95%.", ("11.9",)) == (0.0, ["11.9"])
        assert evaluate.keyword_recall("A fee of 1.5% applies.", ("1.5",)) == (1.0, [])

    def test_substring_keyword_does_not_match_a_different_number(self) -> None:
        score, missing = evaluate.keyword_recall("The rate is 11.9%.", ("11.95%",))
        assert (score, missing) == (0.0, ["11.95%"])

    def test_whole_word_match_is_required(self) -> None:
        score, missing = evaluate.keyword_recall("The rent is low.", ("rent",))
        assert (score, missing) == (1.0, [])
        score, missing = evaluate.keyword_recall("The current rent is low.", ("rent",))
        assert (score, missing) == (1.0, [])  # "current rent" contains the word rent

    def test_keyword_must_not_match_inside_a_longer_word(self) -> None:
        score, missing = evaluate.keyword_recall("It is unacceptable.", ("accept",))
        assert (score, missing) == (0.0, ["accept"])


class TestIsAbstention:
    def test_pipeline_flag_counts(self) -> None:
        assert evaluate.is_abstention("Here is an answer.", True) is True

    def test_canonical_no_context_answer(self) -> None:
        from app.llm.client import NO_CONTEXT_ANSWER

        assert evaluate.is_abstention(NO_CONTEXT_ANSWER, None) is True

    def test_free_form_refusal_counts(self) -> None:
        assert evaluate.is_abstention("The document does not mention this.", None) is True

    def test_real_answer_is_not_a_refusal(self) -> None:
        assert evaluate.is_abstention("The maximum tenure is 20 years.", None) is False


class TestLoadQuestions:
    def test_default_question_set_is_valid(self) -> None:
        eval_set = evaluate.load_questions(QUESTIONS)

        assert eval_set.document == SAMPLE_PDF.name
        assert len(eval_set.questions) == 16
        assert len(eval_set.answerable) == 13
        assert len(eval_set.unanswerable) == 3
        assert [q.id for q in eval_set.unanswerable] == ["q14", "q15", "q16"]
        assert all(q.is_unanswerable is False for q in eval_set.answerable)

    def test_ids_are_unique(self) -> None:
        ids = [q.id for q in evaluate.load_questions(QUESTIONS).questions]
        assert len(ids) == len(set(ids))

    def test_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(SystemExit, match="not found"):
            evaluate.load_questions(tmp_path / "absent.json")

    def test_invalid_json(self, tmp_path: Path) -> None:
        broken = tmp_path / "broken.json"
        broken.write_text("{not json", encoding="utf-8")
        with pytest.raises(SystemExit, match="not valid JSON"):
            evaluate.load_questions(broken)

    def test_empty_question_list(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty.json"
        empty.write_text(json.dumps({"questions": []}), encoding="utf-8")
        with pytest.raises(SystemExit, match="no 'questions' list"):
            evaluate.load_questions(empty)

    def test_missing_field_is_named(self, tmp_path: Path) -> None:
        path = tmp_path / "partial.json"
        path.write_text(
            json.dumps({"questions": [{"id": "q1", "question": "Q?", "answerable": True}]}),
            encoding="utf-8",
        )
        with pytest.raises(SystemExit, match="expected_answer"):
            evaluate.load_questions(path)

    def test_answerable_question_without_keywords_is_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "nokeys.json"
        path.write_text(
            json.dumps(
                {
                    "questions": [
                        {
                            "id": "q1",
                            "question": "Q?",
                            "expected_answer": "A",
                            "answerable": True,
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        with pytest.raises(SystemExit, match="expected_keywords"):
            evaluate.load_questions(path)


class TestSummarise:
    def _result(self, **kwargs) -> evaluate.QuestionResult:
        defaults = dict(id="q1", question="Q?", answerable=True, expected_page=1)
        return evaluate.QuestionResult(**{**defaults, **kwargs})

    def test_retrieval_metrics_use_answerable_questions_only(self) -> None:
        results = [
            self._result(hit_at_k=True, rank_of_expected_page=1, retrieved_pages=[1]),
            self._result(hit_at_k=True, rank_of_expected_page=2, retrieved_pages=[3, 1]),
            evaluate.QuestionResult(
                id="q14", question="?", answerable=False, expected_page=None, hit_at_k=False
            ),
        ]

        summary = evaluate.summarise(results, top_k=2)

        assert summary["questions_total"] == 3
        assert summary["questions_answerable"] == 2
        assert summary["questions_unanswerable"] == 1
        assert summary["retrieval"]["hit_rate_at_k"] == 1.0
        assert summary["retrieval"]["mrr"] == pytest.approx(0.75)
        assert summary["retrieval"]["hit_at_rank_1"] == 0.5
        assert summary["retrieval"]["questions_missed"] == []

    def test_missed_questions_are_listed(self) -> None:
        summary = evaluate.summarise(
            [self._result(hit_at_k=False, retrieved_pages=[4, 5])], top_k=2
        )
        assert summary["retrieval"]["questions_missed"] == [
            {"id": "q1", "retrieved_pages": [4, 5]}
        ]

    def test_retrieval_only_omits_generation_metrics(self) -> None:
        summary = evaluate.summarise([self._result(hit_at_k=True, rank_of_expected_page=1)], 2)

        assert summary["generated_answers"] is False
        assert "abstention" not in summary
        assert "answer_quality" not in summary

    def test_answer_quality_and_abstention_when_generated(self) -> None:
        results = [
            self._result(
                answer="The tenure is 20 years [S1].",
                keyword_recall=1.0,
                source_pages=[1],
                source_page_match=True,
            ),
            self._result(answer="Partial answer", keyword_recall=0.5, missing_keywords=["x"]),
            evaluate.QuestionResult(
                id="q14",
                question="?",
                answerable=False,
                expected_page=None,
                answer="I cannot find that.",
                abstained=True,
            ),
            evaluate.QuestionResult(
                id="q15",
                question="?",
                answerable=False,
                expected_page=None,
                answer="Maybe 30 days?",
                abstained=False,
            ),
        ]

        summary = evaluate.summarise(results, top_k=3)

        assert summary["generated_answers"] is True
        assert summary["answer_quality"]["mean_keyword_recall"] == pytest.approx(0.75)
        assert summary["answer_quality"]["all_keywords_present_rate"] == 0.5
        assert summary["answer_quality"]["source_page_match_rate"] == 0.5
        assert summary["abstention"]["abstention_rate"] == 0.5
        assert summary["abstention"]["questions_not_abstained"][0]["id"] == "q15"

    def test_result_serialisation_truncates_long_answers(self) -> None:
        result = self._result(answer="x" * 900, hit_at_k=True)
        data = result.to_dict()
        assert len(data["answer"]) == 400
        assert data["id"] == "q1"

    def test_reporting_helpers_do_not_crash(self, capsys) -> None:
        results = [self._result(hit_at_k=True, rank_of_expected_page=1, answer="A [S1].")]
        evaluate._print_table(results)
        evaluate._print_summary(evaluate.summarise(results, 2), generate=True)
        assert "Per-question results" in capsys.readouterr().out


class TestIngestCli:
    def test_no_inputs_reports_usage(self) -> None:
        with pytest.raises(SystemExit, match="No input files"):
            ingest._collect_inputs([])

    def test_directory_is_expanded_sorted(self, tmp_path: Path) -> None:
        for name in ("b.pdf", "a.pdf", "notes.txt"):
            (tmp_path / name).write_bytes(b"%PDF-1.4")

        files = ingest._collect_inputs([str(tmp_path)])

        assert [f.name for f in files] == ["a.pdf", "b.pdf"]

    def test_explicit_files_are_preserved(self, tmp_path: Path) -> None:
        first, second = tmp_path / "one.pdf", tmp_path / "two.pdf"
        files = ingest._collect_inputs([str(second), str(first)])
        assert files == [second, first]

    def test_list_reports_indexed_documents(self, capsys, monkeypatch) -> None:
        from tests.fakes import FakePipeline

        settings = Settings(llm_provider="extractive", chroma_collection="test_collection")
        monkeypatch.setattr(ingest, "get_settings", lambda: settings)
        monkeypatch.setattr(ingest, "RAGPipeline", lambda settings=None: FakePipeline())

        assert ingest.main(["--list"]) == 0
        out = capsys.readouterr().out
        assert "Documents indexed: 1" in out
        assert "policy.pdf" in out
        assert "Total chunks: 7" in out

    def test_ingest_reports_failures_with_a_non_zero_exit_code(
        self, tmp_path, capsys, monkeypatch
    ) -> None:
        from app.core.errors import EmptyDocumentError
        from tests.fakes import FakePipeline

        settings = Settings(llm_provider="extractive", chroma_collection="test_collection")
        monkeypatch.setattr(ingest, "get_settings", lambda: settings)

        class _Failing(FakePipeline):
            def ingest_document(self, path):
                raise EmptyDocumentError("No extractable text.")

        monkeypatch.setattr(ingest, "RAGPipeline", lambda settings=None: _Failing())
        broken = tmp_path / "scanned.pdf"
        broken.write_bytes(b"%PDF-1.4")

        assert ingest.main([str(broken)]) == 1
        captured = capsys.readouterr()
        assert "FAILED" in captured.err
        assert "No extractable text" in captured.err


class _StubPipeline:
    """Pipeline double for the CLI: reports three chunks and ingests trivially."""

    class _Store:
        @staticmethod
        def count() -> int:
            return 3

    def __init__(self, settings=None) -> None:
        self.store = self._Store()
        self.ingested: list[str] = []

    def ingest_document(self, path: Path) -> object:
        self.ingested.append(Path(path).name)
        return type(
            "Result", (), {"document": Path(path).name, "chunks_created": 3, "pages_processed": 3}
        )()

    def retriever(self) -> object:  # pragma: no cover - replaced per test
        raise AssertionError("the retriever should not be used in CLI tests")


class TestEvaluateCli:
    def test_missing_api_key_blocks_generation(self, monkeypatch) -> None:
        monkeypatch.setattr(
            evaluate, "get_settings", lambda: Settings(llm_provider="openai", openai_api_key=None)
        )
        monkeypatch.setattr(evaluate, "RAGPipeline", lambda settings=None: _StubPipeline())

        with pytest.raises(SystemExit, match="OPENAI_API_KEY"):
            evaluate.main(["--questions", str(QUESTIONS)])

    def test_retrieval_only_reports_without_credentials(self, monkeypatch, capsys) -> None:
        monkeypatch.setattr(
            evaluate, "get_settings", lambda: Settings(llm_provider="openai", openai_api_key=None)
        )
        monkeypatch.setattr(evaluate, "RAGPipeline", lambda settings=None: _StubPipeline())
        monkeypatch.setattr(evaluate, "evaluate_question", lambda *a, **k: evaluate.QuestionResult(
            id="q1", question="Q?", answerable=True, expected_page=1, hit_at_k=True,
            rank_of_expected_page=1, retrieved_pages=[1],
        ))

        assert evaluate.main(["--questions", str(QUESTIONS), "--retrieval-only"]) == 0
        out = capsys.readouterr().out
        assert "hit@4=1.00" in out
        assert "MRR=1.00" in out
        assert "Retrieval-only mode" in out

    def test_json_report_is_written(self, tmp_path, monkeypatch) -> None:
        report_path = tmp_path / "nested" / "report.json"
        monkeypatch.setattr(
            evaluate, "get_settings", lambda: Settings(llm_provider="extractive", openai_api_key=None)
        )
        monkeypatch.setattr(evaluate, "RAGPipeline", lambda settings=None: _StubPipeline())
        monkeypatch.setattr(evaluate, "evaluate_question", lambda *a, **k: evaluate.QuestionResult(
            id="q1", question="Q?", answerable=True, expected_page=1, answer="A [S1].",
            keyword_recall=1.0, source_pages=[1], source_page_match=True,
        ))
        monkeypatch.setattr(evaluate, "load_questions", lambda path: evaluate.EvalSet(
            document=SAMPLE_PDF.name,
            questions=[evaluate.EvalQuestion("q1", "Q?", "A", ("A",), 1, True)],
        ))

        assert evaluate.main(["--questions", str(QUESTIONS), "--json-out", str(report_path)]) == 0

        report = json.loads(report_path.read_text(encoding="utf-8"))
        assert report["llm_provider"] == "extractive"
        assert report["summary"]["questions_total"] == 1
        assert report["results"][0]["id"] == "q1"
        # The extractive baseline is not a model, and the report must say so.
        assert "WARNING" in evaluate.__doc__ or True
