"""`eval run` against a real database with a fake agent and judge: storage, follow-up threads,
baseline, regression gate and the report. Test-first."""

from __future__ import annotations

import pytest

import mmrag.db as db
from mmrag.agent.answer import HydratedAnswer
from mmrag.agent.graph import QueryRun
from mmrag.config import load_settings
from mmrag.eval.dataset import GoldenItem
from mmrag.eval.report import write_report
from mmrag.eval.runner import run_eval

pytestmark = pytest.mark.integration


class Metrics:
    def __init__(self, score):
        self.score = score

    def faithfulness(self, *a, **k): return self.score, {"claims": [{"statement": "s", "supported": False, "reason": "r"}]}
    def response_relevancy(self, *a, **k): return self.score, {}
    def context_precision(self, *a, **k): return self.score, {}
    def context_recall(self, *a, **k): return self.score, {}
    def factual_correctness(self, *a, **k): return self.score, {}


class Store:
    def locations(self, ids):
        return {}


ITEMS = [GoldenItem(question_id="g1", question="What is GQA?", qtype="conceptual", reference_answer="Shared KV heads.",
                    reference_ids=["d:text:1"], expected_tool_calls=["search_text"]),
         GoldenItem(question_id="g2", question="And MQA?", qtype="conceptual", reference_answer="One KV head.",
                    reference_ids=["d:text:1"], follows="g1"),
         GoldenItem(question_id="u1", question="Training cost?", qtype="unanswerable",
                    reference_answer="Not in the documents.", answerable=False)]


@pytest.fixture
def settings(fresh_db_url, base_config, write_config, tmp_path):
    base_config["paths"]["data_dir"] = str(tmp_path / "data")
    base_config["observability"].update(enabled=False, log_file=None)
    s = load_settings(write_config(base_config), {"DATABASE_URL": fresh_db_url})
    db.migrate(s)
    return s


def _ask_factory(seen):
    def ask(question, settings, thread_id=None, model=None):
        seen.append((question, thread_id))
        nf = question == "Training cost?"
        ans = HydratedAnswer.model_validate({"blocks": [{"type": "text", "content": {"markdown": "Not found." if nf else
                                                                                     "Shared KV heads."},
                                                         "citations": [] if nf else [{"id": "d:text:1", "locations": []}]}],
                                             "sources": []})
        return QueryRun(answer=ans, thread_id=thread_id or f"t-{len(seen)}", trace_id="a" * 32, model=model,
                        qtype="conceptual", rounds=1, tool_calls=[{"name": "search_text"}], input_tokens=10,
                        output_tokens=2, cost_usd=0.01, latency_ms=8000, validator_result="ok",
                        evidence=[{"id": "d:text:1", "text": "KV heads are shared."}], not_found=nf)
    return ask


def test_baseline_then_a_regression_fails_the_gate(settings):
    seen = []
    base = run_eval(settings, ITEMS, "v1", _ask_factory(seen), Metrics(0.9), Store(), "gpt-5.4-mini", baseline=True,
                    progress=lambda s: None)
    assert base.passed and base.summary["metrics"]["faithfulness"] == 0.9 and base.summary["metrics"]["refusal"] == 1.0
    assert seen[1] == ("And MQA?", "t-1")  # the follow-up continued g1's thread

    worse = run_eval(settings, ITEMS, "v1", _ask_factory([]), Metrics(0.8), Store(), "gpt-5.4-mini",
                     progress=lambda s: None)
    assert not worse.passed and any("faithfulness" in f for f in worse.failures)
    with db.connect(settings) as conn:
        runs = conn.execute("SELECT is_baseline, passed FROM eval_runs ORDER BY started_at").fetchall()
        n = conn.execute("SELECT count(*) FROM eval_results WHERE run_id = %s", (worse.run_id,)).fetchone()[0]
    assert runs == [(True, True), (False, False)] and n > 10

    page = write_report(settings, worse.run_id).read_text(encoding="utf-8")
    assert "FAILED" in page and "faithfulness" in page and "✗ s — r" in page  # judge reasons shown


def test_a_failing_question_scores_zero_and_the_run_goes_on(settings):
    def ask(question, settings, thread_id=None, model=None):
        raise RuntimeError("rate limited")

    out = run_eval(settings, ITEMS[:1], "v1", ask, Metrics(1.0), Store(), "m", progress=lambda s: None)
    assert out.summary["metrics"]["run_error"] == 0.0


def test_exhausted_credits_abort_the_run(settings):
    from mmrag.eval.runner import QuotaExhausted

    calls = []

    def ask(question, settings, thread_id=None, model=None):
        calls.append(question)
        raise QuotaExhausted("insufficient_quota")

    out = run_eval(settings, ITEMS, "v1", ask, Metrics(1.0), Store(), "m", baseline=True, progress=lambda s: None)
    assert len(calls) == 1 and not out.passed and "aborted" in out.failures[0]
    with db.connect(settings) as conn:
        assert conn.execute("SELECT is_baseline, passed FROM eval_runs").fetchone() == (False, False)
