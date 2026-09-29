"""The question file for `mmrag ask-batch` and the model comparison page. Test-first."""

from __future__ import annotations

import pytest

from mmrag.agent.answer import HydratedAnswer
from mmrag.agent.batch import load_questions, run_batch, write_comparison_page
from mmrag.agent.graph import QueryRun

pytestmark = pytest.mark.unit


def _run(thread, cost=0.01, latency=5000, result="ok"):
    ans = HydratedAnswer.model_validate({"blocks": [{"type": "text", "content": {"markdown": "x"}, "citations": []}],
                                         "sources": []})
    return QueryRun(answer=ans, thread_id=thread, trace_id="0" * 32, model="m", qtype="conceptual", rounds=1,
                    tool_calls=[], input_tokens=100, output_tokens=10, cost_usd=cost, latency_ms=latency,
                    validator_result=result)


def test_questions_file_with_a_follow_up(tmp_path):
    f = tmp_path / "q.yaml"
    f.write_text("questions:\n  - {id: q1, kind: conceptual, question: 'How?'}\n"
                 "  - {id: q2, kind: follow_up, question: 'And then?', follows: q1}\n", encoding="utf-8")
    qs = load_questions(f)
    assert [q.id for q in qs] == ["q1", "q2"] and qs[1].follows == "q1"


def test_follow_up_must_name_an_earlier_question(tmp_path):
    f = tmp_path / "q.yaml"
    f.write_text("questions:\n  - {id: q1, kind: follow_up, question: 'And?', follows: q9}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="q9"):
        load_questions(f)


def test_run_batch_continues_the_thread_per_model(tmp_path, parse_settings):
    f = tmp_path / "q.yaml"
    f.write_text("questions:\n  - {id: q1, kind: conceptual, question: 'How?'}\n"
                 "  - {id: q2, kind: follow_up, question: 'And then?', follows: q1}\n", encoding="utf-8")
    seen = []

    def ask(question, settings, thread_id=None, model=None):
        seen.append((model, question, thread_id))
        return _run(thread_id or f"{model}-t")

    results = run_batch(load_questions(f), ["a", "b"], parse_settings, ask=ask, page=lambda q, r, s: tmp_path / "p.html")
    assert seen == [("a", "How?", None), ("a", "And then?", "a-t"), ("b", "How?", None), ("b", "And then?", "b-t")]
    assert results["b"]["q2"].run.thread_id == "b-t"


def test_a_failing_question_is_recorded_not_fatal(tmp_path, parse_settings):
    f = tmp_path / "q.yaml"
    f.write_text("questions:\n  - {id: q1, kind: conceptual, question: 'How?'}\n", encoding="utf-8")

    def ask(question, settings, thread_id=None, model=None):
        raise RuntimeError("rate limited")

    results = run_batch(load_questions(f), ["a"], parse_settings, ask=ask, page=lambda q, r, s: tmp_path / "p.html")
    assert results["a"]["q1"].error == "RuntimeError: rate limited"


def test_comparison_page_shows_medians_per_model(tmp_path, parse_settings):
    f = tmp_path / "q.yaml"
    f.write_text("questions:\n  - {id: q1, kind: conceptual, question: 'How?'}\n"
                 "  - {id: q2, kind: visual, question: 'Show?'}\n  - {id: q3, kind: visual, question: 'Why?'}\n",
                 encoding="utf-8")
    runs = iter([_run("t", 0.01, 4000), _run("t", 0.03, 8000, "repaired"), _run("t", 0.02, 6000)])
    results = run_batch(load_questions(f), ["a"], parse_settings, ask=lambda *a, **k: next(runs),
                        page=lambda q, r, s: tmp_path / "p.html")
    html = write_comparison_page(load_questions(f), results, parse_settings).read_text(encoding="utf-8")
    assert "$0.0200" in html and "6.0 s" in html  # medians
    assert "2/3" in html  # validator ok on first try
    assert "p.html" in html


def test_the_cost_cap_stops_the_batch(tmp_path, parse_settings):
    f = tmp_path / "q.yaml"
    f.write_text("questions:\n  - {id: q1, kind: conceptual, question: 'How?'}\n"
                 "  - {id: q2, kind: visual, question: 'Show?'}\n", encoding="utf-8")
    results = run_batch(load_questions(f), ["a", "b"], parse_settings, ask=lambda *a, **k: _run("t", cost=2.0),
                        page=lambda q, r, s: tmp_path / "p.html", max_cost=3.0)
    assert results["a"]["q2"].run and results["b"]["q1"].error.startswith("skipped: cost cap")
