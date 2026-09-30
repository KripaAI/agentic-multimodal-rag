"""Scoring one question, summaries and the regression gate (LLD §5.10). Test-first."""

from __future__ import annotations

import pytest

from mmrag.agent.answer import HydratedAnswer
from mmrag.agent.graph import QueryRun
from mmrag.eval.dataset import GoldenItem
from mmrag.eval.runner import gate, score_item, summarize

pytestmark = pytest.mark.unit


class FakeMetrics:
    def __init__(self):
        self.called = []

    def _m(self, name, *a, **k):
        self.called.append((name, k.get("images")))
        return 0.8, {"m": name}

    def faithfulness(self, *a, **k): return self._m("faithfulness", *a, **k)
    def response_relevancy(self, *a, **k): return self._m("response_relevancy", *a, **k)
    def context_precision(self, *a, **k): return self._m("context_precision", *a, **k)
    def context_recall(self, *a, **k): return self._m("context_recall", *a, **k)
    def factual_correctness(self, *a, **k): return self._m("factual_correctness", *a, **k)


class FakeStore:
    def locations(self, ids):
        return {i: [{"element_id": i.replace(":text:", ":p1:text:")}] for i in ids if ":text:" in i and ":p" not in i}


def _run(blocks, evidence=(), not_found=False, tools=("search_text",)):
    ans = HydratedAnswer.model_validate({"blocks": blocks, "sources": []})
    return QueryRun(answer=ans, thread_id="t", trace_id="f" * 32, model="m", qtype="conceptual", rounds=1,
                    tool_calls=[{"name": t} for t in tools], input_tokens=100, output_tokens=10, cost_usd=0.01,
                    latency_ms=9000, validator_result="ok", evidence=list(evidence), not_found=not_found)


def _cit(eid):
    return {"id": eid, "locations": [{"element_id": eid if ":p" in eid else eid.replace(":text:", ":p1:text:"),
                                      "source_file": "t.pdf", "page": 1, "bbox": [0, 0, 1, 1]}]}


ITEM = GoldenItem(question_id="g1", question="How much does GQA save?", qtype="quantitative",
                  reference_answer="33.5 GB to 6.7 GB", reference_ids=["d:text:1"],
                  expected_chart_values=[33.5, 6.7], expected_tool_calls=["search_text", "make_chart"])


def test_score_item_runs_every_metric(tmp_path):
    blocks = [{"type": "text", "content": {"markdown": "It drops to 6.7 GB."}, "citations": [_cit("d:text:1")]},
              {"type": "chart", "content": {"data_table": [["", "GB"], ["MHA", "33.5"], ["GQA", "6.7"]]},
               "citations": [_cit("d:text:1")]}]
    m = FakeMetrics()
    scores = score_item(ITEM, _run(blocks, [{"id": "d:text:1", "text": "33.5 GB to 6.7 GB"}]), m, FakeStore(), tmp_path)
    assert scores["chart_numeric"][0] == 1.0 and scores["citation_accuracy"][0] == 1.0
    assert scores["tool_call_accuracy"][0] == 0.5  # make_chart was expected, not called
    assert scores["faithfulness"][0] == 0.8 and "multimodal_faithfulness" not in scores  # no figure shown
    assert scores["latency_s"][0] == 9.0 and scores["cost_usd"][0] == 0.01
    assert "refusal" not in scores


def test_figures_shown_get_multimodal_faithfulness_with_their_images(tmp_path):
    item = ITEM.model_copy(update={"qtype": "visual", "expected_figure_ids": ["d:p3:image:1"],
                                   "expected_chart_values": []})
    blocks = [{"type": "text", "content": {"markdown": "See the figure."}, "citations": [_cit("d:text:1")]},
              {"type": "image", "content": {"element_id": "d:p3:image:1", "asset_path": "assets/d/p3.png"},
               "citations": [_cit("d:p3:image:1")]}]
    m = FakeMetrics()
    scores = score_item(item, _run(blocks), m, FakeStore(), tmp_path)
    assert scores["figure_hit"][0] == 1.0
    assert ("faithfulness", [str(tmp_path / "assets/d/p3.png")]) in m.called  # the judge sees the figure
    assert "multimodal_faithfulness" in scores


def test_unanswerable_questions_only_score_refusal_latency_and_cost(tmp_path):
    item = GoldenItem(question_id="u1", question="Training cost?", qtype="unanswerable", reference_answer="Not in the documents.",
                      answerable=False)
    m = FakeMetrics()
    scores = score_item(item, _run([{"type": "text", "content": {"markdown": "Not found."}, "citations": []}],
                                   not_found=True), m, FakeStore(), tmp_path)
    assert scores["refusal"][0] == 1.0 and m.called == [] and set(scores) == {"refusal", "latency_s", "cost_usd"}


def test_summary_and_gate():
    items = [ITEM, ITEM.model_copy(update={"question_id": "g2", "qtype": "conceptual"})]
    results = {"g1": {"faithfulness": (1.0, {}), "chart_numeric": (1.0, {}), "latency_s": (10.0, {}), "cost_usd": (0.02, {})},
               "g2": {"faithfulness": (0.5, {}), "chart_numeric": (None, {}), "latency_s": (20.0, {}), "cost_usd": (0.03, {})}}
    s = summarize(results, items)
    assert s["metrics"]["faithfulness"] == 0.75 and s["metrics"]["chart_numeric"] == 1.0
    assert s["by_type"]["conceptual"]["faithfulness"] == 0.5
    assert s["median_latency_s"] == 15.0 and s["total_cost_usd"] == pytest.approx(0.05)

    ok, failures = gate(s, None, 0.03)
    assert ok and failures == []
    worse = {**s, "metrics": {**s["metrics"], "faithfulness": 0.70}}
    ok, failures = gate(worse, s, 0.03)
    assert not ok and "faithfulness" in failures[0]
    chart_miss = {**s, "metrics": {**s["metrics"], "chart_numeric": 0.9}}
    assert not gate(chart_miss, None, 0.03)[0]  # 100% metrics must stay at 100%
