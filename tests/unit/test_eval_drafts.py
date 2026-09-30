"""Drafting candidate golden questions from the corpus (owner reviews them). Test-first."""

from __future__ import annotations

import pytest

from mmrag.eval.dataset import GoldenItem
from mmrag.eval.drafts import DraftOut, Evidence, draft_one, pick_evidence

pytestmark = pytest.mark.unit


class Judge:
    def __init__(self, out):
        self.out, self.prompts = out, []

    def structured(self, messages, schema):
        self.prompts.append(messages[-1]["content"])
        return self.out


TABLE = Evidence(chunk_id="d:table:1", collection="table", source_file="t.pdf", element_ids=["d:p19:table:1"],
                 text="KV cache | 40 heads | 8 heads\nPer token | 800 KB | 160 KB\nFull context | 33.5 GB | 6.7 GB")


def test_a_quantitative_draft_keeps_only_values_printed_in_the_table():
    judge = Judge(DraftOut(question="Chart KV memory per token for 40 vs 8 heads", reference_answer="800 KB vs 160 KB.",
                           expected_chart_values=[800, 160]))
    item = draft_one("x1", "quantitative", TABLE, judge)
    assert isinstance(item, GoldenItem) and item.expected_chart_values == [800, 160]
    assert item.reference_ids == ["d:p19:table:1"] and item.expected_tool_calls == ["make_chart"]
    assert "800 KB" in judge.prompts[0]  # the judge only sees this evidence


def test_a_draft_with_invented_chart_values_is_dropped():
    judge = Judge(DraftOut(question="q", reference_answer="a", expected_chart_values=[800, 999]))
    assert draft_one("x1", "quantitative", TABLE, judge) is None


def test_visual_drafts_expect_the_figure():
    fig = Evidence(chunk_id="d:figure:1", collection="figure", source_file="t.pdf", element_ids=["d:p3:image:1"],
                   text="The autoregressive loop ...")
    item = draft_one("v1", "visual", fig, Judge(DraftOut(question="Show the loop", reference_answer="It repeats.",
                                                          expected_chart_values=[])))
    assert item.expected_figure_ids == ["d:p3:image:1"] and item.expected_tool_calls == ["search_figures"]


def test_evidence_is_spread_across_books():
    pool = [Evidence(chunk_id=f"{b}:text:{i}", collection="text", source_file=f"{b}.pdf", element_ids=[f"{b}:p1:text:{i}"],
                     text="w " * 100) for b in "abc" for i in range(10)]
    picked = pick_evidence(pool, 6, seed=1)
    assert len(picked) == 6 and {e.source_file for e in picked} == {"a.pdf", "b.pdf", "c.pdf"}
    assert len({e.chunk_id for e in picked}) == 6


def test_drafting_fills_each_quota_from_the_whole_pool_without_reusing_evidence(monkeypatch):
    from mmrag.eval import drafts as dr

    figs = [Evidence(chunk_id=f"b{i % 2}:figure:{i}", collection="figure", source_file=f"b{i % 2}.pdf",
                     element_ids=[f"b{i % 2}:p{i}:image:1"], text="fig " * 40) for i in range(12)]
    tabs = [Evidence(chunk_id=f"t:table:{i}", collection="table", source_file="t.pdf", element_ids=[f"t:p{i}:table:1"],
                     text="a | 1 | 2 | 3 | 4") for i in range(30)]
    monkeypatch.setattr(dr, "load_pool", lambda s: {"text": [], "figure": figs, "table": tabs})
    monkeypatch.setattr(dr, "unanswerable_items", lambda s: [])
    monkeypatch.setattr(dr, "QUOTA", {"visual": 4, "quantitative": 3, "mixed": 4})

    calls = {"n": 0}

    class J:
        def structured(self, messages, schema):
            calls["n"] += 1  # only every 5th table draft uses printed values
            ok = calls["n"] % 5 == 0
            return DraftOut(question="q", reference_answer="a", expected_chart_values=[1, 2] if ok else [9])

    out = dr.draft_questions(None, J(), seed=1, progress=lambda s: None)
    by = {}
    for item, ev in out:
        by.setdefault(item.qtype, []).append(ev.chunk_id)
    assert len(by["quantitative"]) == 3  # kept trying past the first few tables
    assert len(by["visual"]) == 4 and len(by["mixed"]) == 4
    assert not set(by["visual"]) & set(by["mixed"])  # no figure used twice


def test_drafts_record_their_book_and_promote_reports_coverage(tmp_path):
    from mmrag.eval.drafts import coverage, promote

    fig = Evidence(chunk_id="d:figure:1", collection="figure", source_file="a.pdf", element_ids=["d:p3:image:1"],
                   text="loop")
    item = draft_one("v1", "visual", fig, Judge(DraftOut(question="Show the loop", reference_answer="r",
                                                          expected_chart_values=[])))
    assert item.source_file == "a.pdf"
    drafts, golden = tmp_path / "drafts.jsonl", tmp_path / "golden.jsonl"
    drafts.write_text(item.model_dump_json() + "\n", encoding="utf-8")
    promote(drafts, golden, ["v1"])
    assert coverage(golden) == {"a.pdf": {"visual": 1}}
