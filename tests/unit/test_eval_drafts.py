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
