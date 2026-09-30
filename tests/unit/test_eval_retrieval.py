"""Search-only evaluation on the golden set (cheap comparison of search settings). Test-first."""

from __future__ import annotations

import pytest

from mmrag.eval.dataset import GoldenItem
from mmrag.eval.retrieval_eval import apply_overrides, retrieval_scores
from mmrag.retrieval.hybrid import Hit

pytestmark = pytest.mark.unit


def _hit(cid, eid):
    return Hit(chunk_id=cid, collection="text", score=0.1, semantic_rank=1, keyword_rank=None, dense_text="",
               source_file="t.pdf", locations=[{"element_id": eid, "page": 1}])


ITEMS = [GoldenItem(question_id="c1", question="q1", qtype="conceptual", reference_answer="r", reference_ids=["d:text:2"]),
         GoldenItem(question_id="v1", question="q2", qtype="visual", reference_answer="r", reference_ids=["d:p3:image:1"],
                    expected_figure_ids=["d:p3:image:1"]),
         GoldenItem(question_id="u1", question="q3", qtype="unanswerable", reference_answer="r", answerable=False)]


def test_hit_rates_and_mrr_per_collection():
    results = {("q1", "text"): [_hit("d:text:1", "d:p1:text:1"), _hit("d:text:2", "d:p2:text:1")],
               ("q2", "figure"): [_hit(f"d:figure:{i}", f"d:p{i}:image:9") for i in range(10)]}
    out = retrieval_scores(ITEMS, lambda q, coll: results[(q, coll)], lambda ids: {})
    assert out["overall"] == {"hit@5": 0.5, "hit@10": 0.5, "mrr@10": 0.25, "questions": 2}  # c1 at rank 2, v1 missed
    assert out["by_collection"]["text"]["mrr@10"] == 0.5
    assert out["per_question"]["v1"]["rank"] is None


def test_overrides_change_only_the_named_setting(parse_settings):
    s = apply_overrides(parse_settings, ["search.keyword_mode=any", "search.rerank=true", "search.candidates=60"])
    assert (s.search.keyword_mode, s.search.rerank, s.search.candidates) == ("any", True, 60)
    assert s.search.rrf_k == parse_settings.search.rrf_k
    with pytest.raises(ValueError):
        apply_overrides(parse_settings, ["search.nope=1"])
