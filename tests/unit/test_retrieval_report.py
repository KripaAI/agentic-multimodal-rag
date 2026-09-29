"""Retrieval test report: queries file and the top-5 pass rule (plan Phase 3 task 9). Test-first."""

from __future__ import annotations

import pytest

from mmrag.retrieval.hybrid import Hit
from mmrag.retrieval.report import found_rank, load_queries

pytestmark = pytest.mark.unit


def _hit(*eids):
    return Hit(chunk_id="c", collection="figure", score=0.1, semantic_rank=1, keyword_rank=None, dense_text="",
               source_file="x.pdf", locations=[{"element_id": e, "page": 1, "bbox": [0, 0, 1, 1]} for e in eids])


def test_rank_of_the_first_hit_covering_an_expected_element():
    hits = [_hit("d:p1:text:1"), _hit("d:p2:text:1", "d:p3:vector_figure:2"), _hit("d:p4:table:1")]
    assert found_rank(hits, {"d:p3:vector_figure:2", "d:p9:text:1"}) == 2
    assert found_rank(hits, {"d:p8:text:1"}) is None


def test_queries_file(tmp_path):
    f = tmp_path / "q.yaml"
    f.write_text("- id: q1\n  query: steps of generation\n  source: x.pdf\n  collection: figure\n"
                 "  expect: [p3:vector_figure:2]\n  kind: visual\n", encoding="utf-8")
    (q,) = load_queries(f)
    assert q.id == "q1" and q.collection == "figure" and q.expect == ["p3:vector_figure:2"]


def test_queries_file_rejects_unknown_collection(tmp_path):
    f = tmp_path / "q.yaml"
    f.write_text("- id: q1\n  query: x\n  source: x.pdf\n  collection: images\n  expect: [p1:image:1]\n  kind: visual\n",
                 encoding="utf-8")
    with pytest.raises(Exception):
        load_queries(f)


def test_a_query_can_also_accept_passages_in_other_files(tmp_path):
    # Phase 5: "vLLM" is discussed in three books; any of their vLLM passages counts.
    from mmrag.retrieval.report import expected_ids

    f = tmp_path / "q.yaml"
    f.write_text("- id: e3\n  query: vLLM\n  source: a.pdf\n  collection: text\n  expect: [p10:text:6]\n"
                 "  kind: exact_term\n  also:\n    b.pdf: [p99:text:15]\n", encoding="utf-8")
    (q,) = load_queries(f)
    assert expected_ids(q, {"a.pdf": "A", "b.pdf": "B"}) == {"A:p10:text:6", "B:p99:text:15"}
