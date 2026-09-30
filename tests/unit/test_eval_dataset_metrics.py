"""Golden set and the deterministic evaluation metrics (spec §9, LLD §5.10). Test-first."""

from __future__ import annotations

import json

import pytest

from mmrag.eval.custom_metrics import (chart_numeric, chart_values, citation_accuracy, figure_hit, refusal,
                                       tool_call_accuracy)
from mmrag.eval.dataset import GoldenItem, load_golden_set

pytestmark = pytest.mark.unit


def _write(tmp_path, *items):
    f = tmp_path / "golden.jsonl"
    f.write_text("\n".join(json.dumps(i) for i in items), encoding="utf-8")
    return f


BASE = {"question_id": "g1", "question": "What is RLHF?", "qtype": "conceptual", "reference_answer": "RLHF is ...",
        "reference_ids": ["d:text:1"], "expected_tool_calls": ["search_text"], "answerable": True}


def test_golden_set_loads_and_versions(tmp_path):
    items, version = load_golden_set(_write(tmp_path, BASE))
    assert items[0].question_id == "g1" and len(version) == 12


@pytest.mark.parametrize("bad, match", [
    ({**BASE, "qtype": "visual"}, "expected_figure_ids"),
    ({**BASE, "expected_tool_calls": ["google"]}, "unknown tool"),
    ({**BASE, "answerable": False}, "reference"),
    ({**BASE, "follows": "g0"}, "earlier"),
])
def test_golden_set_rejects_incomplete_items(tmp_path, bad, match):
    with pytest.raises(ValueError, match=match):
        load_golden_set(_write(tmp_path, bad))


def test_duplicate_ids_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="duplicate"):
        load_golden_set(_write(tmp_path, BASE, BASE))


def test_chart_values_are_read_from_the_data_table():
    table = [["", "Memory (GB)", "Share"], ["FP32", "280", "45%"], ["INT8", "70", "0.1"]]
    assert chart_values(table) == [280.0, 0.45, 70.0, 0.1]


def test_chart_numeric_is_all_or_nothing_per_chart():
    assert chart_numeric([[280, 140]], [280, 140, 70])[0] == 1.0
    score, details = chart_numeric([[280, 141]], [280, 140])
    assert score == 0.0 and details["wrong"] == [[141.0]]
    assert chart_numeric([[0.45]], [45], percent_expected=True)[0] == 1.0  # 45% == 0.45
    assert chart_numeric([], [1, 2])[0] == 0.0  # a chart was expected
    assert chart_numeric([], [])[0] is None  # not applicable
    assert chart_numeric([[41, 29]], [])[0] is None  # a chart nobody asked for is not graded here


def test_figure_hit_citation_accuracy_refusal_and_tools():
    assert figure_hit(["d:p3:image:1"], ["d:p3:image:1", "d:p5:image:1"]) == 0.5
    assert figure_hit([], []) is None
    assert citation_accuracy([{"d:p1:text:1"}, {"d:p9:text:2"}], {"d:p1:text:1"}) == 0.5
    assert citation_accuracy([], {"x"}) is None
    assert refusal(answerable=False, not_found=True, text="") == 1.0
    assert refusal(answerable=False, not_found=False, text="The documents do not say.") == 1.0
    assert refusal(answerable=False, not_found=False, text="It cost $5M.") == 0.0
    assert refusal(answerable=True, not_found=True, text="") is None  # scored as a false refusal elsewhere
    assert tool_call_accuracy(["search_text", "make_chart", "search_text"], ["search_text", "get_table"]) == 0.5
    assert tool_call_accuracy(["search_text"], []) is None
