"""Tool dispatch and the ledger-only tools (LLD §5.3). Written test-first."""

from __future__ import annotations

import json
import time

import pytest

from mmrag.agent import tools
from mmrag.agent.ledger import Evidence, EvidenceLedger
from mmrag.agent.tools import TOOL_SPECS, ToolCall, ToolContext, dispatch

pytestmark = pytest.mark.unit


@pytest.fixture
def ctx(parse_settings, tmp_path):
    ledger = EvidenceLedger()
    ledger.add(Evidence(id="t19", kind="table", text="", numbers=[33.5, 6.7]))
    return ToolContext(settings=parse_settings, ledger=ledger, embed_query=lambda q: [0.0], chart_dir=tmp_path,
                       source_note=lambda ids: "Source: t.pdf, p. 19")


def _content(result):
    return json.loads(result.content)


def test_every_tool_is_described_to_the_model():
    names = {s["function"]["name"] for s in TOOL_SPECS}
    assert names == set(tools.TOOLS)
    for spec in TOOL_SPECS:
        assert spec["type"] == "function" and spec["function"]["description"]
        assert spec["function"]["parameters"]["type"] == "object"


def test_calls_run_in_parallel_and_keep_their_order(ctx, monkeypatch):
    def slow(ctx, query, k=8):
        time.sleep(0.4)
        return [{"id": query}]

    monkeypatch.setitem(tools.TOOLS, "search_text", slow)
    started = time.monotonic()
    results = dispatch([ToolCall("a", "search_text", {"query": "one"}), ToolCall("b", "search_text", {"query": "two"})],
                       ctx)
    assert time.monotonic() - started < 0.75
    assert [r.call_id for r in results] == ["a", "b"] and _content(results[1]) == [{"id": "two"}]


def test_problems_become_error_results_not_exceptions(ctx, monkeypatch):
    def boom(ctx, query, k=8):
        raise RuntimeError("database down")

    monkeypatch.setitem(tools.TOOLS, "search_figures", boom)
    results = dispatch([
        ToolCall("1", "search_figures", {"query": "x"}),
        ToolCall("2", "no_such_tool", {}),
        ToolCall("3", "search_text", {"k": 5}),  # missing required query
    ], ctx)
    assert all(r.is_error for r in results)
    assert "database down" in _content(results[0])["error"]
    assert "unknown tool" in _content(results[1])["error"]
    assert "query" in _content(results[2])["error"]


def test_a_slow_tool_times_out(ctx, monkeypatch):
    monkeypatch.setitem(tools.TOOLS, "search_tables", lambda ctx, query, k=8: time.sleep(2))
    ctx.settings.agent.tool_timeout_s = 0.2
    (result,) = dispatch([ToolCall("1", "search_tables", {"query": "x"})], ctx)
    assert result.is_error and "timed out" in _content(result)["error"]


def test_compute_checks_inputs_against_their_sources_and_records_the_result(ctx):
    (ok,) = dispatch([ToolCall("1", "compute", {"expression": "(a - b) / a * 100", "refs": {
        "a": {"value": 33.5, "source": "t19"}, "b": {"value": 6.7, "source": "t19"}}})], ctx)
    out = _content(ok)
    assert not ok.is_error and out["value"] == pytest.approx(80.0)
    assert ctx.ledger.find_number(80.0, refs=[out["id"]]) is not None  # charts may now use it

    (bad,) = dispatch([ToolCall("2", "compute", {"expression": "a * 2", "refs": {
        "a": {"value": 34, "source": "t19"}}})], ctx)
    assert bad.is_error and "34" in _content(bad)["error"]


def test_make_chart_tool_returns_an_id_or_the_reasons(ctx):
    args = {"chart_type": "bar", "title": "KV cache", "labels": ["40 heads", "8 heads"],
            "series": [{"name": "GB", "values": [33.5, 6.7], "unit": "GB"}],
            "value_refs": {"GB:40 heads": "t19", "GB:8 heads": "t19"}, "citations": ["t19"]}
    (ok,) = dispatch([ToolCall("1", "make_chart", args)], ctx)
    chart_id = _content(ok)["chart_id"]
    assert not ok.is_error and chart_id in ctx.charts
    bad_args = {**args, "series": [{"name": "GB", "values": [34, 6.7], "unit": "GB"}]}
    (bad,) = dispatch([ToolCall("2", "make_chart", bad_args)], ctx)
    assert bad.is_error and any("34" in e for e in _content(bad)["errors"])
