"""Answer validator: the constitution enforced by code (spec §7.3, LLD §5.6). Written test-first."""

from __future__ import annotations

import pytest

from mmrag.agent.answer import Answer
from mmrag.agent.ledger import Evidence, EvidenceLedger
from mmrag.agent.validator import drop_failing_blocks, validate
from mmrag.charts.engine import ChartResult

pytestmark = pytest.mark.unit

LOC = {"element_id": "d:p3:text:1", "source_file": "t.pdf", "page": 3, "bbox": (1.0, 2.0, 3.0, 4.0)}


class FakeStore:
    elements = {
        "d:p3:vector_figure:2": {"type": "vector_figure", "asset_path": "assets/d/p3.png", "short_caption": "The loop",
                                 "page": 3, "source_file": "t.pdf"},
        "d:p19:table:1": {"type": "table", "columns": ["Heads", "Memory"], "rows": [["40", "33.5 GB"]],
                          "title": "KV", "page": 19, "source_file": "t.pdf"},
    }

    def locations(self, ids):
        out = {}
        for i in ids:
            if i == "d:text:1":
                out[i] = [LOC, {**LOC, "element_id": "d:p3:text:2", "bbox": (1.0, 5.0, 3.0, 6.0)}]
            elif i in self.elements or i == "d:p3:text:1":
                e = self.elements.get(i, {"page": 3, "source_file": "t.pdf"})
                out[i] = [{"element_id": i, "source_file": e["source_file"], "page": e["page"], "bbox": (0, 0, 1, 1)}]
        return out

    def element_info(self, ids):
        return {i: self.elements[i] for i in ids if i in self.elements}


@pytest.fixture
def ledger():
    led = EvidenceLedger()
    for eid, kind in [("d:text:1", "chunk"), ("d:p3:vector_figure:2", "figure"), ("d:p19:table:1", "table"),
                      ("d:p9:text:4", "chunk")]:
        led.add(Evidence(id=eid, kind=kind, text=""))
    return led


@pytest.fixture
def charts():
    return {"chart-abc": ChartResult(ok=True, chart_id="chart-abc", chart_type="bar", approximate=True,
                                     spec={"data": []}, data_table=[["", "GB"], ["40", "33.5"]],
                                     citations=["d:p19:table:1"], title="KV")}


def _answer(*blocks, **kw):
    return Answer.model_validate({"blocks": list(blocks), **kw})


def _text(*ids, md="Four steps."):
    return {"type": "text", "markdown": md, "citations": [{"id": i} for i in ids]}


def _run(answer, ledger, charts=None):
    return validate(answer, ledger, charts or {}, store=FakeStore())


def test_a_good_answer_is_hydrated_from_the_database(ledger, charts):
    r = _run(_answer(_text("d:text:1"),
                     {"type": "image", "element_id": "d:p3:vector_figure:2", "short_caption": "Loop",
                      "citation": {"id": "d:p3:vector_figure:2"}},
                     {"type": "chart", "chart_id": "chart-abc"},
                     {"type": "table", "element_id": "d:p19:table:1", "citation": {"id": "d:p19:table:1"}}),
             ledger, charts)
    assert r.ok, r.failures
    text, image, chart, table = r.answer.blocks
    assert [loc.page for loc in text.citations[0].locations] == [3, 3]  # a chunk expands to its elements
    assert text.citations[0].locations[0].bbox == (1.0, 2.0, 3.0, 4.0)
    assert image.content["asset_path"] == "assets/d/p3.png"
    assert chart.approximate and chart.content["data_table"][1] == ["40", "33.5"]
    assert table.content["rows"] == [["40", "33.5 GB"]]  # rows from the database, not the model
    assert {(s.page, s.element_id) for s in r.answer.sources} >= {(3, "d:p3:vector_figure:2"), (19, "d:p19:table:1")}


def test_uncited_text_fails(ledger):
    r = _run(_answer(_text()), ledger)
    assert not r.ok and r.failed_blocks == {0} and "citation" in r.failures[0]


def test_an_id_no_tool_returned_fails(ledger):
    r = _run(_answer(_text("d:p77:text:1")), ledger)
    assert not r.ok and "d:p77:text:1" in r.failures[0]


def test_an_id_unknown_to_the_database_fails(ledger):
    r = _run(_answer(_text("d:p9:text:4")), ledger)  # in the ledger but not in the corpus
    assert not r.ok and "unknown" in r.failures[0]


def test_an_image_must_be_an_original_figure(ledger):
    r = _run(_answer({"type": "image", "element_id": "d:p19:table:1", "short_caption": "x",
                      "citation": {"id": "d:p19:table:1"}}), ledger)
    assert not r.ok and "figure" in r.failures[0]


def test_a_chart_must_come_from_the_chart_engine(ledger):
    r = _run(_answer({"type": "chart", "chart_id": "chart-invented"}), ledger)
    assert not r.ok and "chart-invented" in r.failures[0]


def test_memory_is_never_a_citation(ledger):
    ledger.add(Evidence(id="memory:7", kind="chunk", text="prefers charts"))
    r = _run(_answer(_text("memory:7")), ledger)
    assert not r.ok and "memory" in r.failures[0].lower()


def test_computed_numbers_cite_their_sources_not_the_computation(ledger):
    ledger.add(Evidence(id="compute:1", kind="compute", text="80"))
    r = _run(_answer(_text("compute:1")), ledger)
    assert not r.ok and "source" in r.failures[0]


def test_not_found_answers_need_no_citations_but_make_no_claims(ledger):
    ok = _run(_answer(_text(md="The documents do not cover Kubernetes."), not_found=True, missing="Kubernetes"),
              ledger)
    assert ok.ok
    bad = _run(_answer(_text(md="Not found."), {"type": "chart", "chart_id": "chart-abc"}, not_found=True), ledger)
    assert not bad.ok


def test_dropping_failing_blocks_keeps_the_rest_with_a_notice(ledger):
    answer = _answer(_text("d:text:1"), _text("d:p77:text:1"))
    r = _run(answer, ledger)
    kept, notices = drop_failing_blocks(answer, r.failed_blocks)
    assert len(kept.blocks) == 1 and notices and _run(kept, ledger).ok
