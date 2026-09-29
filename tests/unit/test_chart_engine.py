"""Chart engine rules (spec §7.4, P4). Written test-first."""

from __future__ import annotations

import json

import pytest

from mmrag.agent.ledger import Evidence, EvidenceLedger
from mmrag.charts.engine import ChartRequest, make_chart, validate_chart

pytestmark = pytest.mark.unit


@pytest.fixture
def ledger():
    led = EvidenceLedger()
    led.add(Evidence(id="t19", kind="table", text="800 KB 160 KB 33.5 GB 6.7 GB", numbers=[800, 160, 33.5, 6.7]))
    led.add(Evidence(id="f4", kind="figure", text="bars", numbers=[41, 29], estimated=[0.4]))
    led.add(Evidence(id="f5", kind="figure", text="shares", numbers=[50, 30, 20]))
    return led


def _bar(values=(33.5, 6.7), refs=None, units=("GB", "GB")):
    return ChartRequest(
        chart_type="bar", title="KV cache at full context", labels=["40 KV heads", "8 KV heads"],
        series=[{"name": "cache", "values": list(values), "unit": units[0]}],
        value_refs=refs or {"cache:40 KV heads": "t19", "cache:8 KV heads": "t19"}, citations=["t19"])


def test_values_found_in_their_source_pass(ledger):
    r = validate_chart(_bar(), ledger)
    assert r.ok and not r.approximate and r.chart_type == "bar"
    assert r.data_table == [["", "cache (GB)"], ["40 KV heads", "33.5"], ["8 KV heads", "6.7"]]


def test_a_value_missing_from_its_source_is_rejected_with_a_hint(ledger):
    r = validate_chart(_bar(values=(34, 6.7)), ledger)
    assert not r.ok
    assert any("34" in e and "33.5" in e for e in r.errors)  # names the closest value in that source


def test_a_value_without_a_source_is_rejected(ledger):
    r = validate_chart(_bar(refs={"cache:40 KV heads": "t19"}), ledger)
    assert not r.ok and any("8 KV heads" in e for e in r.errors)


def test_a_value_from_the_wrong_source_is_rejected(ledger):
    r = validate_chart(_bar(refs={"cache:40 KV heads": "f4", "cache:8 KV heads": "t19"}), ledger)
    assert not r.ok


def test_mixed_units_on_one_axis_are_rejected(ledger):
    req = ChartRequest(chart_type="bar", title="t", labels=["per token", "full context"],
                       series=[{"name": "40 heads", "values": [800, 33.5], "unit": "KB"},
                               {"name": "8 heads", "values": [160, 6.7], "unit": "GB"}],
                       value_refs={"40 heads:per token": "t19", "40 heads:full context": "t19",
                                   "8 heads:per token": "t19", "8 heads:full context": "t19"}, citations=["t19"])
    r = validate_chart(req, ledger)
    assert not r.ok and any("unit" in e.lower() for e in r.errors)


def test_estimated_values_make_the_chart_approximate(ledger):
    req = ChartRequest(chart_type="bar", title="t", labels=["There", "Sure"],
                       series=[{"name": "p", "values": [41, 0.4], "unit": "%"}],
                       value_refs={"p:There": "f4", "p:Sure": "f4"}, citations=["f4"])
    assert validate_chart(req, ledger).approximate


def test_pie_of_percentages_summing_to_100_stays_a_pie(ledger):
    req = ChartRequest(chart_type="pie", title="t", labels=["a", "b", "c"],
                       series=[{"name": "share", "values": [50, 30, 20], "unit": "%"}],
                       value_refs={"share:a": "f5", "share:b": "f5", "share:c": "f5"}, citations=["f5"])
    assert validate_chart(req, ledger).chart_type == "pie"


def test_pie_that_is_not_parts_of_a_whole_becomes_a_bar(ledger):
    req = _bar().model_copy(update={"chart_type": "pie"})
    r = validate_chart(req, ledger)
    assert r.ok and r.chart_type == "bar"
    assert any("bar" in n for n in r.notes)


def test_series_and_labels_must_line_up(ledger):
    req = _bar().model_copy(update={"labels": ["only one"]})
    assert not validate_chart(req, ledger).ok


def test_make_chart_renders_plotly_and_png(ledger, tmp_path):
    r = make_chart(_bar(), ledger, tmp_path, footnote="Source: Transformers-in-Practice-Illustrated.pdf, p. 19")
    assert r.ok and r.chart_id and r.png_path.is_file() and r.png_path.stat().st_size > 1000
    spec = json.dumps(r.spec)
    assert "33.5" in spec and "p. 19" in spec
