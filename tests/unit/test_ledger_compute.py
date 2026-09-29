"""Evidence ledger and safe arithmetic (LLD §5.2, §5.3). Written test-first."""

from __future__ import annotations

import pytest

from mmrag.agent.compute import ComputeError, evaluate
from mmrag.agent.ledger import Evidence, EvidenceLedger, numbers_in

pytestmark = pytest.mark.unit


def test_numbers_in_text_include_percent_fractions():
    assert numbers_in("800 KB vs 160 KB; 33.5 GB, 1,196 tokens, 41%") >= {800, 160, 33.5, 1196, 41, 0.41}


def test_ledger_keeps_first_record_and_round_trips():
    ledger = EvidenceLedger()
    ledger.add(Evidence(id="t1", kind="table", text="A", numbers=[800.0, 160.0]))
    ledger.add(Evidence(id="t1", kind="table", text="B", numbers=[1.0]))
    assert ledger.has("t1") and ledger.items["t1"].text == "A"
    again = EvidenceLedger.from_dict(ledger.to_dict())
    assert again.items["t1"].numbers == [800.0, 160.0]


def test_find_number_in_refs_or_anywhere():
    ledger = EvidenceLedger()
    ledger.add(Evidence(id="t1", kind="table", text="", numbers=[33.5, 6.7]))
    ledger.add(Evidence(id="f1", kind="figure", text="", estimated=sorted(numbers_in("41%"))))
    assert ledger.find_number(33.5, refs=["t1"]).id == "t1"
    assert ledger.find_number(33.5, refs=["f1"]) is None  # not where the model said
    assert ledger.find_number(0.41).id == "f1"  # printed 41% = 0.41
    assert ledger.find_number(34) is None
    assert ledger.find_number(3350) is None  # no silent rescaling of 33.5


def test_compute_arithmetic_with_refs():
    r = evaluate("(a - b) / a * 100", {"a": (33.5, "t1"), "b": (6.7, "t1")})
    assert r.value == pytest.approx(80.0)
    assert r.inputs == {"a": (33.5, "t1"), "b": (6.7, "t1")}
    assert evaluate("round(max(a, b) / min(a, b), 2)", {"a": (800, "t"), "b": (160, "t")}).value == 5.0


@pytest.mark.parametrize("expr", [
    "__import__('os').system('dir')",
    "a.__class__",
    "open('x')",
    "[x for x in (1, 2)]",
    "a ** 1000000",
    "c + 1",  # unknown name
    "lambda: 1",
])
def test_compute_rejects_anything_but_arithmetic(expr):
    with pytest.raises(ComputeError):
        evaluate(expr, {"a": (2.0, "t")})


def test_compute_division_by_zero_is_an_error():
    with pytest.raises(ComputeError, match="zero"):
        evaluate("a / (a - a)", {"a": (2.0, "t")})
