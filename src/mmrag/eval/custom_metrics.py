"""Deterministic metrics (spec §9, LLD §5.10 step 5). Each returns a 0-1 score, or None when the
metric does not apply to the question."""

from __future__ import annotations

import re

_NUMBER = re.compile(r"(?:-?\d[\d,]*\.?\d*|-?\.\d+)(?:[eE][+-]?\d+)?")  # also 1.67772e+07
_REFUSAL = re.compile(r"not (?:be )?(?:found|mention|stat|provid|includ|specif|say|cover|contain|give)|"
                      r"no information|could ?n[o']t find|does not (?:say|state|mention|provide)|"
                      r"do not (?:say|state|mention|provide|contain)", re.IGNORECASE)


def _num(cell: str) -> float | None:
    m = _NUMBER.search(cell or "")
    if not m:
        return None
    v = float(m.group().replace(",", ""))
    return v / 100 if "%" in cell[m.end():m.end() + 2] else v


def chart_values(data_table: list[list[str]]) -> list[float]:
    """All values of a chart's data table (header row and label column skipped), row by row."""
    return [v for row in data_table[1:] for v in (_num(c) for c in row[1:]) if v is not None]


def _same(a: float, b: float) -> bool:
    return abs(a - b) <= 1e-6 * max(1.0, abs(a), abs(b))


def chart_numeric(charts: list[list[float]], expected: list[float], percent_expected: bool = False
                  ) -> tuple[float | None, dict]:
    """Each chart passes only if every value equals an expected value (all-or-nothing); the score
    is the share of charts that pass. A value v also matches an expected e when e == v * 100
    (percent vs fraction, `percent_expected`). None when no chart was expected or drawn."""
    if not expected:
        return None, {}  # nothing to check against (the chart engine already verified every value)
    if not charts:
        return 0.0, {"missing": "a chart was expected"}
    targets = expected + ([e / 100 for e in expected] if percent_expected else [])
    wrong = [[v for v in chart if not any(_same(v, e) for e in targets)] for chart in charts]
    wrong = [w for w in wrong if w]
    return (len(charts) - len(wrong)) / len(charts), {"wrong": wrong} if wrong else {}


def figure_hit(shown: list[str], expected: list[str]) -> float | None:
    if not expected:
        return None
    return sum(e in shown for e in expected) / len(expected)


def citation_accuracy(cited: list[set[str]], reference_elements: set[str]) -> float | None:
    """`cited`: one set of element ids per citation. The share of citations touching the reference."""
    if not cited:
        return None
    return sum(bool(c & reference_elements) for c in cited) / len(cited)


def refusal(answerable: bool, not_found: bool, text: str) -> float | None:
    """For unanswerable questions: 1 when the answer says the documents do not contain it."""
    if answerable:
        return None
    return 1.0 if not_found or _REFUSAL.search(text or "") else 0.0


def tool_call_accuracy(actual: list[str], expected: list[str]) -> float | None:
    """The share of expected tools the agent called at least once (order-free)."""
    if not expected:
        return None
    return sum(t in actual for t in expected) / len(expected)
