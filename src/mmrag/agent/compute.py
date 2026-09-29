"""`compute`: safe arithmetic over numbers from the evidence (spec §7.1, LLD §5.3).

The expression is parsed into a Python syntax tree and evaluated by walking it. Only numbers,
names from `refs`, + - * /, parentheses, and round / sum / min / max are allowed; anything
else (attribute access, calls to other names, comprehensions, imports) is rejected. No `eval`.
"""

from __future__ import annotations

from dataclasses import dataclass


class ComputeError(ValueError):
    """Disallowed syntax, an unknown name, or a ref that is not in the ledger."""


@dataclass
class ComputeResult:
    value: float
    expression: str
    inputs: dict[str, tuple[float, str]]  # name -> (value, evidence id it came from)


def evaluate(expression: str, refs: dict[str, tuple[float, str]]) -> ComputeResult:
    """Evaluate `expression`, where each name in `refs` maps to (value, evidence id)."""
    raise NotImplementedError
