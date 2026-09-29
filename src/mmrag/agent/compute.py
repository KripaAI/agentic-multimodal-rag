"""`compute`: safe arithmetic over numbers from the evidence (spec §7.1, LLD §5.3).

The expression is parsed into a Python syntax tree and evaluated by walking it. Only numbers,
names from `refs`, + - * / and unary minus, parentheses, and round / sum / min / max / abs are
allowed; anything else (attribute access, other calls, powers, comprehensions, lambdas) is
rejected. No `eval`.
"""

from __future__ import annotations

import ast
import operator
from dataclasses import dataclass

_BINARY = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv}
_UNARY = {ast.USub: operator.neg, ast.UAdd: operator.pos}
_FUNCS = {"round": round, "sum": sum, "min": min, "max": max, "abs": abs}


class ComputeError(ValueError):
    """Disallowed syntax, an unknown name, or a ref that is not in the ledger."""


@dataclass
class ComputeResult:
    value: float
    expression: str
    inputs: dict[str, tuple[float, str]]  # name -> (value, evidence id it came from)


def evaluate(expression: str, refs: dict[str, tuple[float, str]]) -> ComputeResult:
    """Evaluate `expression`, where each name in `refs` maps to (value, evidence id)."""
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as e:
        raise ComputeError(f"not an arithmetic expression: {e.msg}") from e
    used: dict[str, tuple[float, str]] = {}

    def walk(node):
        if isinstance(node, ast.Expression):
            return walk(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return node.value
        if isinstance(node, ast.Name):
            if node.id not in refs:
                raise ComputeError(f"unknown name {node.id!r}: every value must be a ref to evidence")
            used[node.id] = refs[node.id]
            return refs[node.id][0]
        if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
            left, right = walk(node.left), walk(node.right)
            if isinstance(node.op, ast.Div) and right == 0:
                raise ComputeError("division by zero")
            return _BINARY[type(node.op)](left, right)
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
            return _UNARY[type(node.op)](walk(node.operand))
        if isinstance(node, (ast.List, ast.Tuple)):
            return [walk(e) for e in node.elts]
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FUNCS \
                and not node.keywords:
            return _FUNCS[node.func.id](*[walk(a) for a in node.args])
        raise ComputeError(f"not allowed: {type(node).__name__}")

    value = walk(tree)
    if not isinstance(value, (int, float)):
        raise ComputeError("the expression must produce one number")
    return ComputeResult(value=float(value), expression=expression, inputs=used)
