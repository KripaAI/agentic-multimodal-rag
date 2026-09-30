"""The golden set (spec §9, LLD §5.10): a versioned JSONL file, validated on load."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

GoldenType = Literal["conceptual", "visual", "quantitative", "mixed", "unanswerable"]


class GoldenItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question_id: str
    question: str
    qtype: GoldenType
    reference_answer: str
    reference_ids: list[str] = []  # element or chunk ids that support the reference answer
    expected_figure_ids: list[str] = []
    expected_chart_values: list[float] = []
    expected_tool_calls: list[str] = []  # tool names the agent should use
    answerable: bool = True
    follows: str | None = None  # a follow-up: asked in the thread of this earlier question
    source_file: str | None = None  # the book the reference comes from (coverage checks)
    note: str | None = None  # for reviewers


def _check(item: GoldenItem, seen: set[str]) -> None:
    from mmrag.agent.tools import TOOLS

    q = item.question_id
    if q in seen:
        raise ValueError(f"{q}: duplicate question_id")
    if item.follows and item.follows not in seen:
        raise ValueError(f"{q}: follows {item.follows}, which is not an earlier question")
    if item.qtype == "visual" and not item.expected_figure_ids:
        raise ValueError(f"{q}: a visual question needs expected_figure_ids")
    unknown = [t for t in item.expected_tool_calls if t not in TOOLS]
    if unknown:
        raise ValueError(f"{q}: unknown tool {unknown}")
    if item.answerable and not item.reference_ids:
        raise ValueError(f"{q}: an answerable question needs reference_ids")
    if not item.answerable and (item.reference_ids or item.qtype != "unanswerable"):
        raise ValueError(f"{q}: an unanswerable question has qtype 'unanswerable' and no reference ids")


def load_golden_set(path: Path) -> tuple[list[GoldenItem], str]:
    """Items in file order and the file's version (first 12 hex chars of its SHA-256)."""
    raw = path.read_bytes()
    items, seen = [], set()
    for n, line in enumerate(raw.decode("utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            item = GoldenItem.model_validate_json(line)
        except Exception as e:  # noqa: BLE001 - point at the line
            raise ValueError(f"line {n}: {e}") from None
        _check(item, seen)
        seen.add(item.question_id)
        items.append(item)
    return items, hashlib.sha256(raw).hexdigest()[:12]
