"""Evidence ledger (LLD §5.2): everything the tools returned during one question, by id.

The validator and the chart engine check answers against it: an answer may only cite ids
that are in the ledger, and a chart may only use numbers found in it (P4).
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Literal

EvidenceKind = Literal["chunk", "figure", "table", "page", "compute"]

_NUMBER = re.compile(r"(?<![\w.])-?\d[\d,]*(?:\.\d+)?(\s*%)?")


def numbers_in(text: str) -> set[float]:
    """Every number written in `text`; a percentage also counts as its fraction (41% = 0.41)."""
    found = set()
    for m in _NUMBER.finditer(text or ""):
        try:
            v = float(m.group(0).replace("%", "").replace(",", "").strip())
        except ValueError:
            continue
        found.add(v)
        if m.group(1):
            found.add(round(v / 100, 10))
    return found


def _same(a: float, b: float) -> bool:
    return abs(a - b) <= 1e-6 * max(1.0, abs(b))


@dataclass
class Evidence:
    id: str  # chunk_id, element_id, "page:<file>:<n>" or "compute:<n>"
    kind: EvidenceKind
    text: str  # what the model saw
    numbers: list[float] = field(default_factory=list)  # printed values (exact), usable in charts
    estimated: list[float] = field(default_factory=list)  # values read off a figure (P4: "approximate")


@dataclass
class EvidenceLedger:
    items: dict[str, Evidence] = field(default_factory=dict)

    def add(self, evidence: Evidence) -> None:
        """Record a tool result; a repeated id keeps the first record."""
        self.items.setdefault(evidence.id, evidence)

    def has(self, evidence_id: str) -> bool:
        return evidence_id in self.items

    def find_number(self, value: float, refs: list[str] | None = None) -> Evidence | None:
        """The evidence containing `value` (within rounding), searched in `refs` if given,
        otherwise everywhere. A printed percentage is stored with its fraction (`numbers_in`),
        so 0.41 matches "41%"; no other rescaling is allowed (33.5 never matches 3350)."""
        pool = [self.items[r] for r in refs if r in self.items] if refs is not None else self.items.values()
        for ev in pool:
            if any(_same(value, n) for n in ev.numbers + ev.estimated):
                return ev
        return None

    def is_estimated(self, value: float, evidence: Evidence) -> bool:
        """True when `value` is in the evidence only as a value read off a figure."""
        return not any(_same(value, n) for n in evidence.numbers) and any(_same(value, n) for n in evidence.estimated)

    def to_dict(self) -> dict:
        return {k: asdict(v) for k, v in self.items.items()}

    @classmethod
    def from_dict(cls, data: dict | None) -> "EvidenceLedger":
        return cls(items={k: Evidence(**v) for k, v in (data or {}).items()})
