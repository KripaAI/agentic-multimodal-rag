"""Evidence ledger (LLD §5.2): everything the tools returned during one question, by id.

The validator and the chart engine check answers against it: an answer may only cite ids
that are in the ledger, and a chart may only use numbers found in it (P4).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

EvidenceKind = Literal["chunk", "figure", "table", "page", "compute"]


@dataclass
class Evidence:
    id: str  # chunk_id, element_id, "page:<file>:<n>" or "compute:<n>"
    kind: EvidenceKind
    text: str  # what the model saw
    numbers: list[float] = field(default_factory=list)  # values usable in charts
    exact: bool = True  # False for values read off a figure (extracted_data flagged estimated)


@dataclass
class EvidenceLedger:
    items: dict[str, Evidence] = field(default_factory=dict)

    def add(self, evidence: Evidence) -> None:
        """Record a tool result; a repeated id keeps the first record."""
        raise NotImplementedError

    def has(self, evidence_id: str) -> bool:
        raise NotImplementedError

    def find_number(self, value: float, refs: list[str] | None = None) -> Evidence | None:
        """The evidence containing `value` (within rounding), searched in `refs` if given,
        otherwise everywhere. Percentages match their fractions (41% = 0.41)."""
        raise NotImplementedError
