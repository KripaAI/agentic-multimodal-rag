"""Answer validator (spec §7.3, LLD §5.6): the constitution, enforced by code.

Rules: every block cites at least one id that exists in the corpus and in the ledger; images
are original figures (P3); every chart comes from the chart engine and every value in it is in
the evidence (P4); no citation points to a memory (P13); a "not found" answer makes no claims.
Citations are hydrated with source_file, page and bbox from PostgreSQL; anything location-like
the model wrote is ignored.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from mmrag.agent.answer import Answer, HydratedAnswer
from mmrag.agent.ledger import EvidenceLedger
from mmrag.config import Settings


@dataclass
class ValidationResult:
    ok: bool
    answer: HydratedAnswer | None = None
    failures: list[str] = field(default_factory=list)  # one line per broken rule, for the repair turn


def validate(answer: Answer, ledger: EvidenceLedger, charts: dict, settings: Settings) -> ValidationResult:
    """Check every rule and hydrate citations. `charts` holds the engine results by chart_id."""
    raise NotImplementedError


def hydrate(ids: list[str], settings: Settings) -> dict[str, list[dict]]:
    """id -> locations [{element_id, source_file, page, bbox}] from PostgreSQL, in reading order;
    a chunk_id expands to its elements. Unknown ids are absent from the result."""
    raise NotImplementedError


def drop_failing_blocks(answer: Answer, failures: list[str]) -> tuple[Answer, list[str]]:
    """After a failed repair: remove the blocks that still break a rule; return notices saying so."""
    raise NotImplementedError
