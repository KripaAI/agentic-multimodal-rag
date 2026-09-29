"""Chart engine (spec §7.4, LLD §5.5). Plain rule-based code, no model.

Four jobs: check every number against the evidence ledger (P4); fit the chart type to the
data (pie only for parts of a whole, else bar; line only for an ordered axis; one unit per
axis); mark charts with estimated values "approximate"; draw an interactive Plotly figure,
a PNG, the data table and a source footnote.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

from mmrag.agent.ledger import EvidenceLedger

ChartType = Literal["bar", "pie", "line"]
MAX_PIE_SLICES = 8


class SeriesSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    values: list[float]
    unit: str | None = None


class ChartRequest(BaseModel):
    """The `make_chart` tool arguments."""

    model_config = ConfigDict(extra="forbid")

    chart_type: ChartType
    title: str
    labels: list[str]
    series: list[SeriesSpec]
    value_refs: dict[str, str]  # "<series>:<label>" -> evidence id each value came from
    citations: list[str]


@dataclass
class ChartResult:
    ok: bool
    chart_id: str | None = None
    chart_type: ChartType | None = None  # may differ from the request (pie -> bar fallback)
    approximate: bool = False
    spec: dict | None = None  # Plotly figure JSON
    png_path: Path | None = None
    data_table: list[list[str]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)  # returned to the model on rejection


def validate_chart(request: ChartRequest, ledger: EvidenceLedger) -> ChartResult:
    """Apply the P4 rules. Every value must be found in the ledger at its `value_refs` id;
    an invalid pie falls back to bar; mixed units on one axis are rejected. Returns a result
    with `ok=False` and readable errors instead of raising."""
    raise NotImplementedError


def render_chart(result: ChartResult, request: ChartRequest, out_dir: Path) -> ChartResult:
    """Draw the validated chart: Plotly JSON + PNG (kaleido), the data table and a footnote."""
    raise NotImplementedError


def make_chart(request: ChartRequest, ledger: EvidenceLedger, out_dir: Path) -> ChartResult:
    """The tool: validate, then render if valid."""
    raise NotImplementedError
