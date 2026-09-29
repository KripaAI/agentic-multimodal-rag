"""Chart engine (spec §7.4, LLD §5.5). Plain rule-based code, no model.

Four jobs: check every number against the evidence ledger (P4); fit the chart type to the
data (pie only for percentages that make a whole, else bar; one unit per axis); mark charts
with estimated values "approximate"; draw an interactive Plotly figure, a PNG, the data table
and a source footnote.
"""

from __future__ import annotations

import hashlib
import json
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
    notes: list[str] = field(default_factory=list)  # changes the engine made (e.g. pie -> bar)
    citations: list[str] = field(default_factory=list)
    title: str = ""


def _fmt(v: float) -> str:
    return f"{v:g}"


def validate_chart(request: ChartRequest, ledger: EvidenceLedger) -> ChartResult:
    """Apply the P4 rules. Every value must be found in the ledger at its `value_refs` id;
    an invalid pie falls back to bar; mixed units on one axis are rejected. Returns a result
    with `ok=False` and readable errors instead of raising."""
    errors, notes, approximate = [], [], False
    for s in request.series:
        if len(s.values) != len(request.labels):
            errors.append(f"series {s.name!r} has {len(s.values)} values for {len(request.labels)} labels")
    units = {s.unit for s in request.series}
    if len(units) > 1:
        errors.append(f"different units on one axis ({', '.join(sorted(u or 'none' for u in units))}); "
                      "make one chart per unit, or convert with compute")
    if not errors:
        for s in request.series:
            for label, value in zip(request.labels, s.values):
                key = f"{s.name}:{label}"
                ref = request.value_refs.get(key)
                if ref is None:
                    errors.append(f"no source given for {key} (add it to value_refs)")
                    continue
                if not ledger.has(ref):
                    errors.append(f"{key}: source {ref} was not returned by any tool in this question")
                    continue
                ev = ledger.find_number(value, refs=[ref])
                if ev is None:
                    known = ledger.items[ref].numbers + ledger.items[ref].estimated
                    hint = f"; closest value there is {_fmt(min(known, key=lambda n: abs(n - value)))}" if known else ""
                    errors.append(f"{key}: value {_fmt(value)} not found in {ref}{hint}")
                elif ledger.is_estimated(value, ev):
                    approximate = True
    for ref in request.citations:
        if not ledger.has(ref):
            errors.append(f"citation {ref} was not returned by any tool in this question")

    chart_type = request.chart_type
    if chart_type == "pie" and not errors:
        s = request.series
        whole = (len(s) == 1 and len(request.labels) <= MAX_PIE_SLICES and all(v >= 0 for v in s[0].values)
                 and s[0].unit == "%" and abs(sum(s[0].values) - 100) <= 1.0)
        if not whole:
            chart_type = "bar"
            notes.append("drawn as a bar chart: a pie needs one series of percentages adding up to 100 "
                         f"(at most {MAX_PIE_SLICES} slices)")

    head = [""] + [f"{s.name} ({s.unit})" if s.unit else s.name for s in request.series]
    rows = [[label] + [_fmt(s.values[i]) for s in request.series] for i, label in enumerate(request.labels)] \
        if not any(len(s.values) != len(request.labels) for s in request.series) else []
    digest = hashlib.sha256(request.model_dump_json().encode("utf-8")).hexdigest()[:10]
    return ChartResult(ok=not errors, chart_id=None if errors else f"chart-{digest}", chart_type=chart_type,
                       approximate=approximate, data_table=[head, *rows], errors=errors, notes=notes,
                       citations=list(request.citations), title=request.title)


def render_chart(result: ChartResult, request: ChartRequest, out_dir: Path, footnote: str = "",
                 png: bool = True) -> ChartResult:
    """Draw the validated chart: Plotly JSON + PNG (kaleido), with the data table and a footnote."""
    import plotly.graph_objects as go

    title = request.title + (" (approximate)" if result.approximate else "")
    unit = request.series[0].unit if request.series else None
    if result.chart_type == "pie":
        s = request.series[0]
        fig = go.Figure(go.Pie(labels=request.labels, values=s.values, textinfo="label+percent", sort=False))
    else:
        fig = go.Figure()
        for s in request.series:
            kwargs = dict(x=request.labels, y=s.values, name=s.name, text=[_fmt(v) for v in s.values],
                          hovertemplate="%{x}: %{y}" + (f" {s.unit}" if s.unit else "") + "<extra>%{fullData.name}</extra>")
            fig.add_trace(go.Bar(**kwargs, textposition="outside") if result.chart_type == "bar"
                          else go.Scatter(**kwargs, mode="lines+markers+text", textposition="top center"))
        fig.update_layout(barmode="group", yaxis_title=unit or None)
    fig.update_layout(title=title, template="plotly_white", margin=dict(t=70, b=90, l=60, r=30),
                      showlegend=len(request.series) > 1 or result.chart_type == "pie")
    if footnote:
        fig.add_annotation(text=footnote, xref="paper", yref="paper", x=0, y=-0.22, showarrow=False,
                           font=dict(size=11, color="#5a6475"), xanchor="left")
    result.spec = json.loads(fig.to_json())
    if png:
        out_dir.mkdir(parents=True, exist_ok=True)
        result.png_path = out_dir / f"{result.chart_id}.png"
        fig.write_image(result.png_path, width=900, height=520, scale=2)
    return result


def make_chart(request: ChartRequest, ledger: EvidenceLedger, out_dir: Path, footnote: str = "",
               png: bool = True) -> ChartResult:
    """The tool: validate, then render if valid."""
    result = validate_chart(request, ledger)
    return render_chart(result, request, out_dir, footnote, png) if result.ok else result
