"""Agent tools (spec §7.1, LLD §5.3): plain Python functions the model may call.

`TOOL_SPECS` describes them to OpenAI (function calling). `dispatch` runs the calls of one
round in parallel: it validates the arguments, runs each function with a timeout, and turns
any error into an error result for the model instead of raising. Every tool records what it
returned in the evidence ledger, which the chart engine and the validator check. No MCP (D16).
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from mmrag import db
from mmrag.agent.compute import ComputeError, evaluate
from mmrag.agent.ledger import Evidence, EvidenceLedger, numbers_in
from mmrag.charts.engine import ChartRequest, ChartResult, make_chart
from mmrag.config import Settings
from mmrag.retrieval.hybrid import search


@dataclass
class ToolContext:
    """What every tool may use during one question."""

    settings: Settings
    ledger: EvidenceLedger
    embed_query: Callable[[str], list[float]]  # cached per query text
    chart_dir: Path  # where make_chart writes PNGs for this question
    charts: dict[str, ChartResult] = field(default_factory=dict)
    computed: int = 0
    source_note: Callable[[list[str]], str] | None = None  # chart footnote from cited ids; default: DB lookup


@dataclass
class ToolCall:
    call_id: str
    name: str
    arguments: dict


@dataclass
class ToolResult:
    call_id: str
    name: str
    content: str  # JSON text sent back to the model
    is_error: bool = False
    image_path: str | None = None  # view_page / get_figure(view_image): shown to the model next


# ---------------------------------------------------------------- argument models (= tool schemas)

class _Args(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SearchArgs(_Args):
    query: str = Field(description="What to look for, in your own words or with exact terms")
    k: int = Field(default=8, ge=1, le=20, description="How many results")


class FigureArgs(_Args):
    element_id: str = Field(description="A figure id from search_figures or a related item")
    view_image: bool = Field(default=False, description="Also look at the image itself")


class TableArgs(_Args):
    element_id: str = Field(description="A table id from search_tables or a related item")


class PageArgs(_Args):
    source_file: str = Field(description="The PDF file name, as shown in results")
    page: int = Field(ge=1)


class RefValue(_Args):
    value: float
    source: str = Field(description="The id of the evidence this number was read from")


class ComputeArgs(_Args):
    expression: str = Field(description="Arithmetic over the ref names, e.g. (a - b) / a * 100")
    refs: dict[str, RefValue] = Field(description="Each name used: its value and where it came from")


# ---------------------------------------------------------------- helpers

def _one(sql: str, params: tuple, settings: Settings):
    with db.connect(settings) as conn:
        return conn.execute(sql, params).fetchone()


def _related(h) -> list[dict]:
    return [{"id": r["element_id"], "type": r["type"], "page": r["page"], "title": r["title"]} for r in h.related]


def _record_related(ctx: ToolContext, hits) -> None:
    for h in hits:
        for r in h.related:
            ctx.ledger.add(Evidence(id=r["element_id"], kind="figure" if r["type"] in ("vector_figure", "image")
                                    else "table" if r["type"] == "table" else "chunk", text=r["title"] or "",
                                    numbers=sorted(numbers_in(r["title"] or ""))))


def _pages(h) -> list[int]:
    return sorted({loc["page"] for loc in h.locations})


# ---------------------------------------------------------------- search (Phase 3 hybrid search)

def search_text(ctx: ToolContext, query: str, k: int = 8) -> list[dict]:
    """Top text chunks: chunk_id, text, section_path, source_file, page, related items."""
    hits = search(ctx.settings, query, "text", k, embed_query=ctx.embed_query)
    for h in hits:
        ctx.ledger.add(Evidence(id=h.chunk_id, kind="chunk", text=h.dense_text, numbers=sorted(numbers_in(h.dense_text))))
    _record_related(ctx, hits)
    return [{"id": h.chunk_id, "source_file": h.source_file, "pages": _pages(h), "text": h.dense_text[:1500],
             "related": _related(h)} for h in hits]


def search_figures(ctx: ToolContext, query: str, k: int = 8) -> list[dict]:
    """Top figures: element_id, caption and description excerpt, page, related paragraphs."""
    hits = search(ctx.settings, query, "figure", k, embed_query=ctx.embed_query)
    out = []
    for h in hits:
        eid = h.locations[0]["element_id"]
        ctx.ledger.add(Evidence(id=eid, kind="figure", text=h.dense_text))
        out.append({"id": eid, "source_file": h.source_file, "page": h.locations[0]["page"],
                    "description": h.dense_text[:700], "related": _related(h)})
    _record_related(ctx, hits)
    return out


def search_tables(ctx: ToolContext, query: str, k: int = 8) -> list[dict]:
    """Top tables: element_id, title, summary, columns, page, related paragraphs."""
    hits = search(ctx.settings, query, "table", k, embed_query=ctx.embed_query)
    out = []
    for h in hits:
        eid = h.locations[0]["element_id"]
        ctx.ledger.add(Evidence(id=eid, kind="table", text=h.dense_text))
        out.append({"id": eid, "source_file": h.source_file, "page": h.locations[0]["page"],
                    "summary": h.dense_text[:700], "related": _related(h)})
    _record_related(ctx, hits)
    return out


# ---------------------------------------------------------------- fetch

def get_figure(ctx: ToolContext, element_id: str, view_image: bool = False) -> dict:
    """The full caption record (description, visible text, chart values with exact/estimated
    flags). With view_image, the figure image is shown to the model next."""
    row = _one("SELECT e.page, d.source_file, e.caption, e.asset_path, fc.short_caption, fc.detailed_description, "
               "fc.visible_text, fc.extracted_data FROM elements e JOIN documents d USING (doc_id) "
               "LEFT JOIN figure_captions fc USING (element_id) "
               "WHERE e.element_id = %s AND e.type IN ('vector_figure', 'image', 'scanned_page')",
               (element_id,), ctx.settings)
    if row is None:
        raise ValueError(f"unknown figure id {element_id}")
    page, source_file, pdf_caption, asset, short, desc, visible, data = row
    exact, estimated = [], []
    for s in ((data or {}).get("chart") or {}).get("series", []):
        percent = ((data or {}).get("chart") or {}).get("unit") == "%"
        for p in s["points"]:
            vals = [p["value"]] + ([round(p["value"] / 100, 10)] if percent else [])
            (exact if p["flag"] == "exact" else estimated).extend(vals)
    ctx.ledger.items.pop(element_id, None)  # replace the search stub with the full record
    ctx.ledger.add(Evidence(id=element_id, kind="figure", text=f"{short}\n{desc}", numbers=exact, estimated=estimated))
    out = {"id": element_id, "source_file": source_file, "page": page, "pdf_caption": pdf_caption,
           "short_caption": short, "description": desc, "visible_text": visible, "chart_values": data}
    if view_image and asset:
        out["image_path"] = str(ctx.settings.resolve(ctx.settings.paths.data_dir) / asset)
    return out


def get_table(ctx: ToolContext, element_id: str) -> dict:
    """The full structured table from doc_tables: columns, every row, units, title."""
    row = _one("SELECT e.page, d.source_file, t.columns, t.rows, t.units, t.title FROM doc_tables t "
               "JOIN elements e USING (element_id) JOIN documents d USING (doc_id) WHERE t.element_id = %s",
               (element_id,), ctx.settings)
    if row is None:
        raise ValueError(f"unknown table id {element_id}")
    page, source_file, columns, rows, units, title = row
    cells = " ".join(" ".join(r) for r in rows)
    ctx.ledger.items.pop(element_id, None)
    ctx.ledger.add(Evidence(id=element_id, kind="table", text=f"{title or ''} {' '.join(columns)} {cells}",
                            numbers=sorted(numbers_in(cells))))
    return {"id": element_id, "source_file": source_file, "page": page, "title": title, "columns": columns,
            "rows": rows, "units": units}


def view_page(ctx: ToolContext, source_file: str, page: int) -> dict:
    """A rendered page image (at ui.page_dpi) for the model to look at. Only files that are
    in `documents` are accepted; the PDF is always read from paths.pdf_dir (LLD §9)."""
    import pymupdf

    row = _one("SELECT doc_id FROM documents WHERE source_file = %s", (source_file,), ctx.settings)
    if row is None:
        raise ValueError(f"{source_file!r} is not an indexed document")
    pdf = ctx.settings.resolve(ctx.settings.paths.pdf_dir) / source_file
    out = ctx.settings.resolve(ctx.settings.paths.data_dir) / "cache" / "pages" / row[0] / f"p{page}.png"
    with pymupdf.open(pdf) as doc:
        if page > doc.page_count:
            raise ValueError(f"{source_file} has {doc.page_count} pages")
        text = doc[page - 1].get_text()
        if not out.is_file():
            out.parent.mkdir(parents=True, exist_ok=True)
            doc[page - 1].get_pixmap(dpi=ctx.settings.ui.page_dpi).save(out)
    eid = f"page:{source_file}:{page}"
    ctx.ledger.add(Evidence(id=eid, kind="page", text=text, numbers=sorted(numbers_in(text))))
    return {"id": eid, "source_file": source_file, "page": page, "text": text[:2000], "image_path": str(out)}


# ---------------------------------------------------------------- calculate and chart

def compute_tool(ctx: ToolContext, expression: str, refs: dict) -> dict:
    """Safe arithmetic. Each ref's value must be in its source; the result is recorded in the
    ledger (id compute:N) so charts may use it."""
    checked, estimated = {}, False
    for name, ref in refs.items():
        ev = ctx.ledger.find_number(ref["value"], refs=[ref["source"]])
        if ev is None:
            raise ComputeError(f"{name} = {ref['value']:g} is not in {ref['source']}")
        estimated |= ctx.ledger.is_estimated(ref["value"], ev)
        checked[name] = (ref["value"], ref["source"])
    result = evaluate(expression, checked)
    ctx.computed += 1
    eid = f"compute:{ctx.computed}"
    value = round(result.value, 6)
    ctx.ledger.add(Evidence(id=eid, kind="compute", text=f"{expression} = {value:g}",
                            numbers=[] if estimated else [value], estimated=[value] if estimated else []))
    return {"id": eid, "value": value, "inputs": {k: {"value": v, "source": s} for k, (v, s) in checked.items()}}


def _source_note(ids: list[str], settings: Settings) -> str:
    """"Source: file, p. N" for a chart footnote, from the cited ids."""
    with db.connect(settings) as conn:
        rows = conn.execute(
            "SELECT d.source_file, e.page FROM elements e JOIN documents d USING (doc_id) WHERE e.element_id = ANY(%s) "
            "UNION SELECT d.source_file, e.page FROM search_chunks c JOIN documents d USING (doc_id) "
            "JOIN elements e ON e.element_id = ANY(c.element_ids) WHERE c.chunk_id = ANY(%s)", (ids, ids)).fetchall()
    by_file: dict[str, set[int]] = {}
    for f, p in rows:
        by_file.setdefault(f, set()).add(p)
    return "Source: " + "; ".join(f"{f}, p. {', '.join(map(str, sorted(ps)))}" for f, ps in by_file.items()) \
        if by_file else ""


def make_chart_tool(ctx: ToolContext, **request) -> dict:
    """charts.engine.make_chart; returns a chart_id on success, or the reasons it was rejected."""
    req = ChartRequest(**request)
    note = ctx.source_note or (lambda ids: _source_note(ids, ctx.settings))
    cited = [c for c in req.citations if not c.startswith(("compute:", "page:"))]
    footnote = note(cited) if cited else ""
    result = make_chart(req, ctx.ledger, ctx.chart_dir, footnote=footnote)
    if not result.ok:
        raise _ChartRejected(result.errors)
    ctx.charts[result.chart_id] = result
    return {"chart_id": result.chart_id, "chart_type": result.chart_type, "approximate": result.approximate,
            "notes": result.notes}


class _ChartRejected(Exception):
    def __init__(self, errors: list[str]):
        super().__init__("; ".join(errors))
        self.errors = errors


TOOLS: dict[str, Callable] = {
    "search_text": search_text,
    "search_figures": search_figures,
    "search_tables": search_tables,
    "get_figure": get_figure,
    "get_table": get_table,
    "view_page": view_page,
    "compute": compute_tool,
    "make_chart": make_chart_tool,
}

_ARGS: dict[str, type[BaseModel]] = {
    "search_text": SearchArgs, "search_figures": SearchArgs, "search_tables": SearchArgs,
    "get_figure": FigureArgs, "get_table": TableArgs, "view_page": PageArgs,
    "compute": ComputeArgs, "make_chart": ChartRequest,
}

_DESCRIPTIONS = {
    "search_text": "Search the documents' text passages (meaning + exact words). Returns passage ids to cite, "
                   "pages, text, and related figures/tables.",
    "search_figures": "Search the figures (diagrams, charts, screenshots) by what they show. Returns figure ids "
                      "to cite, descriptions and related paragraphs.",
    "search_tables": "Search the tables by what they compare or contain. Returns table ids, summaries and columns.",
    "get_figure": "Full description of one figure, including chart values flagged exact or estimated. "
                  "Set view_image to look at the image.",
    "get_table": "Every row of one table, exactly as printed. Use before quoting or charting its numbers.",
    "view_page": "Look at a whole PDF page as an image, when the text is not enough. Costs more; use sparingly.",
    "compute": "Exact arithmetic on numbers from the evidence. Each ref gives its value and the id it came from. "
               "The result gets an id you can use in charts.",
    "make_chart": "Draw a bar, pie or line chart from numbers in the evidence. Every value needs its source id in "
                  "value_refs (key '<series name>:<label>'). Returns a chart_id to put in the answer, or why it "
                  "was rejected.",
}

TOOL_SPECS: list[dict] = [
    {"type": "function", "function": {"name": name, "description": _DESCRIPTIONS[name],
                                      "parameters": _ARGS[name].model_json_schema()}}
    for name in TOOLS
]


def _run(call: ToolCall, ctx: ToolContext) -> ToolResult:
    fn, model = TOOLS.get(call.name), _ARGS.get(call.name)
    if fn is None or model is None:
        return ToolResult(call.call_id, call.name, json.dumps({"error": f"unknown tool {call.name!r}"}), True)
    try:
        args = model.model_validate(call.arguments).model_dump()
    except ValidationError as e:
        return ToolResult(call.call_id, call.name, json.dumps({"error": f"invalid arguments: {e}"}), True)
    try:
        out = fn(ctx, **args)
    except _ChartRejected as e:
        return ToolResult(call.call_id, call.name, json.dumps({"errors": e.errors}), True)
    except Exception as e:  # every failure goes back to the model, never up the stack
        return ToolResult(call.call_id, call.name, json.dumps({"error": f"{type(e).__name__}: {e}"}), True)
    image = out.pop("image_path", None) if isinstance(out, dict) else None
    return ToolResult(call.call_id, call.name, json.dumps(out, default=str), image_path=image)


def dispatch(calls: list[ToolCall], ctx: ToolContext) -> list[ToolResult]:
    """Run one round's calls in parallel (`agent.max_parallel_tools` at once, `agent.tool_timeout_s`
    each), in the order given. Invalid arguments, unknown tools, exceptions and timeouts become
    error results."""
    if not calls:
        return []
    pool = ThreadPoolExecutor(max_workers=min(len(calls), ctx.settings.agent.max_parallel_tools))
    futures = [pool.submit(_run, c, ctx) for c in calls]
    results = []
    for call, fut in zip(calls, futures):
        try:
            results.append(fut.result(timeout=ctx.settings.agent.tool_timeout_s))
        except FutureTimeout:
            results.append(ToolResult(call.call_id, call.name, json.dumps(
                {"error": f"{call.name} timed out after {ctx.settings.agent.tool_timeout_s:g}s"}), True))
    pool.shutdown(wait=False, cancel_futures=True)
    return results
