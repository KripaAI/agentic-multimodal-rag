"""Answer page: one self-contained HTML file per question (plan Phase 4, task 6).

Renders the hydrated blocks in order: text with citation chips, original figures with
captions, interactive Plotly charts with their data tables and "approximate" label, tables,
then the sources list. Each citation shows file and page. Written to data/answers/.
"""

from __future__ import annotations

import base64
import json
import re
from datetime import datetime
from html import escape
from pathlib import Path

from markdown_it import MarkdownIt

from mmrag.agent.answer import HydratedBlock, HydratedCitation
from mmrag.agent.graph import QueryRun
from mmrag.config import Settings

PLOTLY_JS = "https://cdn.jsdelivr.net/npm/plotly.js-dist-min@2/plotly.min.js"
_md = MarkdownIt("commonmark", {"html": False}).enable("table")  # model text is never raw HTML

CSS = """
:root { --bg:#f6f7f9; --panel:#fff; --ink:#1b2330; --muted:#5a6475; --line:#d3d9e1; --accent:#0f766e;
  --warn:#9a3412; --chip:#e3f3f1; }
@media (prefers-color-scheme: dark) { :root { --bg:#12161c; --panel:#181e26; --ink:#e4e9f0; --muted:#9aa5b4;
  --line:#343d4a; --accent:#3cc9b8; --warn:#fdba74; --chip:#16332f; } }
body { margin:0; padding:24px 16px; background:var(--bg); color:var(--ink); font:15px/1.6 system-ui, sans-serif; }
main { max-width:900px; margin:0 auto; display:grid; gap:14px; }
section { background:var(--panel); border:1px solid var(--line); border-radius:10px; padding:12px 16px;
  overflow-x:auto; }
h1 { font-size:20px; margin:0; } h2 { font-size:13px; margin:0 0 6px; text-transform:uppercase;
  letter-spacing:.05em; color:var(--muted); }
.cite { display:inline-block; background:var(--chip); color:var(--accent); border-radius:4px; padding:0 6px;
  margin:2px 4px 0 0; font-size:12px; }
.approx { color:var(--warn); font-weight:600; } .notice { color:var(--warn); } .meta { color:var(--muted); font-size:12px; }
figure { margin:0; } figure img { max-width:100%; height:auto; border:1px solid var(--line); border-radius:6px; }
figcaption { color:var(--muted); font-size:13px; margin-top:4px; }
table { border-collapse:collapse; font-size:13px; margin-top:8px; } th, td { border:1px solid var(--line);
  padding:3px 8px; text-align:left; } th { background:var(--bg); }
details summary { cursor:pointer; color:var(--muted); font-size:13px; margin-top:6px; }
"""


def _chips(citations: list[HydratedCitation]) -> str:
    seen, chips = set(), []
    for c in citations:
        for loc in c.locations:
            key = (loc.source_file, loc.page)
            if key not in seen:
                seen.add(key)
                chips.append(f'<span class="cite" title="{escape(c.id)}">{escape(loc.source_file)}, '
                             f"p. {loc.page}</span>")
    return f"<div>{''.join(chips)}</div>" if chips else ""


def _table(rows: list[list], columns: list | None = None) -> str:
    head = f"<tr>{''.join(f'<th>{escape(str(c))}</th>' for c in columns)}</tr>" if columns else ""
    body = "".join(f"<tr>{''.join(f'<td>{escape(str(v))}</td>' for v in r)}</tr>" for r in rows)
    return f"<table>{head}{body}</table>"


def _image(path: Path) -> str:
    if not path.is_file():
        return f'<p class="notice">Figure file missing: {escape(str(path))}</p>'
    return f'<img alt="" src="data:image/png;base64,{base64.b64encode(path.read_bytes()).decode()}">'


def _block(b: HydratedBlock, n: int, data_dir: Path) -> str:
    c = b.content
    if b.type == "text":
        return f"<section>{_md.render(c['markdown'])}{_chips(b.citations)}</section>"
    if b.type == "image":
        return (f"<section><figure>{_image(data_dir / c['asset_path'])}<figcaption>"
                f"{escape(c.get('short_caption') or '')}</figcaption></figure>{_chips(b.citations)}</section>")
    if b.type == "table":
        title = f"<h2>{escape(c['title'])}</h2>" if c.get("title") else ""
        return f"<section>{title}{_table(c['rows'], c['columns'])}{_chips(b.citations)}</section>"
    # chart: interactive when plotly.js loads; the PNG and the data table always work offline
    label = '<p class="approx">Approximate: some values are estimated from a figure.</p>' if b.approximate else ""
    notes = "".join(f'<p class="meta">{escape(x)}</p>' for x in c.get("notes") or [])
    png = Path(c["png_path"]) if c.get("png_path") else None
    fallback = _image(png) if png else ""
    dt = c.get("data_table") or []
    data = f"<details><summary>Data table</summary>{_table(dt[1:], dt[0])}</details>" if dt else ""
    spec = json.dumps(c["spec"]).replace("</", "<\\/")
    return (f'<section>{label}<div id="chart{n}">{fallback}</div>{notes}{data}{_chips(b.citations)}'
            f"<script>if (window.Plotly) {{ const s = {spec}; const el = document.getElementById('chart{n}');"
            f" el.innerHTML = ''; Plotly.newPlot(el, s.data, s.layout, {{responsive: true}}); }}</script></section>")


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:50] or "answer"


def write_answer_page(question: str, run: QueryRun, settings: Settings) -> Path:
    """Write data/answers/<timestamp>-<slug>.html and return its path. The footer shows the
    thread, rounds, tokens, cost, latency and the Phoenix trace link."""
    data_dir = settings.resolve(settings.paths.data_dir)
    a = run.answer
    has_chart = any(b.type == "chart" for b in a.blocks)
    notices = "".join(f'<p class="notice">{escape(x)}</p>' for x in a.notices)
    sources = "".join(f"<li>{escape(s.source_file)}, p. {s.page} <span class=\"meta\">{escape(s.element_id)}</span></li>"
                      for s in a.sources)
    cost = f"${run.cost_usd:.4f}" if run.cost_usd is not None else "cost n/a (model not priced)"
    endpoint = settings.observability.otlp_traces_endpoint or ""
    base = endpoint.rsplit("/v1/", 1)[0] if endpoint else ""
    trace = (f'<a href="{escape(base)}/redirects/traces/{run.trace_id}">{run.trace_id}</a>' if base
             else run.trace_id)
    html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Answer</title><style>{CSS}</style>
{f'<script src="{PLOTLY_JS}"></script>' if has_chart else ''}</head><body><main>
<h1>{escape(question)}</h1>{notices}
{''.join(_block(b, n, data_dir) for n, b in enumerate(a.blocks))}
<section><h2>Sources</h2><ol>{sources or '<li>(none)</li>'}</ol></section>
<p class="meta">thread {escape(run.thread_id)} · model {escape(run.model)} · {escape(run.qtype)} · {run.rounds} rounds ·
{len(run.tool_calls)} tool calls · {run.input_tokens:,} in / {run.output_tokens:,} out tokens · {cost} ·
{run.latency_ms / 1000:.1f} s · validator {escape(run.validator_result)} · trace {trace}</p>
</main></body></html>
"""
    out = data_dir / "answers" / f"{datetime.now():%Y%m%d-%H%M%S}-{_slug(question)}.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    return out
