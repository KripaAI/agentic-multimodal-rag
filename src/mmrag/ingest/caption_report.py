"""Pilot comparison page: each figure beside every model's caption (plan Phase 2, task 4).

The owner reads it to choose the model path. It shows, per model, whether it loaded,
how many captions were schema-valid, speed and GPU memory; and per figure, the image,
the PDF's caption and each model's type, captions, visible text and chart or table values.
"""

from __future__ import annotations

import json
import os
from html import escape
from pathlib import Path

MODEL_ORDER = ["awq-7b", "nf4-7b", "3b"]
_STYLE = """
:root { --bg:#f6f7f9; --panel:#fff; --ink:#1b2330; --muted:#5a6475; --line:#d3d9e1;
  --ok:#0f766e; --warn:#b45309; --bad:#b42318; }
@media (prefers-color-scheme: dark) { :root { --bg:#12161c; --panel:#181e26; --ink:#e4e9f0; --muted:#9aa5b4;
  --line:#343d4a; --ok:#3cc9b8; --warn:#f0a44b; --bad:#f07a6e; } }
* { box-sizing:border-box; }
body { margin:0; padding:24px 16px; background:var(--bg); color:var(--ink);
  font:14px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; }
main { max-width:1400px; margin:0 auto; display:grid; gap:18px; }
h1 { font-size:22px; margin:0; } h2 { font-size:15px; margin:0 0 8px; }
.muted { color:var(--muted); }
.models { display:grid; grid-template-columns:repeat(auto-fit, minmax(240px, 1fr)); gap:10px; }
.card { background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:12px; }
.card b { font-size:18px; }
.fig { background:var(--panel); border:1px solid var(--line); border-radius:10px; padding:14px;
  display:grid; grid-template-columns:minmax(0, 380px) repeat(auto-fit, minmax(260px, 1fr)); gap:14px; }
@media (max-width:900px) { .fig { grid-template-columns:1fr; } }
.fig img { width:100%; border:1px solid var(--line); background:#fff; }
.col { border-left:1px solid var(--line); padding-left:12px; min-width:0; }
.tag { font-size:11px; font-weight:600; text-transform:uppercase; letter-spacing:.05em; }
.ok { color:var(--ok); } .warn { color:var(--warn); } .bad { color:var(--bad); }
.small { font-size:12px; } .desc { white-space:pre-wrap; }
table { border-collapse:collapse; font-size:12px; margin-top:4px; display:block; overflow-x:auto; }
td, th { border:1px solid var(--line); padding:2px 6px; text-align:left; }
code { font:12px ui-monospace, Consolas, monospace; }
"""


def _load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()] \
        if path.is_file() else []


def _data_html(data: dict | None) -> str:
    if not data:
        return ""
    parts = []
    if chart := data.get("chart"):
        rows = "".join(f"<tr><td>{escape(s['name'])}</td><td>{escape(p['label'])}</td><td>{p['value']:g}</td>"
                       f"<td class=\"{'ok' if p['flag'] == 'exact' else 'warn'}\">{p['flag']}</td></tr>"
                       for s in chart["series"] for p in s["points"])
        parts.append(f"<div class=\"small\">chart: {escape(chart['chart_kind'])}"
                     f"{' · unit ' + escape(chart['unit']) if chart.get('unit') else ''}</div>"
                     f"<table><tr><th>series</th><th>label</th><th>value</th><th>flag</th></tr>{rows}</table>")
    if table := data.get("table"):
        head = "".join(f"<th>{escape(c)}</th>" for c in table["columns"])
        body = "".join("<tr>" + "".join(f"<td>{escape(c)}</td>" for c in r) + "</tr>" for r in table["rows"])
        parts.append(f"<table><tr>{head}</tr>{body}</table>")
    return "".join(parts)


def _caption_html(record: dict | None) -> str:
    if record is None:
        return '<div class="muted">no output</div>'
    head = f"<span class=\"small muted\">{record['seconds']:.1f}s</span>"
    if record["status"] != "ok":
        return (f"<div class=\"tag bad\">needs review</div>{head}<div class=\"small\">{escape(record.get('error') or '')}"
                f"</div><pre class=\"small desc\">{escape((record.get('raw') or '')[:1500])}</pre>")
    c = record["caption"]
    conf = {"high": "ok", "medium": "warn", "low": "bad"}[c["confidence"]]
    return (f"<div class=\"tag\">{escape(c['figure_type'])} · <span class=\"{conf}\">{c['confidence']}</span></div>{head}"
            f"<p><b>{escape(c['short_caption'])}</b></p><p class=\"desc\">{escape(c['detailed_description'])}</p>"
            f"<div class=\"small muted\">visible text ({len(c['visible_text'])}): "
            f"{escape(' | '.join(c['visible_text'])[:600])}</div>"
            f"<div class=\"small muted\">keywords: {escape(', '.join(c['keywords']))}</div>"
            f"{_data_html(c.get('extracted_data'))}")


def build_caption_review(doc_id: str, settings) -> Path:
    """All imported captions of one document beside their figures, for the owner's spot-check
    (plan Phase 2 gate). Written to data/captions/{doc_id}/caption_review.html."""
    data = settings.resolve(settings.paths.data_dir)
    elements = {e["element_id"]: e for e in _load(data / "elements" / doc_id / "elements.jsonl")}
    records = _load(data / "captions" / doc_id / "captions.jsonl")
    records.sort(key=lambda r: (elements.get(r["element_id"], {}).get("page", 0), r["element_id"]))
    out = data / "captions" / doc_id / "caption_review.html"

    rows = []
    for n, r in enumerate(records, 1):
        e = elements.get(r["element_id"], {})
        img = f"../../{e['asset_path']}" if e.get("asset_path") else ""
        note = f"<div class=\"small warn\">check applied: {escape(r['error'])}</div>" if r.get("error") and \
            r["status"] == "ok" else ""
        rows.append(
            f"<section class=\"fig\"><div><div class=\"tag muted\">#{n} · page {e.get('page', '?')}</div>"
            f"<img loading=\"lazy\" src=\"{escape(img)}\" alt=\"{escape(r['element_id'])}\">"
            f"<div class=\"small\"><code>{escape(r['element_id'])}</code></div>"
            f"<div class=\"small muted\">Section: {escape(' > '.join(e.get('section_path', [])) or '(none)')}</div>"
            f"<div class=\"small muted\">PDF caption: {escape(e.get('caption') or '(none)')}</div></div>"
            f"<div class=\"col\">{_caption_html(r)}{note}</div></section>")
    ok = sum(r["status"] == "ok" for r in records)
    out.write_text(f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Caption review</title><style>{_STYLE}</style></head>
<body><main><h1>Caption review: {len(records)} figures</h1>
<div class="muted">{ok} valid · {len(records) - ok} need review · model {escape(records[0]['model_id'] if records else '')}.
Spot-check any 10: a caption passes if it names the key labels correctly and describes the right flow or values.
Green chart values are printed on the figure; amber ones were read off the bars.</div>
{''.join(rows)}</main></body></html>
""", encoding="utf-8")
    return out


def build_pilot_report(run_dir: Path, bundle: Path, out: Path) -> Path:
    jobs = _load(bundle / "jobs.jsonl")
    models = [m for m in MODEL_ORDER if (run_dir / m).is_dir()]
    by_model = {m: {r["element_id"]: r for r in _load(run_dir / m / "captions.jsonl")} for m in models}

    cards = []
    for m in models:
        report_file, error_file = run_dir / m / "run_report.json", run_dir / m / "load_error.txt"
        if report_file.is_file():
            rep = json.loads(report_file.read_text(encoding="utf-8"))
            cards.append(f"<div class=\"card\"><div class=\"tag\">{m}</div><b>{rep['ok']}/{rep['jobs']}</b> valid"
                         f"<div class=\"small muted\">{escape(rep['model_id'])}<br>{rep['seconds_per_figure']}s per figure"
                         f" · load {rep.get('load_seconds', '?')}s<br>GPU MB after run: {rep['gpu_memory_mb']}</div></div>")
        else:
            err = error_file.read_text(encoding="utf-8") if error_file.is_file() else "no report (crashed?)"
            cards.append(f"<div class=\"card\"><div class=\"tag\">{m}</div><b class=\"bad\">failed</b>"
                         f"<div class=\"small\">{escape(err[:400])}</div></div>")

    rel = Path(os.path.relpath(bundle, out.parent))
    figures = []
    for j in jobs:
        cols = "".join(f"<div class=\"col\"><div class=\"tag muted\">{m}</div>{_caption_html(by_model[m].get(j['element_id']))}"
                       f"</div>" for m in models)
        figures.append(f"<section class=\"fig\"><div><img loading=\"lazy\" src=\"{escape((rel / j['image']).as_posix())}\" "
                       f"alt=\"{escape(j['element_id'])}\"><div class=\"small\"><code>{escape(j['element_id'])}</code></div>"
                       f"<div class=\"small muted\">PDF caption: {escape(j['pdf_caption'] or '(none)')}</div></div>"
                       f"{cols}</section>")

    out.write_text(f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Caption pilot</title><style>{_STYLE}</style></head>
<body><main><h1>Caption pilot: {len(jobs)} figures × {len(models)} models</h1>
<div class="muted">Pick the model whose captions name every box and arrow correctly and read chart values accurately.
Green flags are values printed on the figure; amber ones were estimated from bar heights.</div>
<div class="models">{''.join(cards)}</div>{''.join(figures)}</main></body></html>
""", encoding="utf-8")
    return out
