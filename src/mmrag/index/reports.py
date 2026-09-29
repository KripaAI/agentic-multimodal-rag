"""Owner-facing pages for Phase 3: the link report and the table-summary model comparison."""

from __future__ import annotations

from html import escape
from pathlib import Path

from mmrag.config import Settings

_STYLE = """
:root { --bg:#f6f7f9; --panel:#fff; --ink:#1b2330; --muted:#5a6475; --line:#d3d9e1; --ok:#0f766e; --warn:#b45309;
  --bad:#b42318; }
@media (prefers-color-scheme: dark) { :root { --bg:#12161c; --panel:#181e26; --ink:#e4e9f0; --muted:#9aa5b4;
  --line:#343d4a; --ok:#3cc9b8; --warn:#f0a44b; --bad:#f07a6e; } }
* { box-sizing:border-box; }
body { margin:0; padding:24px 16px; background:var(--bg); color:var(--ink);
  font:14px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; }
main { max-width:1300px; margin:0 auto; display:grid; gap:14px; }
h1 { font-size:22px; margin:0; } .muted { color:var(--muted); } .small { font-size:12px; }
.row { background:var(--panel); border:1px solid var(--line); border-radius:10px; padding:12px;
  display:grid; grid-template-columns:minmax(0, 320px) minmax(0, 1fr); gap:14px; }
@media (max-width:800px) { .row { grid-template-columns:1fr; } }
.row img { width:100%; border:1px solid var(--line); background:#fff; }
.tag { font-size:11px; font-weight:600; text-transform:uppercase; letter-spacing:.05em; }
.explicit, .deictic { color:var(--ok); } .related { color:var(--warn); } .unlinked { color:var(--bad); }
table.cmp { border-collapse:collapse; width:100%; display:block; overflow-x:auto; }
.cmp td, .cmp th { border:1px solid var(--line); padding:6px 8px; vertical-align:top; text-align:left; }
code { font:12px ui-monospace, Consolas, monospace; }
"""


def _page(title: str, body: str) -> str:
    return (f"<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\"><meta name=\"viewport\" "
            f"content=\"width=device-width, initial-scale=1\"><title>{escape(title)}</title><style>{_STYLE}</style>"
            f"</head><body><main>{body}</main></body></html>\n")


def write_link_report(report, settings: Settings) -> Path:
    """Every figure and table with its linked paragraph(s), the rule and the score."""
    doc = report.doc
    data = settings.resolve(settings.paths.data_dir)
    out = data / "elements" / doc.doc_id / "link_report.html"
    texts = {e.element_id: e for e in doc.texts()}
    by_target: dict[str, list] = {}
    for lk in report.links:
        by_target.setdefault(lk.target_id, []).append(lk)
    rows = []
    for el in sorted(doc.figures() + doc.table_elements(), key=lambda e: (e.page, e.bbox[1])):
        record = doc.captions.get(el.element_id)
        title = (record.caption.short_caption if record and record.caption else None) or el.caption or \
            (doc.tables[el.element_id].title if el.element_id in doc.tables else None) or "(no caption)"
        img = f"<img loading=\"lazy\" src=\"../../{escape(el.asset_path)}\" alt=\"\">" if el.asset_path else ""
        found = by_target.get(el.element_id, [])
        if found:
            links = "".join(
                f"<p><span class=\"tag {lk.method}\">{lk.method} · {lk.score:.2f}</span> "
                f"<span class=\"small muted\">p{texts[lk.text_element_id].page}</span><br>"
                f"{escape(texts[lk.text_element_id].text[:500])}</p>" for lk in found)
        else:
            links = "<p class=\"tag unlinked\">unlinked: no paragraph scored above the threshold</p>"
        share = report.label_shares.get(el.element_id)
        check = f" · label match {share:.0%}" if share is not None else ""
        rows.append(f"<section class=\"row\"><div>{img}<div class=\"small\"><code>{escape(el.element_id)}</code>"
                    f"</div><div class=\"small muted\">p{el.page} · {escape(el.type)}{check}</div></div>"
                    f"<div><b>{escape(title)}</b>{links}</div></section>")
    methods = {m: sum(lk.method == m for lk in report.links) for m in ("explicit", "deictic", "related")}
    body = (f"<h1>Link report: {escape(doc.source_file)}</h1><div class=\"muted\">{len(rows)} figures and tables · "
            f"{methods['explicit']} explicit · {methods['deictic']} deictic · {methods['related']} related · "
            f"{len(report.unlinked)} unlinked. Green links come from a reference or a pointing phrase; amber ones "
            f"were chosen as the most related nearby paragraph (score = meaning + shared labels).</div>"
            + "".join(rows))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(_page("Link report", body), encoding="utf-8")
    return out


def write_summary_comparison(doc, results: dict[str, dict], out: Path) -> Path:
    """results: model -> {element_id: Completion}. One row per table, one column per model."""
    models = list(results)
    totals = "".join(
        f"<th>{escape(m)}<div class=\"small muted\">{sum(c.input_tokens for c in r.values())} in / "
        f"{sum(c.output_tokens for c in r.values())} out tokens · "
        f"{sum(c.seconds for c in r.values()) / max(len(r), 1):.1f}s per table</div></th>"
        for m, r in results.items())
    rows = []
    for eid, table in doc.tables.items():
        head = f"<b>{escape(table.title or '(untitled)')}</b><div class=\"small muted\">p{eid.split(':')[1][1:]}: " \
               f"{escape(' | '.join(table.columns))}</div>"
        cells = "".join(f"<td>{escape(results[m][eid].text)}<div class=\"small muted\">{results[m][eid].seconds}s</div>"
                        f"</td>" for m in models)
        rows.append(f"<tr><td>{head}</td>{cells}</tr>")
    body = (f"<h1>Table summaries: {escape(doc.source_file)}</h1><div class=\"muted\">Pick the model whose "
            f"summaries say what each table compares, correctly, in the table's own terms. Token counts include "
            f"hidden reasoning tokens for reasoning models.</div><table class=\"cmp\"><tr><th>Table</th>{totals}</tr>"
            + "".join(rows) + "</table>")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(_page("Summary comparison", body), encoding="utf-8")
    return out
