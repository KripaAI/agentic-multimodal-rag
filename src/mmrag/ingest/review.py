"""Review sheet: one HTML page for the owner to check detection (plan Phase 1, task 7).

Per page: a thumbnail with every detected figure, image, table and rejected
region outlined, next to the cropped figure PNGs, the extracted tables and the
rejected regions with the filter that dropped them. Summary counts at the top.
Page thumbnails are written to a `review/` folder next to the sheet; figure
PNGs are referenced from `data/assets/` by relative path.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from html import escape
from pathlib import Path

import pymupdf

from mmrag.ingest.models import ParseResult

_THUMB_DPI = 60
_MAX_TABLE_ROWS = 8

_STYLE = """
:root { --bg:#fafafa; --fg:#1d1d1f; --muted:#6b6b70; --card:#fff; --line:#e3e3e8;
  --figure:#2563eb; --image:#059669; --table:#d97706; --rejected:#dc2626; --scanned:#7c3aed; }
@media (prefers-color-scheme: dark) { :root { --bg:#141416; --fg:#ececef; --muted:#9a9aa2; --card:#1d1d20; --line:#333338; } }
* { box-sizing: border-box; }
body { margin:0; padding:24px 16px; background:var(--bg); color:var(--fg);
  font:14px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; }
main { max-width:1200px; margin:0 auto; }
h1 { font-size:22px; margin:0 0 4px; } h2 { font-size:16px; margin:0 0 12px; }
.muted { color:var(--muted); }
.summary { display:flex; flex-wrap:wrap; gap:8px; margin:16px 0; }
.stat { background:var(--card); border:1px solid var(--line); border-radius:8px; padding:8px 12px; }
.stat b { font-size:18px; display:block; }
.legend span { display:inline-block; margin-right:14px; }
.legend i { display:inline-block; width:12px; height:12px; border:2px solid; margin-right:4px; vertical-align:-1px; }
.controls { margin:8px 0 20px; }
section.page { background:var(--card); border:1px solid var(--line); border-radius:10px; padding:16px; margin:0 0 16px;
  display:grid; grid-template-columns:minmax(0, 360px) minmax(0, 1fr); gap:16px; }
section.page.empty-page { display:none; } body.show-all section.page.empty-page { display:grid; }
@media (max-width:760px) { section.page { grid-template-columns:1fr; } }
.thumb { position:relative; align-self:start; border:1px solid var(--line); }
.thumb img { display:block; width:100%; height:auto; }
.box { position:absolute; border:2px solid; }
.box span { position:absolute; top:-2px; left:-2px; font-size:10px; line-height:1; padding:2px 3px; color:#fff; }
.figure { border-color:var(--figure); } .figure span { background:var(--figure); }
.image { border-color:var(--image); } .image span { background:var(--image); }
.table { border-color:var(--table); } .table span { background:var(--table); }
.scanned { border-color:var(--scanned); } .scanned span { background:var(--scanned); }
.rejected { border-color:var(--rejected); border-style:dashed; } .rejected span { background:var(--rejected); }
.item { border-top:1px solid var(--line); padding:10px 0; }
.item:first-child { border-top:0; padding-top:0; }
.item img { max-width:100%; max-height:420px; border:1px solid var(--line); background:#fff; }
.tag { font-size:11px; font-weight:600; text-transform:uppercase; letter-spacing:.04em; }
.id { font:12px ui-monospace, Consolas, monospace; color:var(--muted); word-break:break-all; }
.cap { font-style:italic; }
table.data { border-collapse:collapse; font-size:12px; margin-top:6px; display:block; overflow-x:auto; }
table.data th, table.data td { border:1px solid var(--line); padding:3px 6px; text-align:left; vertical-align:top; }
"""

_KIND = {"vector_figure": "figure", "image": "image", "table": "table", "scanned_page": "scanned"}


def _box(bbox, page_rect: pymupdf.Rect, cls: str, label: str) -> str:
    x0, y0, x1, y1 = bbox
    w, h = page_rect.width, page_rect.height
    style = (f"left:{100 * x0 / w:.2f}%;top:{100 * y0 / h:.2f}%;"
             f"width:{100 * (x1 - x0) / w:.2f}%;height:{100 * (y1 - y0) / h:.2f}%")
    return f'<div class="box {cls}" style="{style}"><span>{escape(label)}</span></div>'


def build_review_sheet(result: ParseResult, pdf_path: Path, out: Path) -> Path:
    """Write the review sheet to `out`; images are referenced by relative path."""
    thumbs = out.parent / "review"
    thumbs.mkdir(parents=True, exist_ok=True)
    tables = {t.element_id: t for t in result.tables}
    by_page = defaultdict(list)
    for e in result.elements:
        if e.type != "text" and e.status == "ok":
            by_page[e.page].append(e)
    rejected = defaultdict(list)
    for r in result.rejected:
        rejected[r.page].append(r)
    to_data = "../../"  # from data/elements/{doc_id}/ back to data/

    sections = []
    with pymupdf.open(pdf_path) as doc:
        for page in doc:
            n = page.number + 1
            thumb = thumbs / f"p{n:03d}.png"
            page.get_pixmap(dpi=_THUMB_DPI).save(thumb)
            boxes, items = [], []
            for e in by_page[n]:
                kind = _KIND[e.type]
                short = e.element_id.split(":", 1)[1]
                boxes.append(_box(e.bbox, page.rect, kind, f"{kind} {short.rsplit(':', 1)[1]}"))
                parts = [f'<div class="tag" style="color:var(--{kind})">{escape(e.type)}</div>',
                         f'<div class="id">{escape(e.element_id)}</div>']
                if e.caption:
                    parts.append(f'<div class="cap">{escape(e.caption)}</div>')
                if e.asset_path:
                    parts.append(f'<img loading="lazy" src="{escape(to_data + e.asset_path)}" alt="{escape(short)}">')
                if e.type == "table":
                    parts.append(_table_html(tables[e.element_id]))
                items.append(f'<div class="item">{"".join(parts)}</div>')
            for i, r in enumerate(rejected[n], 1):
                boxes.append(_box(r.bbox, page.rect, "rejected", f"x{i}"))
                items.append(f'<div class="item"><div class="tag" style="color:var(--rejected)">rejected x{i}: '
                             f'{escape(r.filter)}</div><div class="muted">{r.shape_count} shape(s) at '
                             f'{", ".join(f"{v:.0f}" for v in r.bbox)}</div></div>')
            empty = "" if items else " empty-page"
            body = "".join(items) or '<p class="muted">Nothing detected on this page.</p>'
            sections.append(
                f'<section class="page{empty}" id="p{n}"><div class="thumb">'
                f'<img loading="lazy" src="review/{thumb.name}" alt="page {n}">{"".join(boxes)}</div>'
                f'<div><h2>Page {n}</h2>{body}</div></section>'
            )

    ok = Counter(e.type for e in result.elements if e.status == "ok")
    stats = [("vector figures", ok["vector_figure"]), ("images", ok["image"]), ("tables", ok["table"]),
             ("scanned pages", ok["scanned_page"]), ("text blocks", ok["text"]),
             ("rejected regions", len(result.rejected)), ("skipped", len(result.skips)
                                                         + sum(e.status == "skipped" for e in result.elements))]
    low = sum(t.low_confidence for t in result.tables)
    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Review sheet: {escape(result.source_file)}</title><style>{_STYLE}</style></head>
<body><main>
<h1>Review sheet: {escape(result.source_file)}</h1>
<div class="muted">doc_id {escape(result.doc_id)} · {result.page_count} pages · {low} low-confidence table(s)</div>
<div class="summary">{"".join(f'<div class="stat"><b>{v}</b>{escape(k)}</div>' for k, v in stats)}</div>
<div class="legend"><span><i style="border-color:var(--figure)"></i>vector figure</span>
<span><i style="border-color:var(--image)"></i>image</span><span><i style="border-color:var(--table)"></i>table</span>
<span><i style="border-color:var(--scanned)"></i>scanned page</span>
<span><i style="border-color:var(--rejected);border-style:dashed"></i>rejected region</span></div>
<div class="controls"><label><input type="checkbox" onchange="document.body.classList.toggle('show-all', this.checked)">
Show pages with nothing detected (to spot missed figures)</label></div>
{"".join(sections)}
</main></body></html>
"""
    out.write_text(html, encoding="utf-8")
    return out


def _table_html(table) -> str:
    head = "".join(f"<th>{escape(c)}</th>" for c in table.columns)
    rows = "".join("<tr>" + "".join(f"<td>{escape(c)}</td>" for c in r) + "</tr>" for r in table.rows[:_MAX_TABLE_ROWS])
    more = len(table.rows) - _MAX_TABLE_ROWS
    note = f'<div class="muted">… {more} more row(s)</div>' if more > 0 else ""
    flags = []
    if table.title:
        flags.append(f"title: {escape(table.title)}")
    if table.numeric_columns:
        flags.append("numeric: " + escape(", ".join(table.numeric_columns)))
    if table.low_confidence:
        flags.append("<b>low confidence</b> (also rendered for the VLM)")
    meta = f'<div class="muted">{" · ".join(flags)}</div>' if flags else ""
    return f'{meta}<table class="data"><tr>{head}</tr>{rows}</table>{note}'
