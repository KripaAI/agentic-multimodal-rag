"""Retrieval test report (plan Phase 3 task 9, the Phase 3 gate).

A fixed list of queries, each with the elements that should come back and the collection
whose search should find them. A query passes when a top-5 result of that collection covers
an expected element. The page also shows the other collections' top results and each
result's related items, for context.
"""

from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict

from mmrag.config import Settings
from mmrag.ingest.ids import doc_id
from mmrag.retrieval.hybrid import COLLECTIONS, Hit, search

TOP = 5


class Query(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    query: str
    source: str  # PDF file name
    collection: Literal["text", "figure", "table"]
    expect: list[str]  # element ids without the doc prefix, e.g. p3:vector_figure:2; any one counts
    kind: Literal["conceptual", "visual", "table", "exact_term"]
    note: str | None = None
    also: dict[str, list[str]] = {}  # other PDF file -> element ids that also count (topic in several books)


def load_queries(path: Path) -> list[Query]:
    return [Query.model_validate(q) for q in yaml.safe_load(path.read_text(encoding="utf-8"))]


def expected_ids(q: Query, doc_ids: dict[str, str]) -> set[str]:
    """Full element ids that satisfy the query: `expect` in `source`, plus any `also` files."""
    wanted = {q.source: q.expect, **q.also}
    return {f"{doc_ids[src]}:{e}" for src, ids in wanted.items() for e in ids}


def found_rank(hits: list[Hit], expected: set[str]) -> int | None:
    for n, h in enumerate(hits, 1):
        if any(loc["element_id"] in expected for loc in h.locations):
            return n
    return None


def run_retrieval_report(settings: Settings, queries_file: Path) -> tuple[Path, int, int]:
    queries = load_queries(queries_file)
    pdf_dir = settings.resolve(settings.paths.pdf_dir)
    doc_ids = {src: doc_id(pdf_dir / src) for q in queries for src in [q.source, *q.also]}
    sections, passed = [], 0
    for q in queries:
        expected = expected_ids(q, doc_ids)
        results = {c: search(settings, q.query, c, TOP) for c in COLLECTIONS}
        rank = found_rank(results[q.collection], expected)
        passed += rank is not None
        cols = []
        for c in COLLECTIONS:
            items = []
            for n, h in enumerate(results[c], 1):
                hit = any(loc["element_id"] in expected for loc in h.locations)
                pages = ",".join(str(p) for p in sorted({loc["page"] for loc in h.locations}))
                rel = "".join(f"<div class=\"rel\">↳ {escape(r['type'])} p{r['page']}: "
                              f"{escape(' '.join((r['title'] or '').split())[:70])}</div>" for r in h.related[:3])
                items.append(f"<li class=\"{'hit' if hit else ''}\"><span class=\"meta\">p{pages} · "
                             f"sem {h.semantic_rank or '–'} / kw {h.keyword_rank or '–'}</span> "
                             f"{escape(' '.join(h.dense_text.split())[:160])}{rel}</li>")
            cols.append(f"<div class=\"col{' target' if c == q.collection else ''}\"><h3>{c}</h3>"
                        f"<ol>{''.join(items) or '<li>(none)</li>'}</ol></div>")
        verdict = f"<span class=\"pass\">found at #{rank}</span>" if rank else "<span class=\"fail\">not in top 5</span>"
        sections.append(f"<section><h2>{escape(q.id)} · {escape(q.kind)} · {verdict}</h2>"
                        f"<p class=\"q\">“{escape(q.query)}”</p><p class=\"meta\">expected in {q.collection}: "
                        f"{escape(', '.join(q.expect))}{' · ' + escape(q.note) if q.note else ''}</p>"
                        f"<div class=\"cols\">{''.join(cols)}</div></section>")

    out = settings.resolve(settings.paths.data_dir) / "retrieval_report.html"
    out.write_text(f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Retrieval report</title><style>
:root {{ --bg:#f6f7f9; --panel:#fff; --ink:#1b2330; --muted:#5a6475; --line:#d3d9e1; --ok:#0f766e; --bad:#b42318;
  --hl:#e3f3f1; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#12161c; --panel:#181e26; --ink:#e4e9f0; --muted:#9aa5b4;
  --line:#343d4a; --ok:#3cc9b8; --bad:#f07a6e; --hl:#16332f; }} }}
body {{ margin:0; padding:24px 16px; background:var(--bg); color:var(--ink); font:14px/1.5 system-ui, sans-serif; }}
main {{ max-width:1400px; margin:0 auto; display:grid; gap:14px; }}
section {{ background:var(--panel); border:1px solid var(--line); border-radius:10px; padding:12px 14px; }}
h1 {{ font-size:22px; margin:0; }} h2 {{ font-size:14px; margin:0; }} h3 {{ font-size:12px; margin:0 0 4px;
  text-transform:uppercase; letter-spacing:.05em; color:var(--muted); }}
.q {{ font-size:16px; margin:6px 0 2px; }} .meta, .rel {{ color:var(--muted); font-size:12px; }}
.cols {{ display:grid; grid-template-columns:repeat(auto-fit, minmax(280px, 1fr)); gap:12px; margin-top:8px; }}
.col.target h3 {{ color:var(--ink); }} ol {{ margin:0; padding-left:20px; }} li {{ margin-bottom:6px; }}
li.hit {{ background:var(--hl); outline:1px solid var(--ok); border-radius:4px; }}
.pass {{ color:var(--ok); }} .fail {{ color:var(--bad); }}
</style></head><body><main><h1>Retrieval report: {passed}/{len(queries)} found in the top {TOP}</h1>
<div class="meta">Gate: at least 80%. Each query is checked in its target collection (bold heading); highlighted rows
cover an expected element. ↳ lines are related items attached through links.</div>{''.join(sections)}</main></body></html>
""", encoding="utf-8")
    return out, passed, len(queries)
