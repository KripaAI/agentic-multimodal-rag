"""Answer validator (spec §7.3, LLD §5.6): the constitution, enforced by code.

Rules: every block cites ids that a tool returned in this question and that exist in the
corpus (P2); images are original figures (P3); charts come from the chart engine, whose values
were checked against the evidence (P4); no citation points to a memory (P13); a computed
number cites the sources of its inputs; a "not found" answer makes no claims (P5).
Citations are hydrated with source_file, page and bbox from PostgreSQL; the model never
supplies locations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from mmrag.agent.answer import (
    Answer, ChartBlock, HydratedAnswer, HydratedBlock, HydratedCitation, ImageBlock, Location, Source, TableBlock,
    TextBlock,
)
from mmrag.agent.ledger import EvidenceLedger
from mmrag.charts.engine import ChartResult

FIGURE_TYPES = {"vector_figure", "image", "scanned_page"}


class Store(Protocol):
    def locations(self, ids: list[str]) -> dict[str, list[dict]]: ...
    def element_info(self, ids: list[str]) -> dict[str, dict]: ...


@dataclass
class ValidationResult:
    ok: bool
    answer: HydratedAnswer | None = None
    failures: list[str] = field(default_factory=list)  # one line per broken rule, for the repair turn
    failed_blocks: set[int] = field(default_factory=set)


def _cited_ids(block) -> list[str]:
    if isinstance(block, TextBlock):
        return [c.id for c in block.citations]
    if isinstance(block, (ImageBlock, TableBlock)):
        return [block.citation.id]
    return []


def validate(answer: Answer, ledger: EvidenceLedger, charts: dict[str, ChartResult], store: Store) -> ValidationResult:
    """Check every rule and hydrate citations. `charts` holds the engine results by chart_id."""
    failures: list[tuple[int, str]] = []
    ids = {i for b in answer.blocks for i in _cited_ids(b)}
    ids |= {b.element_id for b in answer.blocks if isinstance(b, (ImageBlock, TableBlock))}
    ids |= {c for b in answer.blocks if isinstance(b, ChartBlock) and b.chart_id in charts
            for c in charts[b.chart_id].citations}
    locations = store.locations(sorted(i for i in ids if not i.startswith(("memory:", "compute:"))))
    info = store.element_info(sorted(ids))

    def check_id(n: int, cid: str) -> None:
        if cid.startswith("memory:"):
            failures.append((n, f"block {n}: cites {cid}; memories are never evidence, cite the documents"))
        elif cid.startswith("compute:"):
            failures.append((n, f"block {n}: cites {cid}; cite the sources of the computed numbers instead"))
        elif not ledger.has(cid):
            failures.append((n, f"block {n}: cites {cid}, which no tool returned in this question"))
        elif cid not in locations:
            failures.append((n, f"block {n}: cites {cid}, an unknown id"))

    for n, block in enumerate(answer.blocks):
        if isinstance(block, TextBlock):
            if not block.citations and not answer.not_found:
                failures.append((n, f"block {n}: text without a citation; every claim needs a source"))
        elif answer.not_found:
            failures.append((n, f"block {n}: a not-found answer must not contain {block.type} blocks"))
        if isinstance(block, ImageBlock):
            if info.get(block.element_id, {}).get("type") not in FIGURE_TYPES:
                failures.append((n, f"block {n}: {block.element_id} is not an original figure from the documents"))
        elif isinstance(block, TableBlock):
            if info.get(block.element_id, {}).get("type") != "table":
                failures.append((n, f"block {n}: {block.element_id} is not a table from the documents"))
        elif isinstance(block, ChartBlock):
            if block.chart_id not in charts:
                failures.append((n, f"block {n}: {block.chart_id} was not made by make_chart in this question"))
            else:
                for cid in charts[block.chart_id].citations:
                    if not cid.startswith("compute:"):
                        check_id(n, cid)
        for cid in _cited_ids(block):
            check_id(n, cid)

    if failures:
        return ValidationResult(ok=False, failures=[m for _, m in failures], failed_blocks={n for n, _ in failures})

    def hydrate(cid: str) -> HydratedCitation:
        return HydratedCitation(id=cid, locations=[Location(**loc) for loc in locations[cid]])

    blocks, sources, seen = [], [], set()
    for block in answer.blocks:
        if isinstance(block, TextBlock):
            hb = HydratedBlock(type="text", content={"markdown": block.markdown},
                               citations=[hydrate(c.id) for c in block.citations])
        elif isinstance(block, ImageBlock):
            e = info[block.element_id]
            hb = HydratedBlock(type="image", content={"element_id": block.element_id, "asset_path": e["asset_path"],
                                                      "short_caption": block.short_caption},
                               citations=[hydrate(block.citation.id)])
        elif isinstance(block, TableBlock):
            e = info[block.element_id]
            hb = HydratedBlock(type="table", content={"element_id": block.element_id, "title": e.get("title"),
                                                      "columns": e["columns"], "rows": e["rows"]},
                               citations=[hydrate(block.citation.id)])
        else:
            c = charts[block.chart_id]
            hb = HydratedBlock(type="chart", approximate=c.approximate,
                               content={"chart_id": c.chart_id, "chart_type": c.chart_type, "title": c.title,
                                        "spec": c.spec, "png_path": str(c.png_path) if c.png_path else None,
                                        "data_table": c.data_table, "notes": c.notes},
                               citations=[hydrate(i) for i in c.citations if not i.startswith("compute:")])
        blocks.append(hb)
        for hc in hb.citations:
            for loc in hc.locations:
                key = (loc.source_file, loc.page, loc.element_id)
                if key not in seen:
                    seen.add(key)
                    sources.append(Source(source_file=loc.source_file, page=loc.page, element_id=loc.element_id))
    return ValidationResult(ok=True, answer=HydratedAnswer(blocks=blocks, sources=sources))


def drop_failing_blocks(answer: Answer, failed_blocks: set[int]) -> tuple[Answer, list[str]]:
    """After a failed repair: remove the blocks that still break a rule; return notices saying so."""
    kept = [b for n, b in enumerate(answer.blocks) if n not in failed_blocks]
    notices = [f"{len(failed_blocks)} part(s) of the answer were removed because their sources could not be verified."]
    if not kept:
        kept = [TextBlock(markdown="No part of the answer could be verified against the documents.", citations=[])]
        return answer.model_copy(update={"blocks": kept, "not_found": True}), notices
    return answer.model_copy(update={"blocks": kept}), notices


class DbStore:
    """The production Store: element locations and details from PostgreSQL."""

    def __init__(self, settings):
        self.settings = settings

    def locations(self, ids: list[str]) -> dict[str, list[dict]]:
        from mmrag import db

        out: dict[str, list[dict]] = {}
        pages = [i for i in ids if i.startswith("page:")]
        with db.connect(self.settings) as conn:
            for eid, src, page, bbox in conn.execute(
                    "SELECT e.element_id, d.source_file, e.page, e.bbox FROM elements e JOIN documents d USING (doc_id) "
                    "WHERE e.element_id = ANY(%s)", (ids,)):
                out[eid] = [{"element_id": eid, "source_file": src, "page": page, "bbox": tuple(bbox)}]
            for cid, eid, src, page, bbox in conn.execute(
                    "SELECT c.chunk_id, e.element_id, d.source_file, e.page, e.bbox FROM search_chunks c "
                    "JOIN documents d USING (doc_id) CROSS JOIN LATERAL unnest(c.element_ids) WITH ORDINALITY u(eid, ord) "
                    "JOIN elements e ON e.element_id = u.eid WHERE c.chunk_id = ANY(%s) ORDER BY c.chunk_id, u.ord",
                    (ids,)):
                out.setdefault(cid, []).append({"element_id": eid, "source_file": src, "page": page,
                                                "bbox": tuple(bbox)})
            for pid in pages:
                _, src, page = pid.split(":", 2)
                if conn.execute("SELECT 1 FROM documents WHERE source_file = %s", (src,)).fetchone():
                    out[pid] = [{"element_id": pid, "source_file": src, "page": int(page), "bbox": (0, 0, 0, 0)}]
        return out

    def element_info(self, ids: list[str]) -> dict[str, dict]:
        from mmrag import db

        with db.connect(self.settings) as conn:
            rows = conn.execute(
                "SELECT e.element_id, e.type, e.asset_path, e.page, d.source_file, fc.short_caption, "
                "t.columns, t.rows, t.title FROM elements e JOIN documents d USING (doc_id) "
                "LEFT JOIN figure_captions fc USING (element_id) LEFT JOIN doc_tables t USING (element_id) "
                "WHERE e.element_id = ANY(%s)", (ids,)).fetchall()
        return {r[0]: {"type": r[1], "asset_path": r[2], "page": r[3], "source_file": r[4], "short_caption": r[5],
                       "columns": r[6], "rows": r[7], "title": r[8]} for r in rows}
