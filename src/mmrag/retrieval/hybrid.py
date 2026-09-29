"""Hybrid search over one collection (spec §7.1, LLD §5.4).

One SQL statement: semantic top-N (HNSW, cosine) and keyword top-N (english OR simple
tsvector) fused with RRF, then each chunk's element locations aggregated into one array in
reading order, so every chunk is exactly one row. A second query attaches related items
from element_links (spec §7.1 step 6).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Callable, Literal

from psycopg import sql

from mmrag import db
from mmrag.config import Settings
from mmrag.obs import get_tracer

Collection = Literal["text", "figure", "table"]
COLLECTIONS = ("text", "figure", "table")


@dataclass
class Hit:
    chunk_id: str
    collection: str
    score: float
    semantic_rank: int | None
    keyword_rank: int | None
    dense_text: str
    source_file: str
    locations: list[dict]  # [{element_id, page, bbox}] in reading order
    related: list[dict] = field(default_factory=list)  # [{element_id, type, page, title, method}]


_SEARCH = """
WITH semantic AS (
    SELECT chunk_id, row_number() OVER (ORDER BY embedding <=> %(q)s::vector) AS r
    FROM search_chunks WHERE collection = {collection}
    ORDER BY embedding <=> %(q)s::vector LIMIT %(n)s
),
keyword AS (
    SELECT chunk_id, row_number() OVER (ORDER BY rank DESC, chunk_id) AS r FROM (
        SELECT chunk_id, greatest(ts_rank_cd(tsv_english, qe), ts_rank_cd(tsv_simple, qs)) AS rank
        FROM search_chunks, websearch_to_tsquery('english', %(t)s) qe, websearch_to_tsquery('simple', %(t)s) qs
        WHERE collection = {collection} AND (tsv_english @@ qe OR tsv_simple @@ qs)
        ORDER BY rank DESC LIMIT %(n)s
    ) ranked
),
fused AS (
    SELECT coalesce(s.chunk_id, k.chunk_id) AS chunk_id, s.r AS semantic_rank, k.r AS keyword_rank,
           coalesce(1.0 / (%(rrf)s + s.r), 0) + coalesce(1.0 / (%(rrf)s + k.r), 0) AS score
    FROM semantic s FULL OUTER JOIN keyword k ON s.chunk_id = k.chunk_id
    ORDER BY score DESC, chunk_id LIMIT %(k)s
)
SELECT f.chunk_id, c.collection, f.score, f.semantic_rank, f.keyword_rank, c.dense_text, d.source_file,
       loc.locations
FROM fused f
JOIN search_chunks c ON c.chunk_id = f.chunk_id
JOIN documents d ON d.doc_id = c.doc_id
CROSS JOIN LATERAL (
    SELECT jsonb_agg(jsonb_build_object('element_id', e.element_id, 'page', e.page, 'bbox', e.bbox)
                     ORDER BY u.ord) AS locations
    FROM unnest(c.element_ids) WITH ORDINALITY AS u(eid, ord)
    JOIN elements e ON e.element_id = u.eid
) loc
ORDER BY f.score DESC, f.chunk_id
"""

_RELATED = """
SELECT l.target_id, l.text_element_id, l.method, e.element_id, e.type, e.page,
       coalesce(fc.short_caption, dt.title, dt.summary, left(e.text, 80)) AS title
FROM element_links l
JOIN elements e ON e.element_id = CASE WHEN l.target_id = ANY(%(ids)s) THEN l.text_element_id ELSE l.target_id END
LEFT JOIN figure_captions fc ON fc.element_id = e.element_id
LEFT JOIN doc_tables dt ON dt.element_id = e.element_id
WHERE l.target_id = ANY(%(ids)s) OR l.text_element_id = ANY(%(ids)s)
ORDER BY l.score DESC, e.element_id
"""


def search(settings: Settings, query: str, collection: Collection, k: int | None = None,
           embed_query: Callable[[str], list[float]] | None = None) -> list[Hit]:
    """Top-k chunks of one collection for `query`, fused by RRF, with locations and related items."""
    if collection not in COLLECTIONS:
        raise ValueError(f"unknown collection {collection!r}")
    cfg = settings.search
    k = k or cfg.top_k
    if embed_query is None:
        embed_query = _default_embedder(settings)
    with get_tracer("mmrag.retrieval").start_as_current_span("retrieval.hybrid_search") as span:
        span.set_attribute("mmrag.collection", collection)
        span.set_attribute("mmrag.k", k)
        vector = "[" + ",".join(repr(float(x)) for x in embed_query(query)) + "]"
        statement = sql.SQL(_SEARCH).format(collection=sql.Literal(collection))  # literal: matches the partial index
        with db.connect(settings) as conn:
            conn.execute(sql.SQL("SET LOCAL hnsw.ef_search = {}").format(sql.Literal(cfg.hnsw_ef_search)))
            rows = conn.execute(statement, {"q": vector, "t": query, "n": cfg.candidates, "rrf": cfg.rrf_k,
                                            "k": k}).fetchall()
            hits = [Hit(chunk_id=r[0], collection=r[1], score=float(r[2]), semantic_rank=r[3], keyword_rank=r[4],
                        dense_text=r[5], source_file=r[6], locations=r[7] or []) for r in rows]
            _attach_related(conn, hits)
        span.set_attribute("mmrag.semantic_hits", sum(h.semantic_rank is not None for h in hits))
        span.set_attribute("mmrag.keyword_hits", sum(h.keyword_rank is not None for h in hits))
    return hits


def _attach_related(conn, hits: list[Hit]) -> None:
    ids = sorted({loc["element_id"] for h in hits for loc in h.locations})
    if not ids:
        return
    rows = conn.execute(_RELATED, {"ids": ids}).fetchall()
    for h in hits:
        mine = {loc["element_id"] for loc in h.locations}
        seen = set()
        for target, text_id, method, eid, etype, page, title in rows:
            if (target in mine or text_id in mine) and eid not in mine and eid not in seen:
                seen.add(eid)
                h.related.append({"element_id": eid, "type": etype, "page": page, "title": title, "method": method})


_embedders: dict[int, Callable[[str], list[float]]] = {}


def _default_embedder(settings: Settings) -> Callable[[str], list[float]]:
    """Query embeddings through the cached Embedder (spec §7.1: computed once per query text)."""
    key = id(settings)
    if key not in _embedders:
        from mmrag.index.embed import Embedder

        embedder = Embedder(settings)
        _embedders[key] = lru_cache(maxsize=512)(lambda q: embedder.embed([q])[0])
    return _embedders[key]
