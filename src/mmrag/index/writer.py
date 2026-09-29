"""Write one document to PostgreSQL in one transaction (LLD §3.8, spec §6.5, NFR-9).

All embeddings are computed before this is called, so the transaction holds no network
calls. Readers see the previous version until COMMIT, and a failure rolls everything back.
Replacing is delete-then-insert inside the transaction: element ids depend only on the PDF's
content, so a rebuilt document reuses them and could not coexist with its old rows.
"""

from __future__ import annotations

import hashlib
import json
from typing import Literal

from psycopg.types.json import Jsonb

from mmrag import db
from mmrag.config import Settings
from mmrag.index.chunk import Chunk
from mmrag.index.document import Link, LoadedDoc
from mmrag.obs import get_tracer


def build_hash(doc: LoadedDoc, chunks: list[Chunk], links: list[Link], embedding_model: str) -> str:
    """Everything that would be written; equal hashes mean re-ingesting writes nothing."""
    payload = {
        "content": doc.content_hash,
        "elements": [e.model_dump(mode="json") for e in doc.elements],
        "tables": [t.model_dump(mode="json") for t in doc.tables.values()],
        "captions": [c.model_dump(mode="json") for c in doc.captions.values()],
        "chunks": [c.model_dump() for c in chunks],
        "links": [lk.model_dump() for lk in links],
        "embedding_model": embedding_model,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def write_document(settings: Settings, doc: LoadedDoc, chunks: list[Chunk], vectors: list[list[float]],
                   links: list[Link]) -> Literal["written", "unchanged"]:
    if len(vectors) != len(chunks):
        raise ValueError(f"{len(chunks)} chunks but {len(vectors)} vectors")
    model = settings.embed.model
    bhash = build_hash(doc, chunks, links, model)
    with get_tracer("mmrag.index").start_as_current_span("index.write_document") as span, \
            db.connect(settings) as conn, conn.transaction():
        span.set_attribute("mmrag.doc_id", doc.doc_id)
        # Two runs for the same file never interleave.
        conn.execute("SELECT pg_advisory_xact_lock(hashtext(%s))", (doc.source_file,))
        current = conn.execute("SELECT doc_id, build_hash, version FROM documents WHERE source_file = %s",
                               (doc.source_file,)).fetchone()
        if current and current[0] == doc.doc_id and current[1] == bhash:
            span.set_attribute("mmrag.write.result", "unchanged")
            return "unchanged"

        conn.execute("DELETE FROM documents WHERE source_file = %s OR doc_id = %s", (doc.source_file, doc.doc_id))
        conn.execute("INSERT INTO documents (doc_id, source_file, content_hash, build_hash, version) "
                     "VALUES (%s, %s, %s, %s, %s)",
                     (doc.doc_id, doc.source_file, doc.content_hash, bhash, (current[2] + 1) if current else 1))
        with conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO elements (element_id, doc_id, page, bbox, type, section_path, text, caption, "
                "asset_path, content_hash, status, skip_reason) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                [(e.element_id, doc.doc_id, e.page, list(e.bbox), e.type, e.section_path, e.text, e.caption,
                  e.asset_path, e.content_hash, e.status, e.skip_reason) for e in doc.elements])
            known = {e.element_id for e in doc.elements}
            cur.executemany(
                "INSERT INTO figure_captions (element_id, figure_type, short_caption, detailed_description, "
                "visible_text, extracted_data, keywords, confidence, status, review_note, model_id, prompt_version) "
                "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                [(r.element_id, r.caption.figure_type, r.caption.short_caption, r.caption.detailed_description,
                  r.caption.visible_text,
                  Jsonb(r.caption.extracted_data.model_dump()) if r.caption.extracted_data else None,
                  r.caption.keywords, r.caption.confidence, r.status, r.error, r.model_id, r.prompt_version)
                 for r in doc.captions.values() if r.caption and r.element_id in known])
            cur.executemany(
                "INSERT INTO doc_tables (element_id, columns, rows, units, numeric_columns, title, summary, "
                "low_confidence) VALUES (%s,%s,%s,%s,%s,%s,%s,%s)",
                [(t.element_id, Jsonb(t.columns), Jsonb(t.rows), Jsonb(t.units), t.numeric_columns, t.title,
                  t.summary, t.low_confidence) for t in doc.tables.values() if t.element_id in known])
            cur.executemany(
                "INSERT INTO element_links (target_id, text_element_id, method, score) VALUES (%s,%s,%s,%s)",
                [(lk.target_id, lk.text_element_id, lk.method, lk.score) for lk in links])
            cur.executemany(
                "INSERT INTO search_chunks (chunk_id, doc_id, collection, element_ids, dense_text, keyword_text, "
                "embedding, embedding_model) VALUES (%s,%s,%s,%s,%s,%s,%s,%s)",
                [(c.chunk_id, doc.doc_id, c.collection, c.element_ids, c.dense_text, c.keyword_text,
                  "[" + ",".join(repr(float(x)) for x in v) + "]", model) for c, v in zip(chunks, vectors)])
        span.set_attribute("mmrag.write.result", "written")
        span.set_attribute("mmrag.write.chunks", len(chunks))
        return "written"
