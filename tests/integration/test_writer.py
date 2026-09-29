"""Transactional document writes (LLD §3.8, NFR-9) against a real throwaway database. Test-first."""

from __future__ import annotations

import pytest

import mmrag.db as db
from mmrag.config import PROJECT_ROOT, load_settings
from mmrag.index.chunk import Chunk
from mmrag.index.document import Link, LoadedDoc
from mmrag.index.writer import write_document
from mmrag.ingest.models import Element, Table

pytestmark = pytest.mark.integration


@pytest.fixture
def settings(fresh_db_url):
    s = load_settings(PROJECT_ROOT / "config.yaml", {"DATABASE_URL": fresh_db_url})
    db.migrate(s)
    return s


def _doc(doc_id="a" * 16, source="x.pdf", text="LoRA trains small adapters."):
    els = [
        Element(element_id=f"{doc_id}:p1:text:1", doc_id=doc_id, source_file=source, page=1, bbox=(0, 0, 1, 1),
                type="text", text=text, content_hash="t1", section_path=["S"]),
        Element(element_id=f"{doc_id}:p1:table:1", doc_id=doc_id, source_file=source, page=1, bbox=(0, 2, 1, 3),
                type="table", text="A | B", content_hash="t2"),
    ]
    table = Table(element_id=els[1].element_id, columns=["A", "B"], rows=[["1", "2"]], summary="Two numbers.")
    return LoadedDoc(doc_id=doc_id, source_file=source, content_hash=doc_id * 4, elements=els,
                     tables={table.element_id: table})


def _chunks(doc):
    return [
        Chunk(chunk_id=f"{doc.doc_id}:text:1", collection="text", element_ids=[doc.elements[0].element_id],
              dense_text=doc.elements[0].text, keyword_text=doc.elements[0].text),
        Chunk(chunk_id=f"{doc.doc_id}:table:1", collection="table", element_ids=[doc.elements[1].element_id],
              dense_text="Two numbers.", keyword_text="A B 1 2"),
    ]


def _vectors(settings, n):
    return [[0.1] * settings.embed.dims for _ in range(n)]


def _links(doc):
    return [Link(target_id=doc.elements[1].element_id, text_element_id=doc.elements[0].element_id,
                 method="related", score=0.7)]


def _counts(settings):
    with db.connect(settings) as conn:
        return {t: conn.execute(f"SELECT count(*) FROM {t}").fetchone()[0]
                for t in ("documents", "elements", "doc_tables", "element_links", "search_chunks")}


def test_write_then_unchanged_rewrite_is_a_no_op(settings):
    doc = _doc()
    assert write_document(settings, doc, _chunks(doc), _vectors(settings, 2), _links(doc)) == "written"
    assert _counts(settings) == {"documents": 1, "elements": 2, "doc_tables": 1, "element_links": 1,
                                 "search_chunks": 2}
    with db.connect(settings) as conn:
        before = conn.execute("SELECT ingested_at, version FROM documents").fetchone()
    assert write_document(settings, doc, _chunks(doc), _vectors(settings, 2), _links(doc)) == "unchanged"
    with db.connect(settings) as conn:
        assert conn.execute("SELECT ingested_at, version FROM documents").fetchone() == before


def test_changed_build_replaces_the_rows(settings):
    doc = _doc()
    write_document(settings, doc, _chunks(doc), _vectors(settings, 2), _links(doc))
    changed = _doc(text="LoRA trains low-rank adapters.")
    assert write_document(settings, changed, _chunks(changed), _vectors(settings, 2), _links(changed)) == "written"
    assert _counts(settings)["search_chunks"] == 2  # replaced, not duplicated
    with db.connect(settings) as conn:
        assert conn.execute("SELECT version FROM documents").fetchone()[0] == 2
        assert conn.execute("SELECT dense_text FROM search_chunks WHERE collection = 'text'").fetchone()[0] \
            == "LoRA trains low-rank adapters."


def test_new_pdf_version_replaces_the_old_document(settings):
    old = _doc(doc_id="a" * 16)
    write_document(settings, old, _chunks(old), _vectors(settings, 2), _links(old))
    new = _doc(doc_id="b" * 16)  # same file name, new content
    write_document(settings, new, _chunks(new), _vectors(settings, 2), _links(new))
    with db.connect(settings) as conn:
        assert [r[0] for r in conn.execute("SELECT doc_id FROM documents")] == ["b" * 16]
        assert conn.execute("SELECT count(*) FROM elements WHERE doc_id = %s", ("a" * 16,)).fetchone()[0] == 0


def test_failure_mid_write_keeps_the_previous_version(settings):
    doc = _doc()
    write_document(settings, doc, _chunks(doc), _vectors(settings, 2), _links(doc))
    before = _counts(settings)
    changed = _doc(text="New text.")
    broken = _chunks(changed)
    broken[1] = broken[1].model_copy(update={"element_ids": []})  # violates nothing yet...
    bad_vectors = _vectors(settings, 1) + [[0.1] * 3]  # ...but a wrong-size vector fails mid-insert
    with pytest.raises(Exception):
        write_document(settings, changed, broken, bad_vectors, _links(changed))
    assert _counts(settings) == before
    with db.connect(settings) as conn:
        assert conn.execute("SELECT dense_text FROM search_chunks WHERE collection = 'text'").fetchone()[0] \
            == "LoRA trains small adapters."
