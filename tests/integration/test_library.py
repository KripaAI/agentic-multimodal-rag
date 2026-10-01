"""Managing the PDF library one document at a time (plan Phase 8 task 1). Test-first."""

from __future__ import annotations

import pytest

import mmrag.db as db
from mmrag.config import load_settings
from mmrag.index.chunk import Chunk
from mmrag.index.document import LoadedDoc
from mmrag.index.library import add_pdf, list_documents, remove_document, replace_pdf
from mmrag.index.writer import write_document
from mmrag.ingest.models import Element

pytestmark = pytest.mark.integration


@pytest.fixture
def settings(fresh_db_url, base_config, write_config, tmp_path):
    base_config["paths"]["data_dir"] = str(tmp_path / "data")
    base_config["paths"]["pdf_dir"] = str(tmp_path / "data" / "pdfs")
    base_config["observability"].update(enabled=False, log_file=None)
    s = load_settings(write_config(base_config), {"DATABASE_URL": fresh_db_url})
    db.migrate(s)
    (tmp_path / "data" / "pdfs").mkdir(parents=True)
    return s


def _index(settings, doc_id, name):
    el = Element(element_id=f"{doc_id}:p1:text:1", doc_id=doc_id, source_file=name, page=1, bbox=(0, 0, 10, 10),
                 type="text", text="some text", content_hash="t")
    chunk = Chunk(chunk_id=f"{doc_id}:text:1", collection="text", element_ids=[el.element_id], dense_text="some text",
                  keyword_text="some text")
    write_document(settings, LoadedDoc(doc_id=doc_id, source_file=name, content_hash=doc_id, elements=[el]), [chunk],
                   [[1.0] + [0.0] * (settings.embed.dims - 1)], [])


def test_list_shows_each_document_with_its_counts(settings):
    _index(settings, "a" * 16, "a.pdf")
    _index(settings, "b" * 16, "b.pdf")
    docs = list_documents(settings)
    assert [(d.source_file, d.version, d.text) for d in docs] == [("a.pdf", 1, 1), ("b.pdf", 1, 1)]


def test_remove_deletes_one_document_and_leaves_the_rest(settings, tmp_path):
    (tmp_path / "data" / "pdfs" / "a.pdf").write_bytes(b"%PDF a")
    _index(settings, "a" * 16, "a.pdf")
    _index(settings, "b" * 16, "b.pdf")
    remove_document(settings, "a.pdf")
    with db.connect(settings) as conn:
        left = conn.execute("SELECT DISTINCT doc_id FROM search_chunks").fetchall()
        elements = conn.execute("SELECT count(*) FROM elements WHERE doc_id = %s", ("a" * 16,)).fetchone()[0]
    assert left == [("b" * 16,)] and elements == 0  # cascade: nothing of a.pdf remains searchable
    assert not (tmp_path / "data" / "pdfs" / "a.pdf").exists()
    assert (tmp_path / "data" / "pdfs" / "removed" / "a.pdf").exists()  # kept, out of the library
    with pytest.raises(ValueError, match="not in the library"):
        remove_document(settings, "a.pdf")


def test_add_copies_into_the_library_and_refuses_a_duplicate_name(settings, tmp_path):
    src = tmp_path / "new.pdf"
    src.write_bytes(b"%PDF new")
    dest = add_pdf(settings, src)
    assert dest == tmp_path / "data" / "pdfs" / "new.pdf" and dest.read_bytes() == b"%PDF new"
    with pytest.raises(FileExistsError, match="replace"):
        add_pdf(settings, src)


def test_replace_archives_the_old_file_and_keeps_the_name(settings, tmp_path):
    lib = tmp_path / "data" / "pdfs"
    (lib / "a.pdf").write_bytes(b"%PDF old")
    newer = tmp_path / "a-v2.pdf"
    newer.write_bytes(b"%PDF new")
    dest = replace_pdf(settings, "a.pdf", newer)
    assert dest == lib / "a.pdf" and dest.read_bytes() == b"%PDF new"
    archived = list((lib / "replaced").glob("a.*.pdf"))
    assert len(archived) == 1 and archived[0].read_bytes() == b"%PDF old"
    with pytest.raises(FileNotFoundError):
        replace_pdf(settings, "missing.pdf", newer)
