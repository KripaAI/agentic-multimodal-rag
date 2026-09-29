"""Phase 3 end to end on a real fixture page: parse → index (fake embedder and LLM) → search."""

from __future__ import annotations

import hashlib
import shutil

import pytest

import mmrag.db as db
from mmrag.config import PROJECT_ROOT, load_settings
from mmrag.index.pipeline import index_document
from mmrag.ingest.parse import parse_document, write_outputs
from mmrag.retrieval.hybrid import search

pytestmark = pytest.mark.integration


class FakeEmbedder:
    def __init__(self, dims):
        self.dims, self.calls = dims, 0

    def _vec(self, text):
        h = hashlib.sha256(text.encode("utf-8")).digest()
        return [(h[i % 32] - 128) / 128 for i in range(self.dims)]

    def embed(self, texts):
        self.calls += 1
        return [self._vec(t) for t in texts]


@pytest.fixture
def settings(fresh_db_url, base_config, write_config, tmp_path):
    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()
    shutil.copy(PROJECT_ROOT / "tests" / "fixtures" / "pages" / "transformers_p003.pdf", pdf_dir / "p3.pdf")
    base_config["paths"].update(pdf_dir=str(pdf_dir), data_dir=str(tmp_path / "data"))
    base_config["observability"].update(enabled=False, log_file=None)
    s = load_settings(write_config(base_config), {"DATABASE_URL": fresh_db_url})
    db.migrate(s)
    return s


def test_index_then_reindex_then_search(settings):
    pdf = settings.resolve(settings.paths.pdf_dir) / "p3.pdf"
    write_outputs(parse_document(pdf, settings), pdf, settings)
    embedder = FakeEmbedder(settings.embed.dims)

    report = index_document(pdf, settings, embedder=embedder, chat=lambda model, prompt: "A summary.")
    assert report.result == "written"
    assert report.counts["figure"] == 2 and report.counts["text"] >= 1
    assert report.link_report.is_file()
    with db.connect(settings) as conn:
        assert conn.execute("SELECT count(*) FROM search_chunks").fetchone()[0] == len(report.chunks)
        assert conn.execute("SELECT count(*) FROM elements").fetchone()[0] == len(report.doc.elements)

    again = index_document(pdf, settings, embedder=embedder, chat=lambda model, prompt: "A summary.")
    assert again.result == "unchanged"

    query = report.chunks[0].dense_text  # a query equal to a chunk's text lands on it (keyword and vector)
    hits = search(settings, query, "text", 3, embed_query=lambda q: embedder._vec(q))
    assert hits[0].chunk_id == report.chunks[0].chunk_id
    assert all(loc["page"] == 1 for loc in hits[0].locations)
