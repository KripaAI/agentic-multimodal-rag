"""Hybrid search: semantic + keyword + RRF, one row per chunk, related items (spec §7.1, LLD §5.4).
Against a real throwaway database, with hand-made vectors. Written test-first."""

from __future__ import annotations

import pytest

import mmrag.db as db
from mmrag.config import PROJECT_ROOT, load_settings
from mmrag.index.chunk import Chunk
from mmrag.index.document import Link, LoadedDoc
from mmrag.index.writer import write_document
from mmrag.ingest.captions import CaptionRecord, FigureCaption
from mmrag.ingest.models import Element
from mmrag.retrieval.hybrid import search

pytestmark = pytest.mark.integration

DOC = "c" * 16


def _vec(dims, i):
    v = [0.0] * dims
    v[i] = 1.0
    return v


@pytest.fixture
def settings(fresh_db_url):
    s = load_settings(PROJECT_ROOT / "config.yaml", {"DATABASE_URL": fresh_db_url})
    db.migrate(s)
    return s


@pytest.fixture
def indexed(settings):
    def el(eid, page, y, etype="text", text=None):
        return Element(element_id=f"{DOC}:{eid}", doc_id=DOC, source_file="t.pdf", page=page,
                       bbox=(0, y, 100, y + 10), type=etype, text=text, content_hash=eid)

    els = [el("p1:text:1", 1, 10, text="LoRA adapters"), el("p1:text:2", 1, 30, text="more"),
           el("p2:text:1", 2, 10, text="GSM8K benchmark"), el("p2:text:2", 2, 30, text="sampling"),
           el("p3:vector_figure:1", 3, 10, "vector_figure")]
    cap = CaptionRecord(element_id=f"{DOC}:p3:vector_figure:1", image_hash="h", status="ok", model_path="awq-7b",
                        model_id="m", prompt_version="v2", seconds=1.0, caption=FigureCaption(
                            figure_type="diagram", short_caption="The four-step loop", detailed_description="d",
                            visible_text=[], keywords=[], confidence="high"))
    doc = LoadedDoc(doc_id=DOC, source_file="t.pdf", content_hash="x", elements=els,
                    captions={cap.element_id: cap})
    chunks = [
        # element_ids deliberately out of page order in the list: locations must follow this order
        Chunk(chunk_id=f"{DOC}:text:1", collection="text", element_ids=[f"{DOC}:p1:text:2", f"{DOC}:p1:text:1"],
              dense_text="low-rank adaptation fine-tuning", keyword_text="LoRA fine-tuning adapters"),
        Chunk(chunk_id=f"{DOC}:text:2", collection="text", element_ids=[f"{DOC}:p2:text:1"],
              dense_text="math word problems benchmark", keyword_text="GSM8K benchmark results"),
        Chunk(chunk_id=f"{DOC}:text:3", collection="text", element_ids=[f"{DOC}:p2:text:2"],
              dense_text="temperature sampling", keyword_text="temperature sampling top-p"),
        Chunk(chunk_id=f"{DOC}:figure:1", collection="figure", element_ids=[f"{DOC}:p3:vector_figure:1"],
              dense_text="four-step generation loop", keyword_text="four-step generation loop"),
    ]
    dims = settings.embed.dims
    write_document(settings, doc, chunks, [_vec(dims, 0), _vec(dims, 1), _vec(dims, 2), _vec(dims, 3)],
                   [Link(target_id=f"{DOC}:p3:vector_figure:1", text_element_id=f"{DOC}:p2:text:2",
                         method="related", score=0.6)])
    return settings


def _embed_near(settings, i):
    return lambda q: _vec(settings.embed.dims, i)


def test_exact_term_found_by_keyword(indexed):
    hits = search(indexed, "GSM8K", "text", 3, embed_query=_embed_near(indexed, 2))  # vector points elsewhere
    assert hits[0].chunk_id == f"{DOC}:text:2"
    assert hits[0].keyword_rank == 1


def test_meaning_found_without_shared_words(indexed):
    hits = search(indexed, "adapting big models cheaply", "text", 3, embed_query=_embed_near(indexed, 0))
    assert hits[0].chunk_id == f"{DOC}:text:1" and hits[0].keyword_rank is None


def test_found_by_both_ranks_first(indexed):
    hits = search(indexed, "temperature", "text", 3, embed_query=_embed_near(indexed, 2))
    assert hits[0].chunk_id == f"{DOC}:text:3"
    assert hits[0].semantic_rank == 1 and hits[0].keyword_rank == 1
    assert hits[0].score == pytest.approx(2 / (indexed.search.rrf_k + 1))


def test_each_chunk_once_with_locations_in_order(indexed):
    hits = search(indexed, "LoRA", "text", 8, embed_query=_embed_near(indexed, 0))
    ids = [h.chunk_id for h in hits]
    assert len(ids) == len(set(ids))
    first = next(h for h in hits if h.chunk_id == f"{DOC}:text:1")
    assert [loc["element_id"] for loc in first.locations] == [f"{DOC}:p1:text:2", f"{DOC}:p1:text:1"]
    assert first.locations[0]["page"] == 1 and len(first.locations[0]["bbox"]) == 4
    assert first.source_file == "t.pdf"


def test_collections_are_separate(indexed):
    hits = search(indexed, "loop", "figure", 5, embed_query=_embed_near(indexed, 3))
    assert [h.collection for h in hits] == ["figure"]


def test_related_items_come_along_both_ways(indexed):
    text_hit = search(indexed, "temperature", "text", 1, embed_query=_embed_near(indexed, 2))[0]
    assert [(r["element_id"], r["type"], r["title"]) for r in text_hit.related] == \
        [(f"{DOC}:p3:vector_figure:1", "vector_figure", "The four-step loop")]
    fig_hit = search(indexed, "loop", "figure", 1, embed_query=_embed_near(indexed, 3))[0]
    assert [(r["element_id"], r["type"]) for r in fig_hit.related] == [(f"{DOC}:p2:text:2", "text")]
