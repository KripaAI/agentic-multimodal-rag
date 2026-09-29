"""Search and fetch tools against a real database (LLD §5.3). Written test-first."""

from __future__ import annotations

import pytest

import mmrag.db as db
from mmrag.agent.ledger import EvidenceLedger
from mmrag.agent.tools import ToolContext, get_figure, get_table, search_figures, search_text, view_page
from mmrag.config import PROJECT_ROOT, load_settings
from mmrag.index.chunk import Chunk
from mmrag.index.document import Link, LoadedDoc
from mmrag.index.writer import write_document
from mmrag.ingest.captions import CaptionRecord, ChartData, ExtractedData, FigureCaption
from mmrag.ingest.models import Element, Table

pytestmark = pytest.mark.integration
DOC = "e" * 16


def _vec(dims, i):
    v = [0.0] * dims
    v[i] = 1.0
    return v


@pytest.fixture
def ctx(fresh_db_url, tmp_path):
    s = load_settings(PROJECT_ROOT / "config.yaml", {"DATABASE_URL": fresh_db_url})
    db.migrate(s)

    def el(eid, etype, text=None, y=10):
        return Element(element_id=f"{DOC}:{eid}", doc_id=DOC, source_file="t.pdf", page=int(eid.split(":")[0][1:]),
                       bbox=(0, y, 100, y + 10), type=etype, text=text, content_hash=eid,
                       asset_path=f"assets/{DOC}/{eid}.png" if etype == "vector_figure" else None)

    els = [el("p1:text:1", "text", "GQA shrinks the KV cache from 33.5 GB to 6.7 GB."),
           el("p2:vector_figure:1", "vector_figure", y=40), el("p3:table:1", "table", "Heads | Memory")]
    cap = CaptionRecord(element_id=els[1].element_id, image_hash="h", status="ok", model_path="awq-7b", model_id="m",
                        prompt_version="v2", seconds=1.0, caption=FigureCaption(
                            figure_type="chart", short_caption="Next-token probabilities", detailed_description="bars",
                            visible_text=["41%"], confidence="high", keywords=[],
                            extracted_data=ExtractedData(chart=ChartData(chart_kind="bar", unit="%", series=[
                                {"name": "p", "points": [{"label": "There", "value": 41, "flag": "exact"},
                                                         {"label": "Sure", "value": 0.4, "flag": "estimated"}]}]))))
    table = Table(element_id=els[2].element_id, columns=["Heads", "Memory"], rows=[["40", "33.5 GB"], ["8", "6.7 GB"]],
                  title="KV cache", summary="Compares cache memory.")
    doc = LoadedDoc(doc_id=DOC, source_file="t.pdf", content_hash="x", elements=els,
                    tables={table.element_id: table}, captions={cap.element_id: cap})
    dims = s.embed.dims
    chunks = [Chunk(chunk_id=f"{DOC}:text:1", collection="text", element_ids=[els[0].element_id],
                    dense_text="GQA shrinks the KV cache from 33.5 GB to 6.7 GB.", keyword_text="GQA KV cache"),
              Chunk(chunk_id=f"{DOC}:figure:1", collection="figure", element_ids=[els[1].element_id],
                    dense_text="Next-token probabilities bar chart", keyword_text="probabilities")]
    write_document(s, doc, chunks, [_vec(dims, 0), _vec(dims, 1)],
                   [Link(target_id=els[1].element_id, text_element_id=els[0].element_id, method="related", score=0.5)])
    return ToolContext(settings=s, ledger=EvidenceLedger(), embed_query=lambda q: _vec(dims, 0), chart_dir=tmp_path)


def test_search_text_returns_citable_ids_and_records_evidence(ctx):
    (hit,) = search_text(ctx, "GQA", k=3)[:1]
    assert hit["id"] == f"{DOC}:text:1" and hit["pages"] == [1] and "33.5" in hit["text"]
    assert hit["related"][0]["id"] == f"{DOC}:p2:vector_figure:1"
    assert ctx.ledger.find_number(33.5, refs=[hit["id"]]) is not None
    assert ctx.ledger.has(f"{DOC}:p2:vector_figure:1")  # related items can be cited too


def test_search_figures_cites_the_figure_element(ctx):
    hits = search_figures(ctx, "probabilities", k=3)
    assert hits[0]["id"] == f"{DOC}:p2:vector_figure:1" and hits[0]["page"] == 2


def test_get_figure_keeps_exact_and_estimated_values_apart(ctx):
    fig = get_figure(ctx, f"{DOC}:p2:vector_figure:1")
    assert fig["short_caption"] == "Next-token probabilities"
    ev = ctx.ledger.items[f"{DOC}:p2:vector_figure:1"]
    assert 41 in ev.numbers and 0.4 in ev.estimated


def test_get_table_returns_every_row(ctx):
    t = get_table(ctx, f"{DOC}:p3:table:1")
    assert t["columns"] == ["Heads", "Memory"] and t["rows"][1] == ["8", "6.7 GB"]
    assert ctx.ledger.find_number(6.7, refs=[f"{DOC}:p3:table:1"]) is not None


def test_unknown_ids_are_errors_for_the_model(ctx):
    with pytest.raises(ValueError, match="unknown"):
        get_table(ctx, f"{DOC}:p9:table:9")


def test_view_page_only_for_indexed_files(ctx):
    with pytest.raises(ValueError, match="not an indexed document"):
        view_page(ctx, "../../secrets.pdf", 1)
