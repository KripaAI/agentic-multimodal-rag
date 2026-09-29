"""Search documents: text chunks, figure docs, table docs (LLD §3.6, spec §6.4). Written test-first."""

from __future__ import annotations

import pytest

from mmrag.index.chunk import build_figure_docs, build_table_docs, chunk_text, count_tokens
from mmrag.index.document import LoadedDoc, Link
from mmrag.ingest.captions import CaptionRecord, FigureCaption
from mmrag.ingest.models import Element, Table

pytestmark = pytest.mark.unit

DOC = "d" * 16


def _text(n, section, words, page=1):
    return Element(element_id=f"{DOC}:p{page}:text:{n}", doc_id=DOC, source_file="x.pdf", page=page,
                   bbox=(50, 10 * n, 500, 10 * n + 8), type="text", section_path=section,
                   text=" ".join(f"w{n}_{i}" for i in range(words)), content_hash=f"h{n}")


def _doc(elements, tables=(), captions=()):
    return LoadedDoc(doc_id=DOC, source_file="x.pdf", content_hash="c", elements=list(elements),
                     tables={t.element_id: t for t in tables}, captions={c.element_id: c for c in captions})


@pytest.fixture
def cfg(parse_settings):
    return parse_settings.chunk


def test_chunks_never_cross_a_section(cfg):
    a, b = ["Part 1", "Loop"], ["Part 1", "Sampling"]
    doc = _doc([_text(1, a, 60), _text(2, a, 60), _text(3, b, 60)])
    chunks = chunk_text(doc, cfg)
    assert [c.element_ids for c in chunks] == [[f"{DOC}:p1:text:1", f"{DOC}:p1:text:2"], [f"{DOC}:p1:text:3"]]
    assert chunks[0].dense_text.startswith("Part 1 > Loop\n")
    assert chunks[1].dense_text.startswith("Part 1 > Sampling\n")
    assert [c.chunk_id for c in chunks] == [f"{DOC}:text:1", f"{DOC}:text:2"]


def test_chunks_respect_max_tokens_with_overlap(cfg):
    sec = ["S"]
    blocks = [_text(n, sec, 25) for n in range(1, 13)]  # 12 paragraphs of ~100 tokens
    chunks = chunk_text(_doc(blocks), cfg)
    assert len(chunks) > 1
    for c in chunks:
        assert count_tokens(c.dense_text) <= cfg.max_tokens + 20  # + the section label
    # Overlap: each chunk after the first starts with the last block of the previous one.
    for prev, nxt in zip(chunks, chunks[1:]):
        assert nxt.element_ids[0] == prev.element_ids[-1]
    covered = {e for c in chunks for e in c.element_ids}
    assert covered == {b.element_id for b in blocks}  # nothing is lost


def test_short_chunks_only_at_a_section_end(cfg):
    sec = ["S"]
    blocks = [_text(n, sec, 25) for n in range(1, 20)]
    chunks = chunk_text(_doc(blocks), cfg)
    assert len(chunks) >= 2
    assert all(count_tokens(c.dense_text) >= cfg.min_tokens for c in chunks[:-1])


def test_a_block_longer_than_max_is_split(cfg):
    big = _text(1, ["S"], 2000)
    chunks = chunk_text(_doc([big]), cfg)
    assert len(chunks) >= 3
    assert all(c.element_ids == [big.element_id] for c in chunks)
    assert all(count_tokens(c.dense_text) <= cfg.max_tokens + 20 for c in chunks)


def _figure(n=1, caption=None, text="Input\nOutput"):
    return Element(element_id=f"{DOC}:p2:vector_figure:{n}", doc_id=DOC, source_file="x.pdf", page=2,
                   bbox=(50, 300, 500, 400), type="vector_figure", section_path=["Part 1"], text=text,
                   caption=caption, asset_path="a.png", content_hash=f"fig{n}")


def _caption(eid):
    return CaptionRecord(element_id=eid, image_hash="x", status="ok", model_path="awq-7b", model_id="m",
                         prompt_version="v2", seconds=1.0, caption=FigureCaption(
                             figure_type="flowchart", short_caption="The four-step loop",
                             detailed_description="Input Tokens → Compute Probabilities → Select Token",
                             visible_text=["Input Tokens", "Select Token"], keywords=["autoregressive"],
                             confidence="high"))


def test_figure_doc_combines_caption_labels_and_linked_paragraph():
    fig = _figure(caption="The whole of generation is these four steps.")
    para = _text(5, ["Part 1"], 5, page=2)
    doc = _doc([para, fig], captions=[_caption(fig.element_id)])
    (chunk,) = build_figure_docs(doc, [Link(target_id=fig.element_id, text_element_id=para.element_id,
                                            method="related", score=0.6)])
    for part in ("The four-step loop", "Compute Probabilities", "Select Token", "autoregressive",
                 "The whole of generation", para.text):
        assert part in chunk.dense_text
    assert chunk.dense_text == chunk.keyword_text
    assert chunk.collection == "figure" and chunk.element_ids == [fig.element_id]


def test_figure_without_caption_is_still_searchable():
    fig = _figure(caption="Figure 3. Token flow.")
    (chunk,) = build_figure_docs(_doc([fig]), [])
    assert "Figure 3. Token flow." in chunk.dense_text and "Input" in chunk.dense_text


def test_table_doc_splits_meaning_and_keywords():
    t_el = Element(element_id=f"{DOC}:p4:table:1", doc_id=DOC, source_file="x.pdf", page=4, bbox=(0, 0, 1, 1),
                   type="table", text="Method | Score", content_hash="t1")
    table = Table(element_id=t_el.element_id, columns=["Method", "Score"], rows=[["LoRA", "71.2"], ["DPO", "GSM8K"]],
                  title="Results", summary="Compares LoRA and DPO scores.")
    (chunk,) = build_table_docs(_doc([t_el], tables=[table]))
    assert "Compares LoRA and DPO scores." in chunk.dense_text and "Method" in chunk.dense_text
    assert "71.2" not in chunk.dense_text  # raw numbers embed poorly (spec §6.4)
    assert all(v in chunk.keyword_text for v in ("Results", "Method", "LoRA", "71.2", "GSM8K"))
    assert chunk.collection == "table"
