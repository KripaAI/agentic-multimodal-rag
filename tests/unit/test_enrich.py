"""Enrichment: label cross-check, figure/table links, table summaries (LLD §3.5, spec §6.3).
Written test-first. Embeddings and the LLM are replaced by deterministic fakes."""

from __future__ import annotations

import math

import pytest

from mmrag.index.document import LoadedDoc
from mmrag.ingest.captions import CaptionRecord, FigureCaption
from mmrag.ingest.enrich import cross_check_labels, find_links, summarize_tables
from mmrag.ingest.models import Element, Table

pytestmark = pytest.mark.unit

DOC = "d" * 16
TOPICS = ["draft", "quantiz", "loop", "memory", "token"]


def fake_embed(texts):
    """Topic-count vectors: texts about the same things point the same way."""
    out = []
    for t in texts:
        v = [t.lower().count(w) + 0.01 for w in TOPICS]
        n = math.sqrt(sum(x * x for x in v))
        out.append([x / n for x in v])
    return out


def _para(n, y, text, page=1):
    return Element(element_id=f"{DOC}:p{page}:text:{n}", doc_id=DOC, source_file="x.pdf", page=page,
                   bbox=(50, y, 500, y + 30), type="text", text=text, content_hash=f"t{page}{n}")


def _fig(n, y, caption=None, text=None, page=1):
    return Element(element_id=f"{DOC}:p{page}:vector_figure:{n}", doc_id=DOC, source_file="x.pdf", page=page,
                   bbox=(50, y, 500, y + 100), type="vector_figure", text=text, caption=caption,
                   asset_path="a.png", content_hash=f"f{page}{n}")


def _cap(eid, short, desc, labels):
    return CaptionRecord(element_id=eid, image_hash="h", status="ok", model_path="awq-7b", model_id="m",
                         prompt_version="v2", seconds=1.0, caption=FigureCaption(
                             figure_type="diagram", short_caption=short, detailed_description=desc,
                             visible_text=labels, keywords=[], confidence="high"))


def _doc(elements, captions=(), tables=()):
    elements = sorted(elements, key=lambda e: (e.page, e.bbox[1]))
    return LoadedDoc(doc_id=DOC, source_file="x.pdf", content_hash="c", elements=elements,
                     captions={c.element_id: c for c in captions}, tables={t.element_id: t for t in tables})


# ---------------------------------------------------------------- label cross-check

def test_label_cross_check_flags_captions_that_misread_the_figure(parse_settings):
    good = _fig(1, 100, text="Draft model\nTarget model\naccept")
    bad = _fig(2, 400, text="Input Tokens\nSelect Token")
    doc = _doc([good, bad], captions=[_cap(good.element_id, "s", "d", ["Draft model", "Target model"]),
                                      _cap(bad.element_id, "s", "d", ["Encoder", "Decoder", "Softmax"])])
    report = cross_check_labels(doc, parse_settings.enrich)
    assert report[good.element_id] == 1.0
    assert report[bad.element_id] < parse_settings.enrich.label_overlap_min
    assert doc.captions[good.element_id].status == "ok"
    flagged = doc.captions[bad.element_id]
    assert flagged.status == "needs_review" and "labels" in flagged.error


# ---------------------------------------------------------------- links

def _links(doc, settings):
    links, unlinked = find_links(doc, fake_embed, settings.enrich)
    return {(lk.target_id.split(":", 1)[1], lk.text_element_id.split(":", 1)[1], lk.method) for lk in links}, unlinked


def test_explicit_figure_and_table_references(parse_settings):
    fig = _fig(1, 300, caption="Figure 3. The token loop.")
    table_el = Element(element_id=f"{DOC}:p2:table:1", doc_id=DOC, source_file="x.pdf", page=2,
                       bbox=(50, 100, 500, 200), type="table", text="A | B", content_hash="tb")
    table = Table(element_id=table_el.element_id, columns=["A", "B"], rows=[["1", "2"]], title="Table 2: Scores")
    doc = _doc([_para(1, 100, "As Figure 3 shows, tokens loop."), fig, _para(2, 420, "Figure 3. The token loop."),
                _para(1, 400, "Table 2 compares the scores.", page=2), table_el], tables=[table])
    links, unlinked = _links(doc, parse_settings)
    assert ("p1:vector_figure:1", "p1:text:1", "explicit") in links
    assert not any(t == "p1:text:2" for _, t, _ in links)  # the caption line itself is not "discussion"
    assert ("p2:table:1", "p2:text:1", "explicit") in links
    assert unlinked == []


def test_deictic_phrase_links_the_figure_in_that_direction(parse_settings):
    doc = _doc([_fig(1, 50), _para(1, 200, "Memory use is shown in the diagram below."), _fig(2, 260)])
    links, _ = _links(doc, parse_settings)
    assert ("p1:vector_figure:2", "p1:text:1", "deictic") in links
    assert not any(f == "p1:vector_figure:1" and m == "deictic" for f, _, m in links)


def test_wrong_neighbour_links_the_related_paragraph(parse_settings):
    """Owner's case: the paragraph right above the figure is about something else."""
    fig = _fig(1, 200)
    doc = _doc([
        _para(1, 150, "Quantization halves memory: quantized weights take fewer bits."),
        fig,
        _para(2, 320, "So: let a small draft model guess tokens; the target model checks the draft."),
    ], captions=[_cap(fig.element_id, "Draft model drafts, target model checks",
                      "A draft model proposes tokens and the target model verifies the draft.",
                      ["Draft model", "Target model"])])
    links, unlinked = _links(doc, parse_settings)
    assert links == {("p1:vector_figure:1", "p1:text:2", "related")}
    assert unlinked == []


def test_unrelated_neighbours_leave_the_figure_unlinked(parse_settings):
    fig = _fig(1, 200)
    doc = _doc([_para(1, 150, "Quantization halves memory with fewer bits."), fig],
               captions=[_cap(fig.element_id, "Draft model", "A draft model proposes a draft.", ["Draft model"])])
    links, unlinked = _links(doc, parse_settings)
    assert links == set() and unlinked == [fig.element_id]


# ---------------------------------------------------------------- table summaries

def test_table_summaries_are_cached_per_model(tmp_path):
    table = Table(element_id="t1", columns=["Heads", "Memory"], rows=[["40", "800 KB"], ["8", "160 KB"]],
                  title="KV cache")
    calls = []

    def chat(model, prompt):
        calls.append((model, prompt))
        return f"Summary by {model}."

    cache = tmp_path / "summaries.jsonl"
    assert summarize_tables([table], "m1", chat, cache) == {"t1": "Summary by m1."}
    assert "800 KB" in calls[0][1] and "Heads" in calls[0][1] and "KV cache" in calls[0][1]
    assert summarize_tables([table], "m1", chat, cache) == {"t1": "Summary by m1."}
    assert len(calls) == 1  # cached
    summarize_tables([table], "m2", chat, cache)
    assert len(calls) == 2  # another model is another cache entry
