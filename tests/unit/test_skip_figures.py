"""Owner-excluded figures (Phase 5: the Post-Training PDF prints unrelated slides under its
captions). Test-first."""

from __future__ import annotations

import pytest

from mmrag.index.chunk import build_figure_docs, chunk_text
from mmrag.index.document import LoadedDoc
from mmrag.ingest.enrich import skip_figures
from mmrag.ingest.models import Element

pytestmark = pytest.mark.unit
DOC = "d" * 16


def _doc(source_file):
    els = [Element(element_id=f"{DOC}:p1:text:1", doc_id=DOC, source_file=source_file, page=1, bbox=(50, 10, 500, 60),
                   type="text", text=" ".join(f"word{i}" for i in range(40)), content_hash="t"),
           Element(element_id=f"{DOC}:p1:image:1", doc_id=DOC, source_file=source_file, page=1,
                   bbox=(50, 80, 500, 300), type="image", asset_path="assets/x.png", caption="Figure 1: GRPO",
                   content_hash="i")]
    return LoadedDoc(doc_id=DOC, source_file=source_file, content_hash="c", elements=els)


def test_listed_documents_lose_their_figures_but_keep_the_record(parse_settings):
    doc = _doc("post.pdf")
    assert skip_figures(doc, ["post.pdf"]) == 1
    fig = doc.elements[1]
    assert fig.status == "skipped" and "owner" in fig.skip_reason  # P1: kept, with the reason
    assert doc.figures() == [] and build_figure_docs(doc, []) == []  # not searchable, never shown
    assert chunk_text(doc, parse_settings.chunk)  # the text is untouched


def test_other_documents_are_untouched():
    doc = _doc("other.pdf")
    assert skip_figures(doc, ["post.pdf"]) == 0 and len(doc.figures()) == 1


def test_the_config_lists_the_post_training_guide(parse_settings):
    assert parse_settings.enrich.skip_figures_for == [
        "The_Complete_Guide_to_Post_Training_LLMs_v2_Expert_Edition.pdf"]
