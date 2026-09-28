"""Caption bundle: jobs.jsonl + images for the GPU job (LLD §3.3). Written test-first."""

from __future__ import annotations

import json
import zipfile

import pytest

from mmrag.ingest.bundle import build_jobs, write_bundle
from mmrag.ingest.captions import CaptionCache, CaptionRecord, FigureCaption
from mmrag.ingest.models import Element, Table

pytestmark = pytest.mark.unit

DOC = "d" * 16


def _el(page, y, etype="text", n=1, text=None, **kw):
    return Element(element_id=f"{DOC}:p{page}:{etype}:{n}", doc_id=DOC, source_file="x.pdf", page=page,
                   bbox=(50, y, 500, y + 20), type=etype, text=text, content_hash=f"h-{page}-{etype}-{n}", **kw)


def _words(prefix, n):
    return " ".join(f"{prefix}{i}" for i in range(n))


@pytest.fixture
def elements():
    return [
        _el(1, 100, text=_words("a", 200), section_path=["Intro"]),
        _el(2, 100, text="As Figure 3 shows, tokens flow left to right.", n=1, section_path=["Intro", "Flow"]),
        _el(2, 200, "vector_figure", caption="Figure 3. The token flow.", asset_path="assets/d/p2_vector_figure_1.png",
            text="Input\nOutput", section_path=["Intro", "Flow"]),
        _el(2, 300, text="Figure 3. The token flow.", n=2, section_path=["Intro", "Flow"]),
        _el(2, 400, text=_words("b", 200), n=3, section_path=["Intro", "Flow"]),
        _el(3, 100, "image", caption=None, asset_path="assets/d/p3_image_1.png"),
        _el(3, 300, "image", n=2, status="skipped", skip_reason="decorative: 20x20 px"),
        _el(4, 100, "table", asset_path="assets/d/p4_table_1.png", text="A | B"),
        _el(4, 300, "table", n=2, text="C | D"),
    ]


@pytest.fixture
def tables():
    return [
        Table(element_id=f"{DOC}:p4:table:1", columns=["A", "B"], rows=[["1", ""]], low_confidence=True, title="Odd table"),
        Table(element_id=f"{DOC}:p4:table:2", columns=["C", "D"], rows=[["1", "2"]]),
    ]


def test_one_job_per_figure_image_and_low_confidence_table(elements, tables, parse_settings):
    jobs = build_jobs(elements, tables, parse_settings)
    assert [(j.element_id.split(":", 1)[1], j.kind) for j in jobs] == [
        ("p2:vector_figure:1", "vector_figure"),
        ("p3:image:1", "image"),
        ("p4:table:1", "table"),  # low confidence: the VLM transcribes it as a check
    ]


def test_job_carries_section_caption_and_150_words_of_context(elements, tables, parse_settings):
    fig = build_jobs(elements, tables, parse_settings)[0]
    assert fig.section_path == ["Intro", "Flow"]
    assert fig.pdf_caption == "Figure 3. The token flow."
    before, after = fig.context_before.split(), fig.context_after.split()
    assert len(before) == 150 and before[-1] == "right." and "a199" in before  # nearest words, across pages
    assert len(after) == 150 and after[0] == "b0"
    assert "Figure 3. The token flow." not in fig.context_after  # the caption is not repeated as context


def test_figure_references_are_found_elsewhere_in_the_text(elements, tables, parse_settings):
    fig = build_jobs(elements, tables, parse_settings)[0]
    assert fig.figure_refs == ["As Figure 3 shows, tokens flow left to right."]


def test_labels_drawn_in_the_figure_are_not_given_to_the_model(elements, tables, parse_settings):
    """The PDF's own labels are kept for the Phase 3 cross-check, so the VLM must read them unaided."""
    job = build_jobs(elements, tables, parse_settings)[0]
    assert "Input" not in json.dumps(job.model_dump())


def test_cached_figures_are_left_out(elements, tables, parse_settings, tmp_path):
    cache = CaptionCache(tmp_path / "cache.jsonl")
    caption = FigureCaption(figure_type="diagram", short_caption="s", detailed_description="d",
                            visible_text=[], keywords=[], confidence="high")
    cache.put(CaptionRecord(element_id="x", image_hash="h-3-image-1", status="ok", caption=caption,
                            model_path=parse_settings.caption.model_path, model_id="m",
                            prompt_version=parse_settings.caption.prompt_version, seconds=1.0))
    jobs = build_jobs(elements, tables, parse_settings, cache=cache)
    assert "p3:image:1" not in [j.element_id.split(":", 1)[1] for j in jobs]


def test_pages_filter_selects_a_pilot(elements, tables, parse_settings):
    assert [j.page for j in build_jobs(elements, tables, parse_settings, pages=[3, 4])] == [3, 4]


def test_write_bundle(elements, tables, parse_settings, tmp_path):
    data = parse_settings.resolve(parse_settings.paths.data_dir)
    for e in elements:
        if e.asset_path:
            (data / e.asset_path).parent.mkdir(parents=True, exist_ok=True)
            (data / e.asset_path).write_bytes(b"png")
    jobs = build_jobs(elements, tables, parse_settings)
    out = write_bundle(jobs, parse_settings, tmp_path / "bundle", models=["awq-7b", "3b"])
    lines = (out / "jobs.jsonl").read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["image"] for line in lines] == [f"images/{j.element_id.replace(':', '_')}.png" for j in jobs]
    assert all((out / json.loads(line)["image"]).is_file() for line in lines)
    run = json.loads((out / "run.json").read_text(encoding="utf-8"))
    assert run["models"] == ["awq-7b", "3b"]
    assert run["prompt_version"] == parse_settings.caption.prompt_version
    assert run["max_pixels"] == parse_settings.caption.max_pixels
    assert (out / run["prompt_file"]).is_file()
    with zipfile.ZipFile(out.with_suffix(".zip")) as z:
        assert "jobs.jsonl" in z.namelist()
