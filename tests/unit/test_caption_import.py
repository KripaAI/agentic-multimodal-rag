"""Kaggle job settings and caption import (plan Phase 2, tasks 3 and 7). Written test-first."""

from __future__ import annotations

import json

import pytest

from mmrag.ingest.captions import CaptionCache
from mmrag.ingest.caption_import import import_captions
from mmrag.ingest.kaggle_job import dataset_metadata, kernel_metadata

pytestmark = pytest.mark.unit

CAPTION = {"figure_type": "chart", "short_caption": "Next-token probabilities",
           "detailed_description": "Bar chart of candidate tokens.", "visible_text": ["41%"],
           "extracted_data": {"chart": {"chart_kind": "bar", "series": [{"name": "p", "points": [
               {"label": "Paris", "value": 41, "flag": "exact"}]}]}},
           "keywords": ["softmax"], "confidence": "high"}


def _record(eid, status="ok", **kw):
    return {"element_id": eid, "image_hash": f"hash-{eid}", "status": status,
            "caption": CAPTION if status == "ok" else None, "error": None if status == "ok" else "schema",
            "raw": None if status == "ok" else "{broken", "model_path": "awq-7b", "model_id": "Qwen/x",
            "prompt_version": "v1", "seconds": 3.0, **kw}


def _write(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    return path


def test_kernel_is_private_with_gpu_internet_and_the_bundle():
    meta = kernel_metadata("someone", "abc")
    assert meta["is_private"] is True and meta["enable_gpu"] is True and meta["enable_internet"] is True
    assert meta["dataset_sources"] == [dataset_metadata("someone", "abc")["id"]]
    assert meta["code_file"] == "caption.py"
    # Kaggle rejects a kernel whose title is already used by one of the user's datasets.
    assert meta["title"] != dataset_metadata("someone", "abc")["title"]
    assert meta["id"].split("/")[1] != dataset_metadata("someone", "abc")["id"].split("/")[1]


def test_import_validates_merges_and_caches(tmp_path, parse_settings):
    doc = "d" * 16
    run = _write(tmp_path / "run" / "captions.jsonl",
                 [_record("a"), _record("b", status="needs_review"), {"element_id": "c", "status": "ok"}])
    result = import_captions(run, doc, parse_settings)
    assert (result.ok, result.needs_review, len(result.errors)) == (1, 1, 1)  # the malformed line is reported

    stored = parse_settings.resolve(parse_settings.paths.data_dir) / "captions" / doc / "captions.jsonl"
    assert [json.loads(line)["element_id"] for line in stored.read_text(encoding="utf-8").splitlines()] == ["a", "b"]
    cache = CaptionCache(parse_settings.resolve(parse_settings.paths.data_dir) / "captions" / "cache.jsonl")
    assert cache.get("hash-a", "awq-7b", "v1") is not None
    assert cache.get("hash-b", "awq-7b", "v1") is None  # needs_review is never cached

    # A later run that fixes "b" replaces its record instead of adding a second one.
    import_captions(_write(tmp_path / "run2" / "captions.jsonl", [_record("b")]), doc, parse_settings)
    lines = [json.loads(line) for line in stored.read_text(encoding="utf-8").splitlines()]
    assert [(r["element_id"], r["status"]) for r in lines] == [("a", "ok"), ("b", "ok")]


def _chart(points):
    return {**CAPTION, "extracted_data": {"chart": {"chart_kind": "bar", "unit": "%", "series": [
        {"name": "p", "points": [{"label": lab, "value": v, "flag": f} for lab, v, f in points]}]}}}


def test_exact_flags_are_checked_against_the_printed_numbers():
    """Pilot v2: the model marked a bar with no printed number 'exact' (P4). Only numbers
    actually printed on the figure stay exact."""
    from mmrag.ingest.caption_import import verify_chart_values

    caption = _chart([("There", 41, "exact"), ("Yes", 1.4, "exact"), ("Sure", 0.4, "exact")])
    fixed, note = verify_chart_values(caption, "Next token probabilities\n41%\n29%\n1.4%\nThere\nYes\nSure")
    flags = [p["flag"] for p in fixed["extracted_data"]["chart"]["series"][0]["points"]]
    assert flags == ["exact", "exact", "estimated"]
    assert "Sure" in note


def test_printed_values_are_flagged_exact():
    """Pilot v2: after the stricter prompt the model marked printed values (29%, 1.4%) as
    estimated. The flag follows the figure, in both directions."""
    from mmrag.ingest.caption_import import verify_chart_values

    caption = _chart([("There", 41, "exact"), ("The", 29, "estimated"), ("Sure", 0.4, "estimated")])
    fixed, note = verify_chart_values(caption, "41%\n29%\n1.4%")
    assert [p["flag"] for p in fixed["extracted_data"]["chart"]["series"][0]["points"]] == ["exact", "exact", "estimated"]
    assert "The" in note


def test_printed_percentages_match_their_fractions():
    """Full run: the model wrote the printed "41%" as 0.41. Both forms are the printed value."""
    from mmrag.ingest.caption_import import verify_chart_values

    caption = _chart([("41%", 0.41, "estimated"), ("29%", 29, "estimated"), ("x", 0.5, "exact")])
    fixed, _ = verify_chart_values(caption, "41%\n29%")
    assert [p["flag"] for p in fixed["extracted_data"]["chart"]["series"][0]["points"]] == ["exact", "exact", "estimated"]


def test_chart_values_on_a_figure_without_numbers_are_removed():
    """Pilot v2: bars with no printed values and no scale got invented numbers (P4)."""
    from mmrag.ingest.caption_import import verify_chart_values

    caption = _chart([("sharp", 1.0, "exact"), ("flat", 0.4, "exact")])
    fixed, note = verify_chart_values(caption, "Temperature 0.2\nsharp — nearly deterministic\nflat")
    assert fixed["extracted_data"] is None
    assert "no printed numbers" in note


def test_import_uses_the_pdf_labels_of_vector_figures(tmp_path, parse_settings):
    doc = "d" * 16
    el_dir = parse_settings.resolve(parse_settings.paths.data_dir) / "elements" / doc
    el_dir.mkdir(parents=True)
    element = {"element_id": "a", "doc_id": doc, "source_file": "x.pdf", "page": 1, "bbox": [0, 0, 10, 10],
               "type": "vector_figure", "section_path": [], "text": "Paris\n41%", "caption": None,
               "asset_path": "p.png", "content_hash": "hash-a", "status": "ok", "skip_reason": None}
    (el_dir / "elements.jsonl").write_text(json.dumps(element) + "\n", encoding="utf-8")
    rec = _record("a", caption=_chart([("Paris", 41, "exact"), ("Rome", 12, "exact")]))
    import_captions(_write(tmp_path / "captions.jsonl", [rec]), doc, parse_settings)
    stored = json.loads((el_dir.parent.parent / "captions" / doc / "captions.jsonl").read_text(encoding="utf-8"))
    points = stored["caption"]["extracted_data"]["chart"]["series"][0]["points"]
    assert [p["flag"] for p in points] == ["exact", "estimated"]
    assert "Rome" in stored["error"]  # the change is recorded, not silent


def test_reimport_updates_the_cache(tmp_path, parse_settings):
    """Re-importing after a rule fix must not leave the old record in the cache."""
    doc = "d" * 16
    first = _record("a", caption=_chart([("x", 7, "exact")]))
    import_captions(_write(tmp_path / "1" / "captions.jsonl", [first]), doc, parse_settings)
    second = _record("a", caption=_chart([("x", 8, "exact")]))
    import_captions(_write(tmp_path / "2" / "captions.jsonl", [second]), doc, parse_settings)
    cache = CaptionCache(parse_settings.resolve(parse_settings.paths.data_dir) / "captions" / "cache.jsonl")
    assert cache.get("hash-a", "awq-7b", "v1").caption.extracted_data.chart.series[0].points[0].value == 8


def test_caption_review_page(tmp_path, parse_settings):
    from mmrag.ingest.caption_report import build_caption_review

    doc = "d" * 16
    data = parse_settings.resolve(parse_settings.paths.data_dir)
    (data / "elements" / doc).mkdir(parents=True)
    element = {"element_id": "a", "doc_id": doc, "source_file": "x.pdf", "page": 3, "bbox": [0, 0, 10, 10],
               "type": "vector_figure", "section_path": ["Part 1"], "text": "41%", "caption": "The PDF caption",
               "asset_path": f"assets/{doc}/p3_vector_figure_1.png", "content_hash": "hash-a", "status": "ok",
               "skip_reason": None}
    (data / "elements" / doc / "elements.jsonl").write_text(json.dumps(element) + "\n", encoding="utf-8")
    import_captions(_write(tmp_path / "captions.jsonl", [_record("a")]), doc, parse_settings)
    page = build_caption_review(doc, parse_settings).read_text(encoding="utf-8")
    assert "Next-token probabilities" in page and "The PDF caption" in page and "p3_vector_figure_1.png" in page
    assert "Part 1" in page


def test_low_confidence_captions_need_review(tmp_path, parse_settings):
    """Spec §6.3: a caption the model itself rates low is not trusted as-is."""
    low = _record("a", caption={**CAPTION, "confidence": "low"})
    result = import_captions(_write(tmp_path / "captions.jsonl", [low]), "d" * 16, parse_settings)
    assert (result.ok, result.needs_review) == (0, 1)
