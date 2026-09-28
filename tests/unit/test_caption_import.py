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


def test_low_confidence_captions_need_review(tmp_path, parse_settings):
    """Spec §6.3: a caption the model itself rates low is not trusted as-is."""
    low = _record("a", caption={**CAPTION, "confidence": "low"})
    result = import_captions(_write(tmp_path / "captions.jsonl", [low]), "d" * 16, parse_settings)
    assert (result.ok, result.needs_review) == (0, 1)
