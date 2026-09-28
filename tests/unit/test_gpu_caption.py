"""The GPU captioning job's logic, run locally with a fake model (LLD §3.4). Written test-first.

The job file imports nothing from mmrag, so it is loaded by path. Model backends are
only imported inside the job's loaders and are not exercised here.
"""

from __future__ import annotations

import importlib.util
import json
import sys

import pytest

from mmrag.config import PROJECT_ROOT
from mmrag.ingest.captions import CaptionRecord, FigureCaption

pytestmark = pytest.mark.unit

_spec = importlib.util.spec_from_file_location("gpu_caption", PROJECT_ROOT / "gpu_job" / "caption" / "caption.py")
gpu = importlib.util.module_from_spec(_spec)
sys.modules["gpu_caption"] = gpu  # pydantic resolves the module's type hints through sys.modules
_spec.loader.exec_module(gpu)

GOOD = {
    "figure_type": "diagram", "short_caption": "Four-step generation loop",
    "detailed_description": "Input Tokens → Compute Probabilities → Select Token → Stop or continue?",
    "visible_text": ["1: Input Tokens"], "extracted_data": None, "keywords": ["loop"], "confidence": "high",
}


def _job(n, **kw):
    return {"element_id": f"d:p{n}:vector_figure:1", "kind": "vector_figure", "page": n, "image": f"images/{n}.png",
            "image_hash": f"hash{n}", "source_file": "x.pdf", "section_path": ["Part 1", "Loop"],
            "pdf_caption": "The loop.", "context_before": "before words", "context_after": "after words",
            "figure_refs": [], **kw}


class FakeBackend:
    """Replies from a script: one queue of raw strings per element_id."""

    model_id = "fake/model"

    def __init__(self, replies: dict[str, list[str]]):
        self.replies = replies
        self.calls = []

    def generate(self, items, retry=False):
        self.calls.append(([job["element_id"] for job, _, _ in items], retry))
        return [self.replies[job["element_id"]].pop(0) for job, _, _ in items]


def test_schema_matches_the_project_copy():
    assert gpu.FigureCaption.model_json_schema() == FigureCaption.model_json_schema()


def test_visible_text_is_capped():
    """Pilot: the AWQ model looped on a grid ("The", "dog", "bit", ...) until cut off. With the
    cap in the schema, vLLM's guided decoding must close the list instead."""
    with pytest.raises(Exception):
        gpu.FigureCaption.model_validate({**GOOD, "visible_text": ["x"] * (gpu.MAX_VISIBLE_TEXT + 1)})
    assert gpu.FigureCaption.model_json_schema()["properties"]["visible_text"]["maxItems"] == gpu.MAX_VISIBLE_TEXT


def test_prompt_v2_rules_for_chart_values():
    """Pilot: all three models printed a value for a bar with no number on it, flagged exact."""
    template = (PROJECT_ROOT / "gpu_job" / "caption" / "prompts" / "caption_v2.md").read_text(encoding="utf-8")
    prompt = gpu.build_prompt(template, _job(4))
    assert "no number printed" in prompt and "never" in prompt.lower()
    assert "{schema}" not in prompt


def test_prompt_is_filled_in():
    template = (PROJECT_ROOT / "gpu_job" / "caption" / "prompts" / "caption_v1.md").read_text(encoding="utf-8")
    prompt = gpu.build_prompt(template, _job(3, figure_refs=["See Figure 3."]))
    assert "Part 1 > Loop" in prompt and "The loop." in prompt and "See Figure 3." in prompt
    assert '"figure_type"' in prompt  # the JSON schema is included
    assert "{context_before}" not in prompt and "{schema}" not in prompt


@pytest.mark.parametrize("raw", [
    json.dumps(GOOD),
    "```json\n" + json.dumps(GOOD) + "\n```",
    "Here is the caption:\n" + json.dumps(GOOD) + "\nHope this helps.",
    json.dumps(GOOD, indent=1).replace('"high"\n}', '"high",\n}'),  # trailing comma
])
def test_parse_accepts_valid_and_repairable_json(raw):
    caption, error = gpu.parse_caption(raw)
    assert error is None and caption["short_caption"] == "Four-step generation loop"


def test_parse_rejects_wrong_schema():
    caption, error = gpu.parse_caption(json.dumps({**GOOD, "figure_type": "painting"}))
    assert caption is None and "figure_type" in error


def test_run_repairs_retries_then_flags(tmp_path):
    jobs = [_job(1), _job(2), _job(3)]
    backend = FakeBackend({
        "d:p1:vector_figure:1": [json.dumps(GOOD)],
        "d:p2:vector_figure:1": ["not json", json.dumps(GOOD)],  # fixed by the one retry
        "d:p3:vector_figure:1": ["not json", "still not json"],  # flagged
    })
    report = gpu.run_jobs(jobs, backend, tmp_path, tmp_path / "out", "3b", "v1", "{pdf_caption}", batch_size=2)
    records = [CaptionRecord.model_validate_json(line)
               for line in (tmp_path / "out" / "captions.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [r.status for r in records] == ["ok", "ok", "needs_review"]
    assert records[2].raw == "still not json" and records[2].error
    assert records[0].model_id == "fake/model" and records[0].model_path == "3b" and records[0].image_hash == "hash1"
    assert report["ok"] == 2 and report["needs_review"] == 1 and report["validity_rate"] == pytest.approx(2 / 3)


def test_retry_runs_with_different_settings(tmp_path):
    """Pilot v3: greedy decoding looped inside one label; an identical retry loops identically.
    Only the failed figure is retried, and with the retry settings."""
    backend = FakeBackend({"d:p1:vector_figure:1": [json.dumps(GOOD)],
                           "d:p2:vector_figure:1": ["looping...", json.dumps(GOOD)]})
    gpu.run_jobs([_job(1), _job(2)], backend, tmp_path, tmp_path / "out", "awq-7b", "v2", "{pdf_caption}")
    assert backend.calls == [(["d:p1:vector_figure:1", "d:p2:vector_figure:1"], False),
                             (["d:p2:vector_figure:1"], True)]


def test_each_visible_label_is_capped():
    """Pilot v3: the loop happened inside a single label string, which the item cap cannot stop."""
    schema = gpu.FigureCaption.model_json_schema()
    assert schema["properties"]["visible_text"]["items"]["maxLength"] == gpu.MAX_LABEL_CHARS
    with pytest.raises(Exception):
        gpu.FigureCaption.model_validate({**GOOD, "visible_text": ["q1k1 " * 100]})


def test_run_resumes_after_interruption(tmp_path):
    jobs = [_job(1), _job(2)]
    first = FakeBackend({"d:p1:vector_figure:1": [json.dumps(GOOD)], "d:p2:vector_figure:1": [json.dumps(GOOD)]})
    gpu.run_jobs(jobs[:1], first, tmp_path, tmp_path / "out", "3b", "v1", "{pdf_caption}")
    second = FakeBackend({"d:p1:vector_figure:1": [], "d:p2:vector_figure:1": [json.dumps(GOOD)]})
    gpu.run_jobs(jobs, second, tmp_path, tmp_path / "out", "3b", "v1", "{pdf_caption}")
    assert second.calls == [(["d:p2:vector_figure:1"], False)]  # the done job is not captioned again
    lines = (tmp_path / "out" / "captions.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
