"""Scanned-page transcription: the GPU job's transcribe task (fake model), the bundle, import, and
how indexing treats a transcribed page."""

from __future__ import annotations

import importlib.util
import json
import sys

import pytest

from mmrag.config import PROJECT_ROOT
from mmrag.index.chunk import build_figure_docs, chunk_text
from mmrag.index.document import load_document
from mmrag.ingest import kaggle_job
from mmrag.ingest.bundle import TRANSCRIBE_MAX_PIXELS, build_page_jobs, write_bundle
from mmrag.ingest.captions import CaptionRecord, FigureCaption
from mmrag.ingest.ids import doc_id as compute_doc_id
from mmrag.ingest.models import Element
from mmrag.ingest.transcripts import import_transcripts, load_transcripts

pytestmark = pytest.mark.unit

_spec = importlib.util.spec_from_file_location("gpu_caption_t", PROJECT_ROOT / "gpu_job" / "caption" / "caption.py")
gpu = importlib.util.module_from_spec(_spec)
sys.modules["gpu_caption_t"] = gpu
_spec.loader.exec_module(gpu)

TEMPLATE = (PROJECT_ROOT / "gpu_job" / "caption" / "prompts" / "transcribe_v1.md").read_text(encoding="utf-8")


def _page_job(n):
    return {"element_id": f"d:p{n}:scanned_page:1", "kind": "scanned_page", "page": n, "image": f"images/{n}.png",
            "image_hash": f"hash{n}", "source_file": "book.pdf", "section_path": [], "pdf_caption": None,
            "context_before": "", "context_after": "", "figure_refs": []}


class FakeBackend:
    """Replies from a script: a queue of (text, finish_reason) per element id."""

    model_id = "fake/model"

    def __init__(self, replies):
        self.replies, self.calls, self.last_finish = replies, [], []

    def generate(self, items, retry=False):
        self.calls.append(([j["element_id"] for j, _, _ in items], retry))
        out = [self.replies[j["element_id"]].pop(0) for j, _, _ in items]
        self.last_finish = [f for _, f in out]
        return [t for t, _ in out]


# ---------------------------------------------------------------- the GPU job

def test_prompt_names_the_page_and_keeps_scripts():
    prompt = gpu.build_page_prompt(TEMPLATE, _page_job(7))
    assert "book.pdf, page 7" in prompt and "Devanagari" in prompt and "[illegible]" in prompt
    assert "{" not in prompt


def test_transcribe_writes_text_and_retries_a_cut_off_page(tmp_path):
    backend = FakeBackend({
        "d:p1:scanned_page:1": [("The birth chart of the second child.", "stop")],
        "d:p2:scanned_page:1": [("loop loop loop", "length"), ("Ashtakavarga tables.", "stop")],
        "d:p3:scanned_page:1": [("", "stop"), ("", "stop")],
    })
    report = gpu.run_transcribe([_page_job(1), _page_job(2), _page_job(3)], backend, tmp_path, tmp_path / "out",
                                "awq-7b", "transcribe_v1", TEMPLATE)
    rows = [json.loads(x) for x in (tmp_path / "out" / "transcripts.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [(r["status"], r["text"]) for r in rows] == [
        ("ok", "The birth chart of the second child."), ("ok", "Ashtakavarga tables."), ("needs_review", "")]
    assert backend.calls[1] == (["d:p2:scanned_page:1", "d:p3:scanned_page:1"], True)  # only failures retried
    assert (report["ok"], report["needs_review"]) == (2, 1)


def test_transcribe_resumes_after_a_restart(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    (out / "done_ids.txt").write_text("d:p1:scanned_page:1\n", encoding="utf-8")
    (out / "transcripts.jsonl").write_text("", encoding="utf-8")
    backend = FakeBackend({"d:p2:scanned_page:1": [("Page two.", "stop")]})
    gpu.run_transcribe([_page_job(1), _page_job(2)], backend, tmp_path, out, "awq-7b", "transcribe_v1", TEMPLATE)
    assert backend.calls == [(["d:p2:scanned_page:1"], False)]


def test_cut_off_or_empty_is_not_ok():
    assert gpu.transcript_status("text", "stop") == ("ok", None)
    assert gpu.transcript_status("text", "length")[0] == "needs_review"
    assert gpu.transcript_status("  ", "stop")[0] == "needs_review"


# ---------------------------------------------------------------- bundle and Kaggle names

DOC = "e" * 16


def _el(page, etype, **kw):
    return Element(element_id=f"{DOC}:p{page}:{etype}:1", doc_id=DOC, source_file="book.pdf", page=page,
                   bbox=(0, 0, 600, 800), type=etype, content_hash=f"h{page}", **kw)


def test_page_jobs_are_scanned_pages_only(parse_settings, tmp_path):
    elements = [_el(1, "text", text="printed text"), _el(2, "scanned_page", asset_path="assets/e/p2.png"),
                _el(3, "image", asset_path="assets/e/p3.png"), _el(4, "scanned_page", asset_path="assets/e/p4.png")]
    jobs = build_page_jobs(elements, done={f"{DOC}:p4:scanned_page:1"})
    assert [j.page for j in jobs] == [2]

    data = parse_settings.resolve(parse_settings.paths.data_dir)
    (data / "assets/e").mkdir(parents=True)
    (data / "assets/e/p2.png").write_bytes(b"png")
    out = write_bundle(jobs, parse_settings, tmp_path / "pages", models=["awq-7b"], task="transcribe")
    run = json.loads((out / "run.json").read_text(encoding="utf-8"))
    assert run["task"] == "transcribe" and run["prompt_file"] == "transcribe_v1.md"
    assert run["max_pixels"] == TRANSCRIBE_MAX_PIXELS > parse_settings.caption.max_pixels


def test_transcription_has_its_own_kaggle_job():
    assert kaggle_job.kernel_slug(DOC, "transcribe") != kaggle_job.kernel_slug(DOC)
    assert kaggle_job.dataset_slug(DOC, "transcribe") != kaggle_job.dataset_slug(DOC)
    assert kaggle_job.kernel_metadata("u", DOC, "transcribe")["dataset_sources"] == [
        f"u/{kaggle_job.dataset_slug(DOC, 'transcribe')}"]


# ---------------------------------------------------------------- import and indexing

def _record(eid, text, status="ok", image_hash="h2"):
    return {"element_id": eid, "image_hash": image_hash, "status": status, "text": text, "error": None,
            "model_path": "awq-7b", "model_id": "m", "prompt_version": "transcribe_v1", "seconds": 30.0}


def test_import_merges_and_rejects_other_documents(parse_settings, tmp_path):
    run = tmp_path / "transcripts.jsonl"
    run.write_text("\n".join(json.dumps(r) for r in [
        _record(f"{DOC}:p2:scanned_page:1", "Saturn rules Capricorn."),
        _record(f"{DOC}:p3:scanned_page:1", "[blank page]"),
        _record(f"{DOC}:p4:scanned_page:1", "x x x", status="needs_review"),
        _record(f"{'f' * 16}:p1:scanned_page:1", "other book"),
    ]), encoding="utf-8")
    result = import_transcripts(run, DOC, parse_settings)
    assert (result.ok, result.blank, result.needs_review, len(result.errors)) == (1, 1, 1, 1)
    assert list(load_transcripts(result.path)) == [f"{DOC}:p2:scanned_page:1"]  # only usable text is loaded


def test_a_transcribed_page_is_indexed_as_text(parse_settings, tmp_path):
    pdf = tmp_path / "book.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")
    did = compute_doc_id(pdf)
    data = parse_settings.resolve(parse_settings.paths.data_dir)
    els = [Element(element_id=f"{did}:p1:text:1", doc_id=did, source_file="book.pdf", page=1, bbox=(0, 0, 9, 9),
                   type="text", text="Printed page one.", content_hash="t1"),
           Element(element_id=f"{did}:p2:scanned_page:1", doc_id=did, source_file="book.pdf", page=2,
                   bbox=(0, 0, 600, 800), type="scanned_page", content_hash="h2", asset_path="a/p2.png"),
           Element(element_id=f"{did}:p3:scanned_page:1", doc_id=did, source_file="book.pdf", page=3,
                   bbox=(0, 0, 600, 800), type="scanned_page", content_hash="h3", asset_path="a/p3.png")]
    (data / "elements" / did).mkdir(parents=True)
    (data / "elements" / did / "elements.jsonl").write_text("".join(e.model_dump_json() + "\n" for e in els),
                                                           encoding="utf-8")
    (data / "captions" / did).mkdir(parents=True)
    (data / "captions" / did / "transcripts.jsonl").write_text("\n".join(json.dumps(r) for r in [
        _record(f"{did}:p2:scanned_page:1", "Saturn rules Capricorn and Aquarius."),
        _record(f"{did}:p3:scanned_page:1", "Made from an older scan.", image_hash="stale"),  # image changed
    ]), encoding="utf-8")

    doc = load_document(pdf, parse_settings)
    assert doc.transcribed == {f"{did}:p2:scanned_page:1"}
    chunks = chunk_text(doc, parse_settings.chunk)
    assert any("Saturn rules Capricorn" in c.dense_text and f"{did}:p2:scanned_page:1" in c.element_ids
               for c in chunks)
    figure_ids = {eid for c in build_figure_docs(doc, []) for eid in c.element_ids}
    assert f"{did}:p2:scanned_page:1" not in figure_ids  # now text, not an image to describe
    assert f"{did}:p3:scanned_page:1" not in figure_ids  # not transcribed, and nothing describes it

    p3 = f"{did}:p3:scanned_page:1"
    doc.captions[p3] = CaptionRecord(element_id=p3, image_hash="h3", status="ok", model_path="awq-7b", model_id="m",
                                     prompt_version="v2", seconds=1.0, caption=FigureCaption(
                                         figure_type="table_image", short_caption="Planetary friendship table",
                                         detailed_description="d", visible_text=[], keywords=[], confidence="high"))
    assert p3 in {eid for c in build_figure_docs(doc, []) for eid in c.element_ids}  # a captioned page image stays
