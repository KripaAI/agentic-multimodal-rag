"""Caption bundle for the GPU job (LLD §3.3, spec §6.2).

One job per figure, image, textless page and low-confidence table, carrying the
context the VLM needs: section path, the PDF's own caption, about 150 words of text
before and after, and any sentence that refers to the figure by number. The labels
drawn inside a figure are deliberately *not* included: Phase 3 cross-checks the
VLM's `visible_text` against them, which only works if the model read them itself.
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from mmrag.config import PROJECT_ROOT, Settings
from mmrag.fsutil import empty_dir
from mmrag.ingest.captions import CaptionCache
from mmrag.ingest.models import Element, Table

CONTEXT_WORDS = 150
PROMPTS_DIR = PROJECT_ROOT / "gpu_job" / "caption" / "prompts"
_FIGURE_NUMBER = re.compile(r"^\s*(figure|fig\.)\s*(\d+(?:\.\d+)*)", re.IGNORECASE)


class Job(BaseModel):
    """One line of `jobs.jsonl`."""

    model_config = ConfigDict(extra="forbid")

    element_id: str
    kind: str  # vector_figure | image | scanned_page | table
    page: int
    image: str  # path inside the bundle, set by write_bundle
    image_hash: str
    source_file: str
    section_path: list[str]
    pdf_caption: str | None
    context_before: str
    context_after: str
    figure_refs: list[str]


def _norm(text: str | None) -> str:
    return " ".join((text or "").split())


def _figure_refs(caption: str | None, texts: list[Element]) -> list[str]:
    """Text blocks that mention the figure's number ("Figure 3", "Fig. 3"), other than its caption."""
    m = _FIGURE_NUMBER.match(caption or "")
    if not m:
        return []
    pattern = re.compile(rf"\b(figure|fig\.)\s*{re.escape(m.group(2))}\b(?!\.\d)", re.IGNORECASE)
    return [_norm(t.text) for t in texts if pattern.search(t.text or "") and _norm(t.text) != _norm(caption)]


def build_jobs(
    elements: list[Element],
    tables: list[Table],
    settings: Settings,
    cache: CaptionCache | None = None,
    pages: list[int] | None = None,
) -> list[Job]:
    """Jobs in reading order. Figures already cached for the configured model path and
    prompt version are left out, so a re-run makes no model calls (LLD §7)."""
    cfg = settings.caption
    low_conf = {t.element_id: t for t in tables if t.low_confidence}
    order = sorted(elements, key=lambda e: (e.page, e.bbox[1], e.bbox[0]))
    texts = [e for e in order if e.type == "text" and e.text]

    jobs = []
    for i, e in enumerate(order):
        wanted = (e.type in ("vector_figure", "image", "scanned_page") and e.status == "ok") or (
            e.type == "table" and e.element_id in low_conf and e.asset_path
        )
        if not wanted or (pages and e.page not in pages):
            continue
        if cache and cache.get(e.content_hash, cfg.model_path, cfg.prompt_version):
            continue
        caption = e.caption if e.type != "table" else low_conf[e.element_id].title
        before = [t for t in order[:i] if t.type == "text" and t.text]
        after = [t for t in order[i + 1:] if t.type == "text" and t.text and _norm(t.text) != _norm(caption)]
        before_words = " ".join(_norm(t.text) for t in before).split()[-CONTEXT_WORDS:]
        after_words = " ".join(_norm(t.text) for t in after).split()[:CONTEXT_WORDS]
        jobs.append(Job(
            element_id=e.element_id,
            kind=e.type,
            page=e.page,
            image=e.asset_path or "",
            image_hash=e.content_hash,
            source_file=e.source_file,
            section_path=e.section_path,
            pdf_caption=caption,
            context_before=" ".join(before_words),
            context_after=" ".join(after_words),
            figure_refs=_figure_refs(caption, texts),
        ))
    return jobs


def write_bundle(jobs: list[Job], settings: Settings, out: Path, models: list[str]) -> Path:
    """Write `out/` (jobs.jsonl, images/, the prompt, run.json) and `out.zip`.

    `models` are the model paths the GPU job should run: one for a full run, several
    for the pilot comparison (plan Phase 2, task 4)."""
    cfg = settings.caption
    data_dir = settings.resolve(settings.paths.data_dir)
    prompt = PROMPTS_DIR / f"caption_{cfg.prompt_version}.md"
    if not prompt.is_file():
        raise FileNotFoundError(f"prompt file missing: {prompt}")

    empty_dir(out)
    (out / "images").mkdir(exist_ok=True)
    lines = []
    for job in jobs:
        name = f"images/{job.element_id.replace(':', '_')}.png"
        shutil.copyfile(data_dir / job.image, out / name)
        lines.append(job.model_copy(update={"image": name}).model_dump_json())
    (out / "jobs.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    shutil.copyfile(prompt, out / prompt.name)
    run = {"models": models, "prompt_version": cfg.prompt_version, "prompt_file": prompt.name,
           "max_pixels": cfg.max_pixels, "jobs": len(jobs)}
    (out / "run.json").write_text(json.dumps(run, indent=1) + "\n", encoding="utf-8")
    shutil.make_archive(str(out), "zip", out)
    return out
