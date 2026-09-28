"""Bring captions from a GPU run back into the project (plan Phase 2, task 7; spec §6.3).

Every record is re-validated against the caption schema. Low-confidence captions are
downgraded to `needs_review`. Accepted captions go to the cache, so the next bundle
leaves those figures out and a re-run makes no model calls.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from mmrag.config import Settings
from mmrag.ingest.captions import CaptionCache, CaptionRecord, load_records


@dataclass
class ImportResult:
    ok: int = 0
    needs_review: int = 0
    errors: list[str] = field(default_factory=list)
    path: Path | None = None


def captions_path(settings: Settings, doc_id: str) -> Path:
    return settings.resolve(settings.paths.data_dir) / "captions" / doc_id / "captions.jsonl"


def cache_path(settings: Settings) -> Path:
    return settings.resolve(settings.paths.data_dir) / "captions" / "cache.jsonl"


def import_captions(run_file: Path, doc_id: str, settings: Settings) -> ImportResult:
    records, errors = load_records(run_file)
    result = ImportResult(errors=errors, path=captions_path(settings, doc_id))

    checked = []
    for r in records:
        if r.status == "ok" and r.caption and r.caption.confidence == "low":
            r = r.model_copy(update={"status": "needs_review", "error": "model rated its own caption low confidence"})
        checked.append(r)

    stored: dict[str, CaptionRecord] = {}
    if result.path.is_file():
        stored = {r.element_id: r for r in load_records(result.path)[0]}
    for r in checked:
        stored[r.element_id] = r  # a newer run replaces the element's previous record
    result.path.parent.mkdir(parents=True, exist_ok=True)
    result.path.write_text("".join(r.model_dump_json() + "\n" for r in stored.values()), encoding="utf-8")

    cache = CaptionCache(cache_path(settings))
    for r in checked:
        cache.put(r)
    result.ok = sum(r.status == "ok" for r in checked)
    result.needs_review = len(checked) - result.ok
    return result
