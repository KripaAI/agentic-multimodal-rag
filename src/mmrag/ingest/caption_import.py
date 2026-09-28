"""Bring captions from a GPU run back into the project (plan Phase 2, task 7; spec §6.3).

Every record is re-validated against the caption schema. Low-confidence captions are
downgraded to `needs_review`. Accepted captions go to the cache, so the next bundle
leaves those figures out and a re-run makes no model calls.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import json
import re

from mmrag.config import Settings
from mmrag.ingest.captions import CaptionCache, CaptionRecord, load_records

_NUMBER = r"-?\d[\d,]*(?:\.\d+)?"
# A printed data value: a line that is only a number (optionally with % or a unit), or a
# number directly followed by % or a unit. Numbers inside titles ("Temperature 1.0") are not.
_VALUE_LINE = re.compile(rf"^[~≈<>]?\s*({_NUMBER})\s*(?:%|[A-Za-z×]{{1,3}})?$")
_VALUE_WITH_UNIT = re.compile(rf"({_NUMBER})\s*(?:%|GB|MB|KB|TB|ms|×|x)(?![A-Za-z])")


def printed_values(texts: list[str]) -> set[float]:
    values = set()
    for text in texts:
        for line in text.splitlines():
            line = line.strip()
            m = _VALUE_LINE.match(line)
            found = [m.group(1)] if m else _VALUE_WITH_UNIT.findall(line)
            values.update(float(v.replace(",", "")) for v in found)
    return values


def verify_chart_values(caption: dict, figure_text: str | list[str]) -> tuple[dict, str | None]:
    """Enforce P4 on the model's chart values, which the pilot showed it cannot flag reliably
    (too loose with prompt v1, too strict with v2): a value is `exact` if and only if that
    number is printed on the figure. A figure with no printed numbers at all cannot have had
    values read from it, so its chart data is removed.
    Returns the corrected caption and a note of what changed (None if nothing did)."""
    chart = ((caption.get("extracted_data") or {}).get("chart")) or None
    if not chart:
        return caption, None
    printed = printed_values([figure_text] if isinstance(figure_text, str) else figure_text)
    if not printed:
        data = {k: v for k, v in caption["extracted_data"].items() if k != "chart" and v}
        return {**caption, "extracted_data": data or None}, "chart values removed: the figure has no printed numbers"
    downgraded, upgraded = [], []
    for series in chart["series"]:
        for p in series["points"]:
            is_printed = any(abs(p["value"] - v) < 1e-9 for v in printed)
            if p["flag"] == "exact" and not is_printed:
                p["flag"] = "estimated"
                downgraded.append(p["label"])
            elif p["flag"] == "estimated" and is_printed:
                p["flag"] = "exact"
                upgraded.append(p["label"])
    notes = []
    if downgraded:
        notes.append(f"not printed on the figure, flagged estimated: {', '.join(downgraded)}")
    if upgraded:
        notes.append(f"printed on the figure, flagged exact: {', '.join(upgraded)}")
    return caption, "; ".join(notes) or None


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

    # The PDF's own labels are the ground truth for drawn figures; for embedded images the
    # model's visible_text is all there is.
    el_file = settings.resolve(settings.paths.data_dir) / "elements" / doc_id / "elements.jsonl"
    pdf_text = {}
    if el_file.is_file():
        for line in el_file.read_text(encoding="utf-8").splitlines():
            e = json.loads(line)
            if e["type"] == "vector_figure" and e.get("text"):
                pdf_text[e["element_id"]] = e["text"]

    checked = []
    for r in records:
        if r.status == "ok" and r.caption:
            source = pdf_text.get(r.element_id, r.caption.visible_text)
            fixed, note = verify_chart_values(r.caption.model_dump(), source)
            if note:
                r = r.model_copy(update={"caption": r.caption.model_validate(fixed), "error": note})
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
