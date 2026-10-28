"""Scanned-page transcripts from the VLM (the GPU job's "transcribe" task).

A scanned page has no text layer, so its words are not searchable. The GPU job copies the page's
text; `transcripts.jsonl` (next to captions.jsonl) keeps one record per page, and indexing then
treats each transcribed page as one block of text citing that page.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from mmrag.config import Settings
from mmrag.fsutil import jsonl_lines

BLANK_PAGE = "[blank page]"


class TranscriptRecord(BaseModel):
    """One line of transcripts.jsonl, written by the GPU job and re-validated on import."""

    model_config = ConfigDict(extra="forbid")

    element_id: str
    image_hash: str
    status: Literal["ok", "needs_review"]
    text: str
    error: str | None = None
    model_path: str
    model_id: str
    prompt_version: str
    seconds: float = Field(ge=0)


def load_transcripts(path: Path) -> dict[str, TranscriptRecord]:
    """Usable transcripts by element id: status ok, and not a blank page. A missing file is empty."""
    if not path.is_file():
        return {}
    out = {}
    for line in jsonl_lines(path):
        if line:
            r = TranscriptRecord.model_validate_json(line)
            if r.status == "ok" and r.text.strip() and r.text.strip() != BLANK_PAGE:
                out[r.element_id] = r
    return out


@dataclass
class ImportResult:
    path: Path
    ok: int = 0
    needs_review: int = 0
    blank: int = 0
    errors: list[str] = field(default_factory=list)


def import_transcripts(run_file: Path, doc_id: str, settings: Settings) -> ImportResult:
    """Validate a pulled run's transcripts.jsonl and merge it into the document's transcripts
    (a page transcribed again replaces its earlier record). Records for another document fail."""
    target = settings.resolve(settings.paths.data_dir) / "captions" / doc_id / "transcripts.jsonl"
    result = ImportResult(path=target)
    if not run_file.is_file():
        raise FileNotFoundError(f"no transcripts in {run_file}")
    existing = {}
    if target.is_file():
        for line in jsonl_lines(target):
            if line:
                r = TranscriptRecord.model_validate_json(line)
                existing[r.element_id] = r
    for n, line in enumerate(jsonl_lines(run_file), 1):
        if not line:
            continue
        try:
            r = TranscriptRecord.model_validate(json.loads(line))
        except (ValueError, ValidationError) as e:
            result.errors.append(f"line {n}: {str(e).splitlines()[0]}")
            continue
        if not r.element_id.startswith(f"{doc_id}:"):
            result.errors.append(f"line {n}: {r.element_id} belongs to another document")
            continue
        existing[r.element_id] = r
        if r.status != "ok":
            result.needs_review += 1
        elif r.text.strip() == BLANK_PAGE:
            result.blank += 1
        else:
            result.ok += 1
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("".join(r.model_dump_json() + "\n" for r in existing.values()), encoding="utf-8")
    return result
