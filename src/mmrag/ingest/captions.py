"""Figure captions from the VLM (spec §5.2, §6.2; LLD §3.4–3.5, §7).

`FigureCaption` is what the model must produce. `gpu_job/caption/caption.py` holds an
identical copy (the GPU job imports nothing from `mmrag`); a unit test keeps them equal.
`CaptionRecord` is one line of `captions.jsonl`: the caption plus provenance.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

MAX_VISIBLE_TEXT = 80  # labels in the densest pilot figure: about 40
MAX_LABEL_CHARS = 300  # one label; stops a loop inside a single string

FigureType = Literal["diagram", "flowchart", "chart", "table_image", "screenshot", "photo", "equation", "decorative"]


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DataPoint(_Model):
    label: str
    value: float
    flag: Literal["exact", "estimated"]  # exact: printed on the figure; estimated: read off a bar or line


class Series(_Model):
    name: str
    points: list[DataPoint]


class ChartData(_Model):
    chart_kind: Literal["bar", "line", "pie", "scatter", "other"]
    x_label: str | None = None
    y_label: str | None = None
    unit: str | None = None
    series: list[Series]


class TableData(_Model):
    columns: list[str]
    rows: list[list[str]]


class ExtractedData(_Model):
    chart: ChartData | None = None
    table: TableData | None = None


class FigureCaption(_Model):
    """The VLM's answer for one figure (spec §5.2, without `model_id`, which the job adds)."""

    figure_type: FigureType
    short_caption: str = Field(min_length=1)
    detailed_description: str = Field(min_length=1)
    # Both caps stop label-repeating loops: guided decoding must close the string and the list.
    visible_text: list[Annotated[str, Field(max_length=MAX_LABEL_CHARS)]] = Field(max_length=MAX_VISIBLE_TEXT)
    extracted_data: ExtractedData | None = None
    keywords: list[str]
    confidence: Literal["high", "medium", "low"]


class CaptionRecord(_Model):
    """One line of `captions.jsonl`, written by the GPU job and re-validated on import."""

    element_id: str
    image_hash: str
    status: Literal["ok", "needs_review"]
    caption: FigureCaption | None = None
    error: str | None = None
    raw: str | None = None  # the model's text, kept when it could not be parsed
    model_path: str
    model_id: str
    prompt_version: str
    seconds: float

    @model_validator(mode="after")
    def _ok_has_caption(self) -> "CaptionRecord":
        if self.status == "ok" and self.caption is None:
            raise ValueError("status ok requires a caption")
        return self


def cache_key(image_hash: str, model_path: str, prompt_version: str) -> str:
    """Captions are cached by image content, model path and prompt version (LLD §7)."""
    return f"{image_hash}:{model_path}:{prompt_version}"


class CaptionCache:
    """Accepted captions, one JSON record per line in `data/captions/cache.jsonl`."""

    def __init__(self, path: Path):
        self.path = path
        self._records: dict[str, CaptionRecord] = {}
        if path.is_file():
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    r = CaptionRecord.model_validate_json(line)
                    self._records[cache_key(r.image_hash, r.model_path, r.prompt_version)] = r

    def get(self, image_hash: str, model_path: str, prompt_version: str) -> CaptionRecord | None:
        return self._records.get(cache_key(image_hash, model_path, prompt_version))

    def put(self, record: CaptionRecord) -> None:
        """Store an accepted (`ok`) caption; `needs_review` records are never cached."""
        if record.status != "ok":
            return
        key = cache_key(record.image_hash, record.model_path, record.prompt_version)
        if key not in self._records:
            self._records[key] = record
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as f:
                f.write(record.model_dump_json() + "\n")


def load_records(path: Path) -> tuple[list[CaptionRecord], list[str]]:
    """Re-validate a `captions.jsonl` from the GPU job. Lines that fail validation are
    returned as error messages, never silently dropped (P1)."""
    records, errors = [], []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            records.append(CaptionRecord.model_validate(json.loads(line)))
        except (ValidationError, json.JSONDecodeError) as e:
            errors.append(f"line {n}: {e}")
    return records, errors
