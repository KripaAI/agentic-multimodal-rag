"""Records produced by parsing (spec §5.1, §5.3; LLD §3.2).

Every record is written to JSONL as-is, so field names are part of the
on-disk contract read by later phases.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

BBox = tuple[float, float, float, float]  # x0, y0, x1, y1 in PDF points, origin top-left

ElementType = Literal["text", "image", "vector_figure", "table", "scanned_page"]
Status = Literal["ok", "skipped", "needs_review"]


class _Record(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Element(_Record):
    """One extracted item with full provenance (P2).

    `status == "skipped"` requires `skip_reason` (P1). Skipped elements are
    kept (e.g. decorative images) so nothing disappears without a trace.
    """

    element_id: str
    doc_id: str
    source_file: str
    page: int  # 1-based
    bbox: BBox
    type: ElementType
    section_path: list[str] = []
    text: str | None = None  # text elements: the text; figures: the labels drawn inside them
    caption: str | None = None  # figures and images: the nearby "Figure N…" or italic caption line
    asset_path: str | None = None  # relative to paths.data_dir
    content_hash: str
    status: Status = "ok"
    skip_reason: str | None = None


class Table(_Record):
    """Structured content of a `table` element. `summary` is filled in Phase 3."""

    element_id: str
    columns: list[str]
    rows: list[list[str | None]]
    units: dict[str, str] = {}
    numeric_columns: list[str] = []
    title: str | None = None
    summary: str | None = None
    low_confidence: bool = False  # also rendered as an image for the VLM


class SkipRecord(_Record):
    """An intentional drop that has no element of its own (P1 audit trail)."""

    doc_id: str
    page: int
    kind: Literal["header_footer", "duplicate_image", "undisplayed_image", "empty_page"]
    reason: str
    bbox: BBox | None = None
    text: str | None = None


class RejectedRegion(_Record):
    """A vector-drawing region discarded by a figure filter; shown on the review sheet."""

    doc_id: str
    page: int
    bbox: BBox
    filter: Literal["isolated_shape", "too_few_shapes", "text_callout", "too_small", "table_region"]
    shape_count: int


class PageProfile(_Record):
    page: int
    text_chars: int
    images: int
    drawings: int
    tables: int
    textless: bool
    vector_heavy: bool


class DocProfile(_Record):
    """Corpus profiler output for one PDF (plan Phase 1, task 1)."""

    doc_id: str
    source_file: str
    pages: list[PageProfile]


class ParseResult(_Record):
    """Everything `parse_document` produces for one PDF, before it is written to disk."""

    doc_id: str
    source_file: str
    page_count: int
    elements: list[Element]
    tables: list[Table]
    skips: list[SkipRecord]
    rejected: list[RejectedRegion]
