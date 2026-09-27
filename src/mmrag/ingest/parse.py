"""PDF → elements, figure PNGs, tables and textless pages (LLD §3.2, spec §6.1).

Pipeline for one document, in this order (tables must precede figures):

    profile → furniture text → per page: text, images, tables, vector figures
    → textless pages → section paths → write outputs

Invariant (P1): every page yields at least one element or a SkipRecord.
Tracing: one `ingest.document` trace, an `ingest.parse` span, and a span per
step with page, element and skip counts as attributes (P12).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pymupdf

from mmrag.config import Parse, Settings
from mmrag.ingest.models import (
    BBox,
    DocProfile,
    Element,
    ElementType,
    ParseResult,
    RejectedRegion,
    SkipRecord,
    Table,
)


@dataclass
class ParseContext:
    """Per-document state shared by the extraction steps."""

    doc_id: str
    source_file: str
    cfg: Parse
    asset_dir: Path  # data/assets/{doc_id}/
    furniture: set[tuple[str, int]] = field(default_factory=set)  # (normalized text, rounded y)
    seen_images: set[str] = field(default_factory=set)  # content hashes, for de-duplication
    skips: list[SkipRecord] = field(default_factory=list)
    rejected: list[RejectedRegion] = field(default_factory=list)
    _counters: dict[tuple[int, ElementType], int] = field(default_factory=dict)

    def next_element_id(self, page: int, etype: ElementType) -> str:
        """Allocate the next `element_id` for this page and type (LLD §3.1)."""
        raise NotImplementedError


# ---------------------------------------------------------------- entry points

def profile_pdf(path: Path, cfg: Parse) -> DocProfile:
    """Per-page counts of text characters, images, drawings and tables; flags textless
    and vector-heavy pages. Read-only: writes no assets."""
    raise NotImplementedError


def parse_document(path: Path, settings: Settings) -> ParseResult:
    """Run the full parse for one PDF (traced). Does not write to disk; see `write_outputs`."""
    raise NotImplementedError


def write_outputs(result: ParseResult, settings: Settings) -> dict[str, Path]:
    """Write the audit files (P8) and return their paths by name:

    - `data/elements/{doc_id}/elements.jsonl`
    - `data/elements/{doc_id}/skip_log.jsonl`
    - `data/tables/{doc_id}/tables.jsonl`
    - `data/elements/{doc_id}/review_sheet.html` (via `ingest.review`)

    PNGs are already in `data/assets/{doc_id}/`, written during extraction.
    """
    raise NotImplementedError


# ---------------------------------------------------------------- text

def find_furniture_text(doc: pymupdf.Document, cfg: Parse) -> set[tuple[str, int]]:
    """Text repeated at the same vertical position on more than half the pages
    (running headers, footers, page numbers with digits normalized)."""
    raise NotImplementedError


def extract_text_blocks(page: pymupdf.Page, ctx: ParseContext) -> list[tuple[Element, float | None]]:
    """Text blocks in reading order with bboxes, each paired with its font size if it is
    a heading (else None). Furniture text is dropped and logged to `ctx.skips`."""
    raise NotImplementedError


def _body_font_size(page: pymupdf.Page) -> float:
    """Median span font size on the page, weighted by character count."""
    raise NotImplementedError


def _is_heading(size: float, bold: bool, chars: int, body_size: float, cfg: Parse) -> bool:
    """Larger than body text by `heading_size_margin_pt`, or bold and short."""
    raise NotImplementedError


def assign_section_paths(blocks: list[tuple[Element, float | None]], others: list[Element]) -> None:
    """Fill `section_path` in place, in document order. Heading levels come from font size
    (larger = higher level). Non-text elements (`others`) take the path in force at their
    page and vertical position."""
    raise NotImplementedError


# ---------------------------------------------------------------- images

def extract_images(page: pymupdf.Page, ctx: ParseContext) -> list[Element]:
    """Embedded images at native resolution, saved as PNG. Duplicates (by hash) are logged
    and skipped; images under `min_image_px` become `skipped: decorative` elements."""
    raise NotImplementedError


# ---------------------------------------------------------------- tables

def detect_tables(page: pymupdf.Page, ctx: ParseContext) -> list[tuple[Element, Table]]:
    """PyMuPDF table finder. Runs before figure detection. Low-confidence tables are also
    rendered as PNG (`asset_path` set, `low_confidence=True`) for VLM transcription."""
    raise NotImplementedError


# ---------------------------------------------------------------- vector figures

def detect_vector_figures(
    page: pymupdf.Page, table_bboxes: list[BBox], ctx: ParseContext
) -> list[Element]:
    """The 5-step filter of spec §6.1. Rejected regions go to `ctx.rejected`."""
    raise NotImplementedError


def _mask_table_regions(drawings: list[dict], table_bboxes: list[BBox]) -> list[dict]:
    """Step 1: drop drawing ops inside table bboxes (grid lines are never figures)."""
    raise NotImplementedError


def _drop_page_furniture(drawings: list[dict], page_rect: pymupdf.Rect, cfg: Parse) -> list[dict]:
    """Step 2: drop lines wider than `furniture_width_ratio` of the page, and thin lines
    (aspect > `line_aspect_ratio`) near the top or bottom margin."""
    raise NotImplementedError


def _cluster_drawings(drawings: list[dict], cfg: Parse) -> list[list[dict]]:
    """Group drawings whose bboxes lie within `cluster_merge_distance_pt`, then merge
    overlapping clusters."""
    raise NotImplementedError


def _classify_cluster(cluster: list[dict], page: pymupdf.Page, cfg: Parse) -> str | None:
    """Steps 3–4: return the rejecting filter name (`isolated_shape`, `text_callout`,
    `too_few_shapes`), or None to keep it as a box-and-arrow figure."""
    raise NotImplementedError


def _figure_region(cluster: list[dict], page: pymupdf.Page, cfg: Parse) -> pymupdf.Rect:
    """Step 5: union bbox, grown to include text labels inside it, plus `figure_margin_pt`."""
    raise NotImplementedError


def find_figure_caption(page: pymupdf.Page, region: pymupdf.Rect) -> str | None:
    """A nearby "Figure N…" / "Fig. N…" line below or above the region, if any."""
    raise NotImplementedError


# ---------------------------------------------------------------- textless pages

def detect_textless_pages(doc: pymupdf.Document, ctx: ParseContext) -> list[Element]:
    """Pages with fewer than `textless_chars` characters, rendered whole at `render_dpi`
    as `scanned_page` elements for VLM transcription."""
    raise NotImplementedError


# ---------------------------------------------------------------- rendering

def render_region(page: pymupdf.Page, region: pymupdf.Rect, dpi: int, out: Path) -> Path:
    """Render a page region (or the whole page) to PNG at `dpi`."""
    raise NotImplementedError
