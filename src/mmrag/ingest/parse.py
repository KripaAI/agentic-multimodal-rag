"""PDF → elements, figure PNGs, tables and textless pages (LLD §3.2, spec §6.1).

Pipeline for one document. Per page, tables come first so their grid lines are
never mistaken for diagrams, and text comes last so text already captured by a
table or figure is not duplicated:

    furniture text → per page: tables, images, vector figures, text
    → textless pages → section paths

Invariant (P1): every page yields at least one element or a SkipRecord.
Tracing: one `ingest.document` trace with an `ingest.parse` span and a
`parse.page` span per page carrying element counts (P12).
"""

from __future__ import annotations

import re
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import pymupdf

from mmrag.config import Parse, Settings
from mmrag.ingest.ids import content_hash, doc_id, element_id
from mmrag.ingest.models import (
    BBox,
    DocProfile,
    Element,
    ElementType,
    PageProfile,
    ParseResult,
    RejectedRegion,
    SkipRecord,
    Table,
)
from mmrag.obs import get_logger, get_tracer

_log = get_logger("mmrag.ingest.parse")
pymupdf.no_recommend_layout()  # silence PyMuPDF's one-time stdout advert for its layout add-on

_CAPTION_START = re.compile(r"^\s*(figure|fig\.)\s*\d", re.IGNORECASE)
_NUMERIC = re.compile(r"^[~≈<>]?\s*[-+]?\$?\d[\d,]*(\.\d+)?\s*(%|[kKmMbBxX×])?$")
_BOLD_FONT = re.compile(r"bold|semibold|black|heavy", re.IGNORECASE)
_ITALIC_FONT = re.compile(r"italic|oblique", re.IGNORECASE)
_MONO_FONT = re.compile(r"mono|courier|consol", re.IGNORECASE)

_MIN_PAGES_FOR_FURNITURE = 3  # on shorter documents every line would look "repeated"
_ISOLATED_MAX_ITEMS = 30  # a single path with more segments than this may be a whole drawing
_BACKGROUND_AREA_RATIO = 0.5  # a drawing covering this share of the page is a background or frame
_THIN_LINE_MAX_PT = 3


@dataclass
class TextBlock:
    """A PyMuPDF text block, reduced to what the parsing rules need."""

    bbox: pymupdf.Rect
    text: str
    size: float  # font size of the dominant span
    bold: bool
    italic: bool
    mono: bool = False

    @property
    def chars(self) -> int:
        return len(self.text)


@dataclass
class ParseContext:
    """Per-document state shared by the extraction steps."""

    doc_id: str
    source_file: str
    cfg: Parse
    asset_dir: Path  # data/assets/{doc_id}/
    furniture: set[tuple[str, int]] = field(default_factory=set)  # (normalized text, y bucket)
    seen_images: dict[str, str] = field(default_factory=dict)  # content hash -> element_id
    skips: list[SkipRecord] = field(default_factory=list)
    rejected: list[RejectedRegion] = field(default_factory=list)
    _counters: dict[tuple[int, ElementType], int] = field(default_factory=dict)

    def next_element_id(self, page: int, etype: ElementType) -> str:
        """Allocate the next `element_id` for this page and type (LLD §3.1)."""
        n = self._counters.get((page, etype), 0) + 1
        self._counters[(page, etype)] = n
        return element_id(self.doc_id, page, etype, n)

    def asset(self, eid: str) -> tuple[Path, str]:
        """PNG path on disk and its `asset_path` relative to the data dir."""
        _, page, etype, n = eid.split(":")
        name = f"{page}_{etype}_{n}.png"
        return self.asset_dir / name, f"assets/{self.doc_id}/{name}"

    def element(self, page: int, etype: ElementType, bbox: pymupdf.Rect, content: str | bytes, **kw) -> Element:
        return Element(
            element_id=self.next_element_id(page, etype),
            doc_id=self.doc_id,
            source_file=self.source_file,
            page=page,
            bbox=_bbox(bbox),
            type=etype,
            content_hash=content_hash(content),
            **kw,
        )


# ---------------------------------------------------------------- geometry helpers

def _bbox(r: pymupdf.Rect) -> BBox:
    return (round(r.x0, 2), round(r.y0, 2), round(r.x1, 2), round(r.y1, 2))


def _rect_distance(a: pymupdf.Rect, b: pymupdf.Rect) -> float:
    """Largest axis gap between two rectangles; 0 when they touch or overlap."""
    dx = max(0.0, a.x0 - b.x1, b.x0 - a.x1)
    dy = max(0.0, a.y0 - b.y1, b.y0 - a.y1)
    return max(dx, dy)


def _center_in(r: pymupdf.Rect, region: pymupdf.Rect) -> bool:
    return region.contains(pymupdf.Point((r.x0 + r.x1) / 2, (r.y0 + r.y1) / 2))


def _union(rects) -> pymupdf.Rect:
    out = pymupdf.Rect(rects[0])
    for r in rects[1:]:
        out |= r
    return out


# ---------------------------------------------------------------- entry points

def profile_pdf(path: Path, cfg: Parse) -> DocProfile:
    """Per-page counts of text characters, images, drawings and tables; flags textless
    and vector-heavy pages. Read-only: writes no assets."""
    pages = []
    with pymupdf.open(path) as doc:
        for page in doc:
            chars = len(page.get_text().strip())
            drawings = len(page.get_drawings())
            pages.append(PageProfile(
                page=page.number + 1,
                text_chars=chars,
                images=len(page.get_images(full=True)),
                drawings=drawings,
                tables=len(_find_tables(page, cfg)),
                textless=chars < cfg.textless_chars,
                vector_heavy=drawings >= cfg.vector_heavy_drawings,
            ))
    return DocProfile(doc_id=doc_id(path), source_file=path.name, pages=pages)


def parse_document(path: Path, settings: Settings) -> ParseResult:
    """Run the full parse for one PDF (traced). Writes figure PNGs to the asset dir;
    everything else is returned for `write_outputs`."""
    cfg = settings.parse
    did = doc_id(path)
    asset_dir = settings.resolve(settings.paths.data_dir) / "assets" / did
    asset_dir.mkdir(parents=True, exist_ok=True)
    for old in asset_dir.iterdir():  # a re-run must not leave stale PNGs behind
        old.unlink()  # empty the folder rather than delete it: OneDrive/indexers can hold the folder open
    ctx = ParseContext(did, path.name, cfg, asset_dir)
    tracer = get_tracer("mmrag.ingest")

    text_blocks: list[tuple[Element, float | None]] = []
    others: list[Element] = []
    tables: list[Table] = []
    with tracer.start_as_current_span("ingest.document") as doc_span, pymupdf.open(path) as doc:
        doc_span.set_attribute("mmrag.doc_id", did)
        doc_span.set_attribute("mmrag.source_file", path.name)
        doc_span.set_attribute("mmrag.pages", doc.page_count)
        with tracer.start_as_current_span("ingest.parse") as parse_span:
            all_blocks = [_page_blocks(page) for page in doc]
            body_size = _body_font_size([b for blocks in all_blocks for b in blocks])
            ctx.furniture = find_furniture_text(doc, cfg)

            for page, blocks in zip(doc, all_blocks):
                with tracer.start_as_current_span("parse.page") as span:
                    n = page.number + 1
                    page_tables = detect_tables(page, blocks, ctx)
                    images = extract_images(page, blocks, ctx)
                    masks = [e.bbox for e, _ in page_tables] + [e.bbox for e in images if e.status == "ok"]
                    figures = detect_vector_figures(page, blocks, masks, body_size, ctx)
                    exclude = [pymupdf.Rect(e.bbox) for e, _ in page_tables] + [pymupdf.Rect(f.bbox) for f in figures]
                    texts = extract_text_blocks(page, blocks, exclude, body_size, ctx)

                    text_blocks += texts
                    others += [e for e, _ in page_tables] + images + figures
                    tables += [t for _, t in page_tables]
                    span.set_attribute("mmrag.page", n)
                    span.set_attribute("mmrag.text_blocks", len(texts))
                    span.set_attribute("mmrag.tables", len(page_tables))
                    span.set_attribute("mmrag.images", len(images))
                    span.set_attribute("mmrag.vector_figures", len(figures))

            others += detect_textless_pages(doc, ctx)
            assign_section_paths(text_blocks, others)

            elements = sorted([e for e, _ in text_blocks] + others, key=lambda e: (e.page, e.bbox[1], e.bbox[0]))
            covered = {e.page for e in elements} | {s.page for s in ctx.skips}
            for n in range(1, doc.page_count + 1):
                if n not in covered:
                    ctx.skips.append(SkipRecord(doc_id=did, page=n, kind="empty_page", reason="no content found on page"))

            counts = Counter(e.type for e in elements)
            for etype, count in counts.items():
                parse_span.set_attribute(f"mmrag.elements.{etype}", count)
            parse_span.set_attribute("mmrag.skips", len(ctx.skips))
            parse_span.set_attribute("mmrag.rejected_regions", len(ctx.rejected))
            page_count = doc.page_count

    _log.info("parsed %s: %d elements, %d tables, %d skips", path.name, len(elements), len(tables), len(ctx.skips))
    return ParseResult(doc_id=did, source_file=path.name, page_count=page_count, elements=elements,
                       tables=tables, skips=ctx.skips, rejected=ctx.rejected)


def write_outputs(result: ParseResult, pdf_path: Path, settings: Settings) -> dict[str, Path]:
    """Write the audit files (P8) and return their paths by name:

    - `data/elements/{doc_id}/elements.jsonl`
    - `data/elements/{doc_id}/skip_log.jsonl`
    - `data/elements/{doc_id}/rejected_regions.jsonl`
    - `data/tables/{doc_id}/tables.jsonl`
    - `data/elements/{doc_id}/review_sheet.html` (via `ingest.review`)

    PNGs are already in `data/assets/{doc_id}/`, written during extraction.
    """
    from mmrag.ingest.review import build_review_sheet

    data_dir = settings.resolve(settings.paths.data_dir)
    el_dir = data_dir / "elements" / result.doc_id
    tab_dir = data_dir / "tables" / result.doc_id
    el_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)

    def _jsonl(path: Path, records) -> Path:
        path.write_text("".join(r.model_dump_json() + "\n" for r in records), encoding="utf-8")
        return path

    return {
        "elements": _jsonl(el_dir / "elements.jsonl", result.elements),
        "skip_log": _jsonl(el_dir / "skip_log.jsonl", result.skips),
        "rejected": _jsonl(el_dir / "rejected_regions.jsonl", result.rejected),
        "tables": _jsonl(tab_dir / "tables.jsonl", result.tables),
        "review_sheet": build_review_sheet(result, pdf_path, el_dir / "review_sheet.html"),
    }


# ---------------------------------------------------------------- text

def _page_blocks(page: pymupdf.Page) -> list[TextBlock]:
    blocks = []
    for b in page.get_text("dict", sort=True)["blocks"]:
        if b["type"] != 0:
            continue
        spans = [s for line in b["lines"] for s in line["spans"] if s["text"].strip()]
        if not spans:
            continue
        text = "\n".join("".join(s["text"] for s in line["spans"]).strip() for line in b["lines"]).strip()
        main = max(spans, key=lambda s: len(s["text"].strip()))
        font, flags = main["font"], main["flags"]
        blocks.append(TextBlock(
            bbox=pymupdf.Rect(b["bbox"]),
            text=text,
            size=round(main["size"], 1),
            bold=bool(flags & 16) or bool(_BOLD_FONT.search(font)),
            italic=bool(flags & 2) or bool(_ITALIC_FONT.search(font)),
            mono=bool(flags & 8) or bool(_MONO_FONT.search(font)),
        ))
    return blocks


def _normalize(text: str) -> str:
    return re.sub(r"#+", "#", re.sub(r"\d", "#", " ".join(text.lower().split())))


def _furniture_key(block: TextBlock, page_height: float, cfg: Parse) -> tuple[str, int] | None:
    """The key used to spot repeated headers/footers, or None outside the top/bottom margins."""
    margin = page_height * cfg.furniture_margin_ratio
    if block.bbox.y1 > margin and block.bbox.y0 < page_height - margin:
        return None
    return _normalize(block.text), round(block.bbox.y0 / 4)


def find_furniture_text(doc: pymupdf.Document, cfg: Parse) -> set[tuple[str, int]]:
    """Text repeated at the same vertical position, in the top or bottom margin, on more
    than `furniture_page_ratio` of the pages (running headers, footers, page numbers
    with digits normalized)."""
    if doc.page_count < _MIN_PAGES_FOR_FURNITURE:
        return set()
    pages_by_key: dict[tuple[str, int], set[int]] = defaultdict(set)
    for page in doc:
        for block in _page_blocks(page):
            key = _furniture_key(block, page.rect.height, cfg)
            if key:
                pages_by_key[key].add(page.number)
    return {k for k, pages in pages_by_key.items() if len(pages) > cfg.furniture_page_ratio * doc.page_count}


def _body_font_size(blocks: list[TextBlock]) -> float:
    """Median font size, weighted by character count (the size most text is set in)."""
    sizes = [b.size for b in blocks for _ in range(b.chars)]
    return statistics.median(sizes) if sizes else 10.0


def _is_heading(size: float, bold: bool, chars: int, body_size: float, cfg: Parse) -> bool:
    """Larger than body text by `heading_size_margin_pt`, or bold and short."""
    if chars > cfg.heading_max_chars or chars < 2:
        return False
    return size >= body_size + cfg.heading_size_margin_pt or bold


def extract_text_blocks(
    page: pymupdf.Page, blocks: list[TextBlock], exclude: list[pymupdf.Rect], body_size: float, ctx: ParseContext
) -> list[tuple[Element, float | None]]:
    """Text blocks in reading order, each paired with its font size if it is a heading
    (else None). Furniture text is dropped and logged to `ctx.skips`; text inside a
    table or figure (`exclude`) is left to that element."""
    n = page.number + 1
    out = []
    for block in blocks:
        key = _furniture_key(block, page.rect.height, ctx.cfg)
        if key and key in ctx.furniture:
            ctx.skips.append(SkipRecord(doc_id=ctx.doc_id, page=n, kind="header_footer",
                                        reason="repeated header/footer text", bbox=_bbox(block.bbox), text=block.text))
            continue
        if any(_center_in(block.bbox, r) for r in exclude):
            continue
        heading = block.size if _is_heading(block.size, block.bold, block.chars, body_size, ctx.cfg) else None
        out.append((ctx.element(n, "text", block.bbox, block.text, text=block.text), heading))
    return out


def assign_section_paths(blocks: list[tuple[Element, float | None]], others: list[Element]) -> None:
    """Fill `section_path` in place, in document order. Heading levels come from font size
    (larger = higher level). Non-text elements (`others`) take the path in force at their
    page and vertical position."""
    items = [(e.page, e.bbox[1], 0, e, size) for e, size in blocks] + [(e.page, e.bbox[1], 1, e, None) for e in others]
    stack: list[tuple[float, str]] = []
    prev: tuple[Element, float] | None = None  # the previous item, if it was a heading
    for _, _, _, element, size in sorted(items, key=lambda i: i[:3]):
        if size is not None:
            level = round(size * 2) / 2  # sizes within half a point are the same level
            title = " ".join(element.text.split())
            if (prev and prev[1] == level and prev[0].page == element.page
                    and element.bbox[1] - prev[0].bbox[3] < size):
                # The same heading wrapped onto a second line: extend it.
                stack[-1] = (level, f"{stack[-1][1]} {title}")
                prev[0].section_path = [t for _, t in stack]
            else:
                while stack and stack[-1][0] <= level:
                    stack.pop()
                stack.append((level, title))
            prev = (element, level)
        else:
            prev = None
        element.section_path = [t for _, t in stack]


# ---------------------------------------------------------------- captions

def _is_caption_like(block: TextBlock) -> bool:
    return bool(_CAPTION_START.match(block.text)) or (block.italic and block.chars >= 20)


def find_figure_caption(blocks: list[TextBlock], region: pymupdf.Rect, cfg: Parse) -> str | None:
    """The nearest "Figure N…" / "Fig. N…" or italic caption line just below the region,
    else a "Figure N…" line just above it."""
    block = _caption_block(blocks, region, cfg)
    return " ".join(block.text.split()) if block else None


def _caption_block(blocks: list[TextBlock], region: pymupdf.Rect, cfg: Parse) -> TextBlock | None:
    def overlaps_x(b: TextBlock) -> bool:
        return b.bbox.x0 < region.x1 and b.bbox.x1 > region.x0

    below = [b for b in blocks if _is_caption_like(b) and overlaps_x(b)
             and -2 <= b.bbox.y0 - region.y1 <= cfg.caption_gap_pt]
    above = [b for b in blocks if _CAPTION_START.match(b.text) and overlaps_x(b)
             and -2 <= region.y0 - b.bbox.y1 <= cfg.caption_gap_pt]
    if below:
        return min(below, key=lambda b: b.bbox.y0)
    if above:
        return max(above, key=lambda b: b.bbox.y1)
    return None


# ---------------------------------------------------------------- images

def extract_images(page: pymupdf.Page, blocks: list[TextBlock], ctx: ParseContext) -> list[Element]:
    """Embedded images at native resolution, saved as PNG. Repeats (by hash) are logged
    and skipped; images under `min_image_px` become `skipped: decorative` elements."""
    n = page.number + 1
    out = []
    for img in page.get_images(full=True):
        xref, smask = img[0], img[1]
        rects = [r & page.rect for r in page.get_image_rects(xref)]
        rects = [r for r in rects if not r.is_empty]
        if not rects:
            ctx.skips.append(SkipRecord(doc_id=ctx.doc_id, page=n, kind="undisplayed_image",
                                        reason=f"image xref {xref} is referenced but not drawn on this page"))
            continue
        pix = _image_pixmap(page.parent, xref, smask)
        png = pix.tobytes("png") if pix else None
        for rect in rects:
            if png is None:  # unreadable image data: fall back to rendering its area
                png = page.get_pixmap(dpi=ctx.cfg.render_dpi, clip=rect).tobytes("png")
            h = content_hash(png)
            if h in ctx.seen_images:
                ctx.skips.append(SkipRecord(doc_id=ctx.doc_id, page=n, kind="duplicate_image",
                                            reason=f"same image as {ctx.seen_images[h]}", bbox=_bbox(rect)))
                continue
            width, height = (pix.width, pix.height) if pix else (rect.width, rect.height)
            if min(width, height) < ctx.cfg.min_image_px:
                e = ctx.element(n, "image", rect, png, status="skipped",
                                skip_reason=f"decorative: {width}x{height} px, below min_image_px")
            else:
                e = ctx.element(n, "image", rect, png, caption=find_figure_caption(blocks, rect, ctx.cfg))
                disk, rel = ctx.asset(e.element_id)
                disk.write_bytes(png)
                e.asset_path = rel
            ctx.seen_images[h] = e.element_id
            out.append(e)
    return out


def _image_pixmap(doc: pymupdf.Document, xref: int, smask: int) -> pymupdf.Pixmap | None:
    """The image at native resolution as an RGB(A) or gray pixmap, with its soft mask applied."""
    try:
        pix = pymupdf.Pixmap(doc, xref)
        if smask:
            pix = pymupdf.Pixmap(pix, pymupdf.Pixmap(doc, smask))
        if pix.colorspace and pix.colorspace.n not in (1, 3):
            pix = pymupdf.Pixmap(pymupdf.csRGB, pix)
        return pix
    except Exception as e:  # corrupt or exotic image data
        _log.warning("image xref %d unreadable (%s); rendering its area instead", xref, e)
        return None


# ---------------------------------------------------------------- tables

def _find_tables(page: pymupdf.Page, cfg: Parse) -> list:
    return page.find_tables(strategy=cfg.table_strategy).tables


def _clean(cell) -> str:
    return " ".join(str(cell).split()) if cell is not None else ""


def _numeric_columns(columns: list[str], rows: list[list[str]]) -> list[str]:
    """Columns where at least 80% of the non-empty cells are numbers (optionally with %, k/M/B, ×)."""
    out = []
    for i, name in enumerate(columns):
        values = [r[i] for r in rows if r[i]]
        if values and sum(bool(_NUMERIC.match(v)) for v in values) >= 0.8 * len(values):
            out.append(name)
    return out


def _table_title(blocks: list[TextBlock], bbox: pymupdf.Rect, cfg: Parse) -> str | None:
    """A short bold or "Table N" line just above the table."""
    above = [b for b in blocks if 0 <= bbox.y0 - b.bbox.y1 <= cfg.caption_gap_pt and b.chars <= cfg.heading_max_chars
             and (b.bold or re.match(r"^\s*table\s*\d", b.text, re.IGNORECASE))]
    return " ".join(max(above, key=lambda b: b.bbox.y1).text.split()) if above else None


def detect_tables(page: pymupdf.Page, blocks: list[TextBlock], ctx: ParseContext) -> list[tuple[Element, Table]]:
    """PyMuPDF table finder. Runs before figure detection. A table needs a header and at
    least one row, and two non-empty columns (a single shaded box is a code block, not a
    table). Low-confidence tables are also rendered as PNG for VLM transcription."""
    n = page.number + 1
    out = []
    for tab in _find_tables(page, ctx.cfg):
        grid = [[_clean(c) for c in row] for row in tab.extract()]
        header = [_clean(c) for c in tab.header.names]
        body = grid if tab.header.external else grid[1:]
        keep = [i for i in range(len(header)) if header[i] or any(i < len(r) and r[i] for r in body)]
        header = [header[i] for i in keep]
        body = [[r[i] if i < len(r) else "" for i in keep] for r in body]
        if len(keep) < 2 or not body:
            continue

        cells = [c for r in body for c in r]
        low_confidence = (not all(header) or len(set(header)) < len(header)
                          or sum(not c for c in cells) > 0.3 * len(cells))
        bbox = pymupdf.Rect(tab.bbox)
        text = "\n".join(" | ".join(r) for r in [header, *body])
        e = ctx.element(n, "table", bbox, text, text=text)
        if low_confidence:
            disk, rel = ctx.asset(e.element_id)
            render_region(page, bbox, ctx.cfg.render_dpi, disk)
            e.asset_path = rel
        table = Table(element_id=e.element_id, columns=header, rows=body,
                      numeric_columns=_numeric_columns(header, body),
                      title=_table_title(blocks, bbox, ctx.cfg), low_confidence=low_confidence)
        out.append((e, table))
    return out


# ---------------------------------------------------------------- vector figures

def detect_vector_figures(
    page: pymupdf.Page, blocks: list[TextBlock], masks: list[BBox], body_size: float, ctx: ParseContext
) -> list[Element]:
    """The 5-step filter of spec §6.1. `masks` are table and image boxes whose drawings
    are never figures. Rejected clusters go to `ctx.rejected`."""
    cfg = ctx.cfg
    n = page.number + 1
    drawings = _mask_regions(page.get_drawings(), masks)
    drawings = _drop_page_furniture(drawings, page.rect, cfg)

    def reject(region: pymupdf.Rect, reason: str, shapes: int) -> None:
        ctx.rejected.append(RejectedRegion(doc_id=ctx.doc_id, page=n, bbox=_bbox(region), filter=reason,
                                           shape_count=shapes))

    kept = []
    for cluster in _cluster_drawings(drawings, cfg):
        reason = _classify_cluster(cluster, blocks, body_size, cfg)
        if reason:
            reject(_union([d["rect"] for d in cluster]), reason, len(cluster))
        else:
            kept.append((_figure_region(cluster, blocks, body_size, cfg), len(cluster)))

    out = []
    for region, shapes in _merge_by_caption(_merge_regions(kept), blocks, cfg):
        if max(region.width, region.height) < cfg.min_figure_pt:
            reject(region, "too_small", shapes)  # icons and bullets next to headings
            continue
        labels = [b for b in blocks if _center_in(b.bbox, region)]
        caption = find_figure_caption(blocks, region, cfg)
        region = (region + (-cfg.figure_margin_pt, -cfg.figure_margin_pt, cfg.figure_margin_pt, cfg.figure_margin_pt)) & page.rect
        e_id = ctx.next_element_id(n, "vector_figure")
        disk, rel = ctx.asset(e_id)
        render_region(page, region, cfg.render_dpi, disk)
        png = disk.read_bytes()
        out.append(Element(element_id=e_id, doc_id=ctx.doc_id, source_file=ctx.source_file, page=n,
                           bbox=_bbox(region), type="vector_figure", content_hash=content_hash(png),
                           text="\n".join(b.text for b in labels) or None, caption=caption, asset_path=rel))
    return out


def _mask_regions(drawings: list[dict], boxes: list[BBox]) -> list[dict]:
    """Step 1: drop drawing ops inside table (and image) boxes: grid lines are never figures."""
    rects = [pymupdf.Rect(b) + (-1, -1, 1, 1) for b in boxes]
    return [d for d in drawings if not any(r.contains(d["rect"]) for r in rects)]


def _drop_page_furniture(drawings: list[dict], page_rect: pymupdf.Rect, cfg: Parse) -> list[dict]:
    """Step 2: drop page backgrounds and frames, thin lines wider than `furniture_width_ratio`
    of the page, and thin lines (aspect > `line_aspect_ratio`) in the top or bottom margin."""
    width, height = page_rect.width, page_rect.height
    margin = height * cfg.furniture_margin_ratio
    kept = []
    for d in drawings:
        r = d["rect"]
        thin = min(r.width, r.height)
        if r.width * r.height >= _BACKGROUND_AREA_RATIO * width * height:
            continue
        if thin <= _THIN_LINE_MAX_PT and r.width >= cfg.furniture_width_ratio * width:
            continue
        in_margin = r.y1 <= margin or r.y0 >= height - margin
        if in_margin and max(r.width, r.height) / max(thin, 0.5) >= cfg.line_aspect_ratio:
            continue
        kept.append(d)
    return kept


def _cluster_drawings(drawings: list[dict], cfg: Parse) -> list[list[dict]]:
    """Group drawings whose boxes lie within `cluster_merge_distance_pt` of each other
    (single-link clustering), in top-to-bottom order."""
    parent = list(range(len(drawings)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(drawings)):
        for j in range(i + 1, len(drawings)):
            if _rect_distance(drawings[i]["rect"], drawings[j]["rect"]) <= cfg.cluster_merge_distance_pt:
                parent[find(i)] = find(j)
    groups: dict[int, list[dict]] = defaultdict(list)
    for i, d in enumerate(drawings):
        groups[find(i)].append(d)
    return sorted(groups.values(), key=lambda g: (min(d["rect"].y0 for d in g), min(d["rect"].x0 for d in g)))


def _classify_cluster(cluster: list[dict], blocks: list[TextBlock], body_size: float, cfg: Parse) -> str | None:
    """Steps 3–4: return the rejecting filter (`isolated_shape`, `too_few_shapes`,
    `text_callout`), or None to keep the cluster as a figure. Box-and-arrow diagrams made
    of simple shapes are kept: no complex paths are required."""
    if len(cluster) == 1 and len(cluster[0]["items"]) <= _ISOLATED_MAX_ITEMS:
        return "isolated_shape"
    if len(cluster) < cfg.min_cluster_shapes and sum(len(d["items"]) for d in cluster) <= _ISOLATED_MAX_ITEMS:
        return "too_few_shapes"
    region = _union([d["rect"] for d in cluster])
    inside = [b for b in blocks if _center_in(b.bbox, region)]
    chars = sum(b.chars for b in inside)
    body_chars = sum(b.chars for b in inside if b.mono or b.size >= body_size - 1)
    if chars and body_chars >= 0.5 * chars:
        return "text_callout"  # a box around body text or code, not a diagram
    # Boxes mostly covered by text: highlights behind words in a paragraph (body-size text,
    # any length), and code blocks or text cards (dense text in any font).
    touching = [b for b in blocks if b.bbox.intersects(region)]
    coverage = sum((b.bbox & region).get_area() for b in touching) / max(region.get_area(), 1.0)
    if coverage >= cfg.callout_text_coverage:
        touch_chars = sum(b.chars for b in touching)
        body_touch = sum(b.chars for b in touching if b.mono or b.size >= body_size - 1)
        if (touch_chars and body_touch >= 0.5 * touch_chars) or touch_chars >= cfg.callout_min_chars:
            return "text_callout"
    return None


def _is_label(block: TextBlock, body_size: float, cfg: Parse) -> bool:
    return not _is_caption_like(block) and block.size < body_size - cfg.heading_size_margin_pt


def _figure_region(cluster: list[dict], blocks: list[TextBlock], body_size: float, cfg: Parse) -> pymupdf.Rect:
    """Step 5: the cluster's box, grown to take in small-font labels within `label_attach_pt`
    (titles, axis labels, annotations), repeated until nothing more attaches."""
    region = _union([d["rect"] for d in cluster])
    labels = [b for b in blocks if _is_label(b, body_size, cfg)]
    grown = True
    while grown:
        grown = False
        for b in labels:
            if not region.contains(b.bbox) and _rect_distance(b.bbox, region) <= cfg.label_attach_pt:
                region |= b.bbox
                grown = True
    return region


def _merge_regions(regions: list[tuple[pymupdf.Rect, int]]) -> list[tuple[pymupdf.Rect, int]]:
    """Merge overlapping figure regions (e.g. chart panels sharing a title line), adding
    up their shape counts."""
    regions = [(pymupdf.Rect(r), n) for r, n in regions]
    merged = True
    while merged:
        merged = False
        for i in range(len(regions)):
            for j in range(i + 1, len(regions)):
                if regions[i][0].intersects(regions[j][0]):
                    r, n = regions.pop(j)
                    regions[i] = (regions[i][0] | r, regions[i][1] + n)
                    merged = True
                    break
            if merged:
                break
    return sorted(regions, key=lambda rn: (rn[0].y0, rn[0].x0))


def _merge_by_caption(
    regions: list[tuple[pymupdf.Rect, int]], blocks: list[TextBlock], cfg: Parse
) -> list[tuple[pymupdf.Rect, int]]:
    """One caption, one figure: panels that share the same caption line are merged."""
    by_caption: dict[int, int] = {}  # id(caption block) -> index in out
    out: list[tuple[pymupdf.Rect, int]] = []
    for region, shapes in regions:
        caption = _caption_block(blocks, region, cfg)
        if caption is not None and id(caption) in by_caption:
            i = by_caption[id(caption)]
            out[i] = (out[i][0] | region, out[i][1] + shapes)
            continue
        if caption is not None:
            by_caption[id(caption)] = len(out)
        out.append((region, shapes))
    return _merge_regions(out)  # a merged region may now overlap a neighbour


# ---------------------------------------------------------------- textless pages

def detect_textless_pages(doc: pymupdf.Document, ctx: ParseContext) -> list[Element]:
    """Pages with fewer than `textless_chars` characters, rendered whole at `render_dpi`
    as `scanned_page` elements for VLM transcription."""
    out = []
    for page in doc:
        text = page.get_text().strip()
        if len(text) >= ctx.cfg.textless_chars:
            continue
        n = page.number + 1
        e_id = ctx.next_element_id(n, "scanned_page")
        disk, rel = ctx.asset(e_id)
        render_region(page, page.rect, ctx.cfg.render_dpi, disk)
        out.append(Element(element_id=e_id, doc_id=ctx.doc_id, source_file=ctx.source_file, page=n,
                           bbox=_bbox(page.rect), type="scanned_page", content_hash=content_hash(disk.read_bytes()),
                           text=text or None, asset_path=rel))
    return out


# ---------------------------------------------------------------- rendering

def render_region(page: pymupdf.Page, region: pymupdf.Rect, dpi: int, out: Path) -> Path:
    """Render a page region (or the whole page) to PNG at `dpi`."""
    page.get_pixmap(dpi=dpi, clip=region).save(out)
    return out
