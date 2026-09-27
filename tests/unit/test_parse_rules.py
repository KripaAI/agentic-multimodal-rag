"""Deterministic parsing rules on synthetic inputs (LLD §3.2, spec §6.1). Written test-first."""

from __future__ import annotations

import json

import pymupdf
import pytest

from mmrag.ingest import parse
from mmrag.ingest.models import Element
from mmrag.ingest.parse import TextBlock

pytestmark = pytest.mark.unit

PAGE = pymupdf.Rect(0, 0, 600, 800)


# ---------------------------------------------------------------- helpers

def _drawing(x0, y0, x1, y1, kind="re", fill=(0.5, 0.5, 0.5)):
    r = pymupdf.Rect(x0, y0, x1, y1)
    return {"rect": r, "items": [(kind, r)], "type": "f", "fill": fill, "color": None}


def _block(x0, y0, x1, y1, text="label", size=7.0, bold=False, italic=False):
    return TextBlock(pymupdf.Rect(x0, y0, x1, y1), text, size, bold, italic)


def _text_element(n, text, page=1, y=0.0):
    return Element(
        element_id=f"d:p{page}:text:{n}", doc_id="d", source_file="x.pdf", page=page,
        bbox=(0, y, 100, y + 10), type="text", text=text, content_hash="h",
    )


def _pdf_with_pages(tmp_path, pages: list[list[tuple[float, float, str, float]]], name="doc.pdf"):
    """Build a PDF; each page is a list of (x, y, text, fontsize)."""
    doc = pymupdf.open()
    for items in pages:
        page = doc.new_page(width=600, height=800)
        for x, y, text, size in items:
            page.insert_text((x, y), text, fontsize=size)
    path = tmp_path / name
    doc.save(path)
    return path


def _png(w, h, gray=200):
    pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, w, h), False)
    pix.clear_with(gray)
    return pix.tobytes("png")


BODY = [(50, 100 + 14 * i, "Body text line number %d with enough words to be a paragraph." % i, 10) for i in range(12)]


# ---------------------------------------------------------------- headings and sections

@pytest.mark.parametrize(
    "size, bold, chars, expected",
    [
        (14.0, False, 20, True),    # clearly larger than body
        (11.0, False, 20, False),   # within the margin
        (10.8, True, 30, True),     # bold and short
        (10.8, True, 300, False),   # a bold paragraph is not a heading
    ],
)
def test_is_heading(parse_settings, size, bold, chars, expected):
    assert parse._is_heading(size, bold, chars, 10.8, parse_settings.parse) is expected


def test_section_paths_follow_heading_levels():
    blocks = [
        (_text_element(1, "Chapter 1", y=10), 20.0),
        (_text_element(2, "intro", y=40), None),
        (_text_element(3, "1.1 Setup", y=60), 14.0),
        (_text_element(4, "setup text", y=80), None),
        (_text_element(5, "Note", y=100), 10.8),       # bold run-in heading: deepest level
        (_text_element(6, "note text", y=120), None),
        (_text_element(7, "1.2 Use", y=140), 14.0),    # sibling: replaces 1.1 and Note
        (_text_element(8, "use text", y=160), None),
        (_text_element(1, "Chapter 2", page=2, y=10), 20.0),
        (_text_element(2, "more", page=2, y=40), None),
    ]
    figure = _text_element(9, None, y=150).model_copy(update={"type": "image", "text": None})
    parse.assign_section_paths(blocks, [figure])
    paths = [e.section_path for e, _ in blocks]
    assert paths == [
        ["Chapter 1"],
        ["Chapter 1"],
        ["Chapter 1", "1.1 Setup"],
        ["Chapter 1", "1.1 Setup"],
        ["Chapter 1", "1.1 Setup", "Note"],
        ["Chapter 1", "1.1 Setup", "Note"],
        ["Chapter 1", "1.2 Use"],
        ["Chapter 1", "1.2 Use"],
        ["Chapter 2"],
        ["Chapter 2"],
    ]
    assert figure.section_path == ["Chapter 1", "1.2 Use"]


def test_heading_wrapped_over_two_blocks_is_one_heading():
    blocks = [
        (_text_element(1, "Chapter 6: The Reference Corpus and", y=100).model_copy(update={"bbox": (0, 100, 400, 118)}), 15.0),
        (_text_element(2, "Ingestion Staging", y=120).model_copy(update={"bbox": (0, 120, 200, 138)}), 15.0),
        (_text_element(3, "body", y=160), None),
        (_text_element(4, "Chapter 7: Next", y=400).model_copy(update={"bbox": (0, 400, 200, 418)}), 15.0),
    ]
    parse.assign_section_paths(blocks, [])
    title = "Chapter 6: The Reference Corpus and Ingestion Staging"
    assert [e.section_path for e, _ in blocks] == [[title], [title], [title], ["Chapter 7: Next"]]


# ---------------------------------------------------------------- headers and footers

def test_repeated_header_and_footer_are_furniture(tmp_path, parse_settings):
    pages = [[(50, 30, "My Book Title", 9), *BODY, (280, 780, f"Page {n}", 9)] for n in range(1, 5)]
    doc = pymupdf.open(_pdf_with_pages(tmp_path, pages))
    furniture = parse.find_furniture_text(doc, parse_settings.parse)
    texts = {t for t, _ in furniture}
    assert "my book title" in texts
    assert "page #" in texts  # digits normalized, so every page number matches
    assert not any("body text" in t for t in texts)


def test_single_page_document_has_no_furniture(tmp_path, parse_settings):
    doc = pymupdf.open(_pdf_with_pages(tmp_path, [[(50, 30, "Title", 9), *BODY]]))
    assert parse.find_furniture_text(doc, parse_settings.parse) == set()


def test_furniture_text_is_removed_and_logged(tmp_path, parse_settings):
    pages = [[(50, 30, "My Book Title", 9), *BODY, (280, 780, f"Page {n}", 9)] for n in range(1, 5)]
    result = parse.parse_document(_pdf_with_pages(tmp_path, pages), parse_settings)
    texts = " ".join(e.text or "" for e in result.elements)
    assert "My Book Title" not in texts and "Page 3" not in texts
    kinds = [s.kind for s in result.skips]
    assert kinds.count("header_footer") == 8
    assert all(s.reason for s in result.skips)


# ---------------------------------------------------------------- vector figure filters

def test_mask_regions_drops_drawings_inside_tables():
    inside = _drawing(110, 110, 150, 120)
    outside = _drawing(300, 300, 350, 340)
    kept = parse._mask_regions([inside, outside], [(100, 100, 200, 200)])
    assert kept == [outside]


def test_page_furniture_filter(parse_settings):
    wide_rule = _drawing(20, 400, 580, 401, "l")         # > 85% of page width
    top_rule = _drawing(50, 20, 350, 21, "l")            # thin line in the top margin
    background = _drawing(40, 40, 560, 760)              # page background
    box = _drawing(100, 300, 200, 340)
    arrow_shaft = _drawing(200, 320, 260, 321, "l")      # thin, but mid-page
    kept = parse._drop_page_furniture([wide_rule, top_rule, background, box, arrow_shaft], PAGE, parse_settings.parse)
    assert kept == [box, arrow_shaft]


def test_clustering_groups_nearby_shapes(parse_settings):
    a = _drawing(100, 100, 150, 130)
    arrow = _drawing(153, 114, 170, 116, "l")
    b = _drawing(173, 100, 223, 130)
    far = _drawing(100, 400, 150, 430)
    clusters = parse._cluster_drawings([a, arrow, b, far], parse_settings.parse)
    assert sorted(len(c) for c in clusters) == [1, 3]


def test_single_shape_is_isolated(parse_settings):
    assert parse._classify_cluster([_drawing(100, 100, 400, 101, "l")], [], 10.8, parse_settings.parse) == "isolated_shape"


def test_two_shapes_are_too_few(parse_settings):
    cluster = [_drawing(100, 100, 150, 130), _drawing(152, 100, 200, 130)]
    assert parse._classify_cluster(cluster, [], 10.8, parse_settings.parse) == "too_few_shapes"


def test_box_with_body_text_is_a_callout(parse_settings):
    cluster = [_drawing(50, 100, 550, 200), _drawing(50, 100, 53, 200), _drawing(50, 100, 550, 101, "l")]
    blocks = [_block(60, 110, 540, 190, "IMPORTANT The opposite meaning case is an idealization." * 3, size=10.8)]
    assert parse._classify_cluster(cluster, blocks, 10.8, parse_settings.parse) == "text_callout"


def test_code_block_in_small_font_is_a_callout(parse_settings):
    """Code set in a small font with no usable name: recognized by how densely text fills the box."""
    cluster = [_drawing(50, 100, 550, 300), _drawing(50, 100, 53, 300), _drawing(50, 100, 550, 101, "l")]
    code = "\n".join(f"result = transform(store, asset, extractor, model_id, params) # {i}" for i in range(12))
    blocks = [_block(60, 105, 540, 295, code, size=8.3)]
    assert parse._classify_cluster(cluster, blocks, 10.5, parse_settings.parse) == "text_callout"


def test_inline_code_highlights_are_callouts(parse_settings):
    """Shaded boxes behind words inside a paragraph are text decoration, not a figure."""
    cluster = [_drawing(311, 589, 338, 601), _drawing(343, 589, 370, 601), _drawing(375, 589, 401, 601)]
    line = _block(57, 588, 504, 600, 'A file such as .mp4 / .avi / .mkv is a container', size=10.5)  # one short line
    assert parse._classify_cluster(cluster, [line], 10.5, parse_settings.parse) == "text_callout"


def test_panels_sharing_one_caption_are_one_figure(tmp_path, parse_settings):
    doc = pymupdf.open()
    page = doc.new_page(width=600, height=800)
    for x, y, text, size in BODY:
        page.insert_text((x, y), text, fontsize=size)
    for x0 in (60, 330):  # two panels, 130 pt apart: too far to cluster
        for dx in (0, 70, 140):
            page.draw_rect(pymupdf.Rect(x0 + dx, 300, x0 + dx + 60, 340), fill=(0.2, 0.4, 0.8))
    page.insert_text((60, 365), "Figure 1: two panels that belong together, described by a single caption line.",
                     fontsize=9)
    path = tmp_path / "panels.pdf"
    doc.save(path)
    figures = [e for e in parse.parse_document(path, parse_settings).elements if e.type == "vector_figure"]
    assert len(figures) == 1
    assert figures[0].caption.startswith("Figure 1:")
    assert figures[0].bbox[0] < 60 and figures[0].bbox[2] > 530


def test_small_drawing_is_rejected_as_too_small(tmp_path, parse_settings):
    doc = pymupdf.open()
    page = doc.new_page(width=600, height=800)
    for x, y, text, size in BODY:
        page.insert_text((x, y), text, fontsize=size)
    for x in (100, 112, 124):  # an icon-sized cluster of three shapes
        page.draw_rect(pymupdf.Rect(x, 500, x + 10, 510), fill=(0.2, 0.4, 0.8))
    path = tmp_path / "icon.pdf"
    doc.save(path)
    result = parse.parse_document(path, parse_settings)
    assert not [e for e in result.elements if e.type == "vector_figure"]
    assert [r.filter for r in result.rejected] == ["too_small"]


def test_box_and_arrow_diagram_is_kept(parse_settings):
    cluster = [
        _drawing(60, 430, 160, 466), _drawing(162, 448, 181, 449, "l"),
        _drawing(182, 430, 283, 466), _drawing(284, 448, 303, 449, "l"),
        _drawing(305, 430, 406, 466),
    ]
    blocks = [_block(97, 437, 123, 458, "1: Input"), _block(212, 437, 253, 458, "2: Compute"),
              _block(341, 437, 369, 458, "3: Select")]
    assert parse._classify_cluster(cluster, blocks, 10.8, parse_settings.parse) is None


# ---------------------------------------------------------------- tables

def test_numeric_columns():
    columns = ["Model", "Params", "Share", "Notes"]
    rows = [["A", "7B", "41%", "fast"], ["B", "1,196", "0.85", "slow"], ["C", "", "≈ 2.15", "ok"]]
    assert parse._numeric_columns(columns, rows) == ["Params", "Share"]


# ---------------------------------------------------------------- pages and images

def test_textless_page_becomes_scanned_page(tmp_path, parse_settings):
    doc = pymupdf.open()
    page = doc.new_page(width=600, height=800)
    page.insert_text((50, 50), "Hi", fontsize=10)
    page.insert_image(pymupdf.Rect(50, 100, 550, 700), stream=_png(400, 480))
    path = tmp_path / "scan.pdf"
    doc.save(path)
    result = parse.parse_document(path, parse_settings)
    scanned = [e for e in result.elements if e.type == "scanned_page"]
    assert len(scanned) == 1
    assert (parse_settings.resolve(parse_settings.paths.data_dir) / scanned[0].asset_path).is_file()


def test_small_image_is_skipped_as_decorative(tmp_path, parse_settings):
    doc = pymupdf.open()
    page = doc.new_page(width=600, height=800)
    for x, y, text, size in BODY:
        page.insert_text((x, y), text, fontsize=size)
    page.insert_image(pymupdf.Rect(50, 400, 60, 410), stream=_png(20, 20))
    path = tmp_path / "icon.pdf"
    doc.save(path)
    images = [e for e in parse.parse_document(path, parse_settings).elements if e.type == "image"]
    assert len(images) == 1
    assert images[0].status == "skipped"
    assert "decorative" in images[0].skip_reason


def test_repeated_image_is_kept_once(tmp_path, parse_settings):
    doc = pymupdf.open()
    png = _png(300, 200)
    for _ in range(2):
        page = doc.new_page(width=600, height=800)
        for x, y, text, size in BODY:
            page.insert_text((x, y), text, fontsize=size)
        page.insert_image(pymupdf.Rect(50, 400, 350, 600), stream=png)
    path = tmp_path / "dup.pdf"
    doc.save(path)
    result = parse.parse_document(path, parse_settings)
    assert len([e for e in result.elements if e.type == "image" and e.status == "ok"]) == 1
    assert [s.kind for s in result.skips] == ["duplicate_image"]


def test_reparse_replaces_assets(tmp_path, parse_settings):
    """Parsing the same PDF again works and leaves no stale PNGs (P8: re-runnable)."""
    doc = pymupdf.open()
    page = doc.new_page(width=600, height=800)
    page.insert_text((50, 50), "Hi", fontsize=10)
    path = tmp_path / "again.pdf"
    doc.save(path)
    first = parse.parse_document(path, parse_settings)
    asset_dir = parse_settings.resolve(parse_settings.paths.data_dir) / "assets" / first.doc_id
    (asset_dir / "stale.png").write_bytes(b"old")
    second = parse.parse_document(path, parse_settings)
    assert [e.element_id for e in second.elements] == [e.element_id for e in first.elements]
    assert sorted(p.name for p in asset_dir.iterdir()) == ["p1_scanned_page_1.png"]


# ---------------------------------------------------------------- outputs

def test_write_outputs(tmp_path, parse_settings):
    pages = [[*BODY, (50, 400, "Heading Two", 16), *[(x, y + 320, t, s) for x, y, t, s in BODY]]]
    path = _pdf_with_pages(tmp_path, pages)
    result = parse.parse_document(path, parse_settings)
    files = parse.write_outputs(result, path, parse_settings)
    for key in ("elements", "skip_log", "tables", "review_sheet"):
        assert files[key].is_file(), key
    lines = files["elements"].read_text(encoding="utf-8").splitlines()
    assert len(lines) == len(result.elements) > 0
    assert json.loads(lines[0])["element_id"] == result.elements[0].element_id
    assert "<html" in files["review_sheet"].read_text(encoding="utf-8").lower()
