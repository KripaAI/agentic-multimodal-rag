"""Source panel (spec §7.5, LLD §5.7): the cited PDF page with the element highlighted."""

from __future__ import annotations

from pathlib import Path

import pymupdf

HIGHLIGHT = (1.0, 0.8, 0.0)  # amber, drawn semi-transparent


def bbox_to_pixels(pdf: Path, page: int, bbox, dpi: int) -> tuple[float, float, float, float]:
    """A bbox in PDF points (origin top-left, unrotated page) as pixels on the rendered page image:
    rotation applied, then scaled by dpi/72."""
    with pymupdf.open(pdf) as doc:
        p = doc[page - 1]
        r = pymupdf.Rect(bbox) * p.rotation_matrix * pymupdf.Matrix(dpi / 72, dpi / 72)
        return (r.x0, r.y0, r.x1, r.y1)


def page_with_highlights(pdf: Path, page: int, bboxes, dpi: int) -> bytes:
    """PNG of the page at `dpi` with each bbox tinted. The PDF on disk is never modified."""
    with pymupdf.open(pdf) as doc:
        p = doc[page - 1]
        for bbox in bboxes:
            p.draw_rect(pymupdf.Rect(bbox), color=HIGHLIGHT, fill=HIGHLIGHT, fill_opacity=0.3, width=1.5)
        return p.get_pixmap(dpi=dpi).tobytes("png")  # rendering applies the page rotation
