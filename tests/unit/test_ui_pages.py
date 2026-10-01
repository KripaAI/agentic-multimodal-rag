"""Source panel: PDF points -> pixels and highlighted page images (spec §7.5, LLD §5.7). Test-first."""

from __future__ import annotations

import pymupdf
import pytest

from mmrag.ui.pages import bbox_to_pixels, page_with_highlights

pytestmark = pytest.mark.unit


@pytest.fixture
def pdf(tmp_path):
    doc = pymupdf.open()
    doc.new_page(width=200, height=100)
    rotated = doc.new_page(width=200, height=100)
    rotated.set_rotation(90)
    path = tmp_path / "t.pdf"
    doc.save(path)
    return path


def test_points_become_pixels_at_the_page_dpi(pdf):
    assert bbox_to_pixels(pdf, 1, (72, 36, 144, 72), dpi=144) == pytest.approx((144, 72, 288, 144))


def test_rotated_pages_are_handled(pdf):
    x0, y0, x1, y1 = bbox_to_pixels(pdf, 2, (0, 0, 20, 10), dpi=72)
    # 90 degrees clockwise: the box's top-left corner moves to the right edge of the (now 100 wide) image
    assert (x1 - x0, y1 - y0) == pytest.approx((10, 20)) and x1 == pytest.approx(100)


def test_highlighted_page_is_a_png_with_the_box_coloured(pdf):
    png = page_with_highlights(pdf, 1, [(20, 20, 80, 60)], dpi=72)
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    pix = pymupdf.Pixmap(png)
    inside, outside = pix.pixel(50, 40), pix.pixel(150, 80)
    assert outside == (255, 255, 255) and inside != outside  # tinted inside the cited box only
