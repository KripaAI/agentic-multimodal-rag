"""Planet placements from South Indian charts (Phase 10): grid geometry, labels, houses, chart type.
A drawn grid and a fake OCR stand in for real charts; the cell geometry matches Crux of Astrology's
(thin outer cells, a wide middle band)."""

from __future__ import annotations

import numpy as np
import pytest

from mmrag.ingest.charts import ChartReading, chart_kind, grid_frame, read_chart, split_label

pytestmark = pytest.mark.unit

W = H = 360
XS = YS = [2, 66, 175, 284, 356]  # frame, inner, (midpoint), inner, frame: outer cells are thin


def _grid() -> np.ndarray:
    img = np.full((H, W), 255, np.uint8)
    for v in (2, 66, 284, 356):  # full-length lines
        img[v - 1:v + 2, :] = 0
        img[:, v - 1:v + 2] = 0
    img[174:177, 2:66] = img[174:177, 284:356] = 0  # the side cells' middle lines (centre box open)
    img[2:66, 174:177] = img[284:356, 174:177] = 0
    return img


def _at(row: int, col: int, text: str):
    """A fake OCR word centred in grid cell (row, col), in the 3x upscaled coordinates."""
    cx, cy = (XS[col] + XS[col + 1]) / 2 * 3, (YS[row] + YS[row + 1]) / 2 * 3
    return [[(cx - 5, cy - 5), (cx + 5, cy - 5), (cx + 5, cy + 5), (cx - 5, cy + 5)], text, 0.99]


def test_the_grid_is_found_with_its_real_boundaries():
    cols, rows = grid_frame(_grid())
    assert [round(v) for v in cols] == XS and [round(v) for v in rows] == YS


def test_a_blank_image_or_a_table_is_not_a_chart():
    assert grid_frame(np.full((H, W), 255, np.uint8)) is None
    table = np.full((H, W), 255, np.uint8)
    for v in range(2, 357, 50):  # an even grid of many lines
        table[v, :] = table[:, v] = 0
    assert grid_frame(table) is None


def test_chart_12_of_crux_of_astrology_is_read_exactly():
    words = [_at(0, 0, "KETU"), _at(0, 1, "LAG"), _at(1, 1, "CHART 12"), _at(2, 0, "MERC"), _at(2, 0, "SUN"),
             _at(2, 3, "A.L."), _at(3, 0, "JUP"), _at(3, 0, "MARS"), _at(3, 0, "VENSAT"), _at(3, 1, "MOON"),
             _at(3, 3, "RAHU")]
    r = read_chart(_grid(), lambda img: words)
    assert r.style == "south_indian" and r.kind == "Rasi chart" and r.lagna == "Aries"
    assert r.placements["Saturn"] == "Sagittarius" and r.house("Saturn") == 9
    assert (r.house("Moon"), r.house("Sun"), r.house("Rahu"), r.house("Ketu")) == (8, 10, 6, 12)
    assert r.unread == ["A.L."]
    assert r.text() == ("Rasi chart, Lagna Aries. Rahu in Virgo (6th house); Moon in Scorpio (8th house); "
                        "Mars, Jupiter, Venus, Saturn in Sagittarius (9th house); Sun, Mercury in Capricorn "
                        "(10th house); Ketu in Pisces (12th house).")


def test_without_a_lagna_mark_there_are_signs_but_no_houses():
    r = read_chart(_grid(), lambda img: [_at(0, 1, "SAT")])
    assert r.placements == {"Saturn": "Aries"} and r.house("Saturn") is None
    assert r.text() == "Rasi chart, Lagna not marked. Saturn in Aries."


@pytest.mark.parametrize("token,bodies", [
    ("VENSAT", ["Venus", "Saturn"]), ("JUPRX", ["Jupiter"]), ("JUPRY", ["Jupiter"]), ("NOOW", ["Moon"]), ("SUP", ["Jupiter"]), ("BAT", ["Saturn"]),
    ("JUPITER", ["Jupiter"]), ("SAT 12°", ["Saturn"]), ("A.L.", []), ("UL", []), ("A7", []), ("MALE", []),
])
def test_labels(token, bodies):
    assert split_label(token) == bodies


def test_the_chart_type_comes_from_the_centre():
    assert chart_kind("VIMSAMSA (D-20) CHART") == "Vimsamsa (D-20) chart"  # was reported as Rasi
    assert chart_kind("D-9") == chart_kind("NAVAMSA") == "Navamsa (D-9) chart"
    assert chart_kind("CHART 12 MALE 23:01:1960 12:32 PM IST") == "Rasi chart"
    assert chart_kind("ROAD 9") == "Rasi chart"


def test_a_non_chart_reads_as_nothing():
    r = ChartReading(style="not_south_indian")
    assert r.text() == "" and r.house("Sun") is None


def test_a_charts_figure_document_carries_its_placements(parse_settings, tmp_path):
    import json

    from mmrag.index.chunk import build_figure_docs
    from mmrag.index.document import load_document
    from mmrag.ingest.ids import doc_id
    from mmrag.ingest.models import Element

    pdf = tmp_path / "book.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")
    did = doc_id(pdf)
    data = parse_settings.resolve(parse_settings.paths.data_dir)
    figs = [Element(element_id=f"{did}:p29:image:{n}", doc_id=did, source_file="book.pdf", page=29,
                    bbox=(0, 0, 9, 9), type="image", caption="Chart 12", content_hash=f"h{n}", asset_path="a.png")
            for n in (1, 2)]
    (data / "elements" / did).mkdir(parents=True)
    (data / "elements" / did / "elements.jsonl").write_text("".join(e.model_dump_json() + "\n" for e in figs),
                                                           encoding="utf-8")
    (data / "captions" / did).mkdir(parents=True)
    (data / "captions" / did / "charts.jsonl").write_text("\n".join(json.dumps(r) for r in [
        {"element_id": f"{did}:p29:image:1", "image_hash": "h1", "text": "Rasi chart, Lagna Aries. Saturn in Sagittarius (9th house)."},
        {"element_id": f"{did}:p29:image:2", "image_hash": "older", "text": "Rasi chart, Lagna Leo."},  # image since changed
    ]) + "\n", encoding="utf-8")
    docs = {c.element_ids[0]: c.dense_text for c in build_figure_docs(load_document(pdf, parse_settings), [])}
    assert "Placements: Rasi chart, Lagna Aries. Saturn in Sagittarius (9th house)." in docs[f"{did}:p29:image:1"]
    assert "Placements" not in docs[f"{did}:p29:image:2"]
