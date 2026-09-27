"""Parser against real fixture pages with hand-written expectations (LLD §10, plan Phase 1 task 10)."""

from __future__ import annotations

import json
import re
from collections import Counter

import pymupdf
import pytest

from mmrag.config import PROJECT_ROOT
from mmrag.ingest import parse

pytestmark = pytest.mark.unit

PAGES_DIR = PROJECT_ROOT / "tests" / "fixtures" / "pages"
EXPECTED = {k: v for k, v in json.loads((PAGES_DIR / "expected.json").read_text(encoding="utf-8")).items()
            if not k.startswith("_")}


@pytest.fixture(scope="module")
def results(tmp_path_factory):
    """Parse each fixture page once per test module."""
    import yaml

    from mmrag.config import load_settings
    from tests.conftest import UNIT_ENV

    cfg = yaml.safe_load((PROJECT_ROOT / "config.yaml").read_text(encoding="utf-8"))
    tmp = tmp_path_factory.mktemp("fixtures")
    cfg["paths"]["data_dir"] = str(tmp / "data")
    cfg["observability"].update(enabled=False, log_file=None)
    (tmp / "config.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
    settings = load_settings(tmp / "config.yaml", UNIT_ENV)
    return settings, {name: parse.parse_document(PAGES_DIR / f"{name}.pdf", settings) for name in EXPECTED}


@pytest.mark.parametrize("name", EXPECTED)
def test_element_counts(results, name):
    _, res = results
    ok = Counter(e.type for e in res[name].elements if e.status == "ok")
    assert {t: ok.get(t, 0) for t in EXPECTED[name]["counts"]} == EXPECTED[name]["counts"]


@pytest.mark.parametrize("name", [n for n, e in EXPECTED.items() if "figure_captions" in e])
def test_figure_captions(results, name):
    _, res = results
    figures = [e for e in res[name].elements if e.type in ("image", "vector_figure") and e.status == "ok"]
    captions = [e.caption or "" for e in figures]
    expected = EXPECTED[name]["figure_captions"]
    assert len(captions) == len(expected)
    for got, want in zip(captions, expected):
        assert got.startswith(want), (got, want)


@pytest.mark.parametrize("name", [n for n, e in EXPECTED.items() if "figure_must_cover" in e])
def test_figures_cover_all_their_parts(results, name):
    """Owner-reported parts (text cards, side notes, headers) lie inside their figure's box."""
    _, res = results
    figures = [e for e in res[name].elements if e.type == "vector_figure" and e.status == "ok"]
    expected = EXPECTED[name]["figure_must_cover"]
    assert len(figures) == len(expected)
    for fig, parts in zip(figures, expected):
        x0, y0, x1, y1 = fig.bbox
        for px0, py0, px1, py1 in parts:
            assert x0 <= px0 + 1 and y0 <= py0 + 1 and x1 >= px1 - 1 and y1 >= py1 - 1, (fig.bbox, [px0, py0, px1, py1])


@pytest.mark.parametrize("name", [n for n, e in EXPECTED.items() if "figure_must_not_cover" in e])
def test_figures_leave_page_text_alone(results, name):
    """Text boxes in the page flow next to a figure are not swallowed by it."""
    _, res = results
    figures = [e for e in res[name].elements if e.type == "vector_figure" and e.status == "ok"]
    for fig, parts in zip(figures, EXPECTED[name]["figure_must_not_cover"]):
        for part in parts:
            assert not pymupdf.Rect(fig.bbox).intersects(pymupdf.Rect(part)), (fig.bbox, part)


@pytest.mark.parametrize("name", [n for n, e in EXPECTED.items() if "table_columns" in e])
def test_table_columns(results, name):
    _, res = results
    assert [t.columns for t in res[name].tables] == EXPECTED[name]["table_columns"]
    for t in res[name].tables:
        assert t.rows and all(len(r) == len(t.columns) for r in t.rows)


@pytest.mark.parametrize("name", [n for n, e in EXPECTED.items() if "headings" in e])
def test_headings_in_section_paths(results, name):
    _, res = results
    seen = []
    for e in res[name].elements:
        for h in e.section_path:
            if h not in seen:
                seen.append(h)
    assert [h for h in seen if h in EXPECTED[name]["headings"]] == EXPECTED[name]["headings"]


@pytest.mark.parametrize("name", EXPECTED)
def test_provenance_and_assets(results, name):
    settings, res = results
    r = res[name]
    data_dir = settings.resolve(settings.paths.data_dir)
    ids = [e.element_id for e in r.elements]
    assert len(ids) == len(set(ids)), "element IDs must be unique"
    for e in r.elements:
        assert re.fullmatch(rf"{r.doc_id}:p{e.page}:{e.type}:\d+", e.element_id)
        assert e.source_file == f"{name}.pdf" and e.page == 1
        x0, y0, x1, y1 = e.bbox
        assert x0 < x1 and y0 < y1
        assert re.fullmatch(r"[0-9a-f]{64}", e.content_hash)
        if e.status == "skipped":
            assert e.skip_reason
        if e.type in ("vector_figure", "scanned_page") or (e.type == "image" and e.status == "ok"):
            assert e.asset_path and (data_dir / e.asset_path).is_file()


@pytest.mark.parametrize("name", EXPECTED)
def test_every_page_accounted_for(results, name):
    _, res = results
    r = res[name]
    pages = {e.page for e in r.elements} | {s.page for s in r.skips}
    assert pages == set(range(1, r.page_count + 1))


def test_figure_labels_are_not_body_text(results):
    """Labels inside a diagram belong to the figure, not to the text flow."""
    _, res = results
    r = res["transformers_p003"]
    body = [e.text for e in r.elements if e.type == "text"]
    assert not any("repeat until done" in t for t in body)
    figures = [e for e in r.elements if e.type == "vector_figure"]
    assert "repeat until done" in (figures[1].text or "")
