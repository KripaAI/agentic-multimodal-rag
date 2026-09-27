"""Whole-PDF parse results must match the owner-approved snapshots (W5).

Skipped when the source PDF is absent (it is not in git) or has changed content.
A failure means detection changed: check the review sheet, and if the owner approves
the new result, refresh with `python -m tests.regression.snapshot`.
"""

from __future__ import annotations

import json

import pytest

from mmrag.config import PROJECT_ROOT
from mmrag.ingest.ids import doc_id
from mmrag.ingest.parse import parse_document
from tests.regression.snapshot import DOCUMENTS, SNAPSHOT_DIR, snapshot

pytestmark = pytest.mark.regression

BBOX_TOLERANCE_PT = 2.0


@pytest.fixture(scope="module", params=list(DOCUMENTS.items()), ids=list(DOCUMENTS.values()))
def parsed(request, tmp_path_factory):
    pdf_name, snap_name = request.param
    pdf = PROJECT_ROOT / "data" / "pdfs" / pdf_name
    expected = json.loads((SNAPSHOT_DIR / f"{snap_name}.json").read_text(encoding="utf-8"))
    if not pdf.is_file():
        pytest.skip(f"{pdf_name} not in data/pdfs")
    if doc_id(pdf) != expected["doc_id"]:
        pytest.skip(f"{pdf_name} content changed since the snapshot was approved")

    import yaml

    from mmrag.config import load_settings
    from tests.conftest import UNIT_ENV

    cfg = yaml.safe_load((PROJECT_ROOT / "config.yaml").read_text(encoding="utf-8"))
    tmp = tmp_path_factory.mktemp(snap_name)
    cfg["paths"]["data_dir"] = str(tmp / "data")
    cfg["observability"].update(enabled=False, log_file=None)
    (tmp / "config.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
    settings = load_settings(tmp / "config.yaml", UNIT_ENV)
    return snapshot(parse_document(pdf, settings)), expected


def test_counts_match(parsed):
    got, expected = parsed
    assert got["counts"] == expected["counts"]
    assert got["skips"] == expected["skips"]


def test_figures_images_and_tables_match(parsed):
    got, expected = parsed
    assert [(i["page"], i["type"]) for i in got["items"]] == [(i["page"], i["type"]) for i in expected["items"]]
    for g, e in zip(got["items"], expected["items"]):
        where = f"p{e['page']} {e['type']}"
        assert all(abs(a - b) <= BBOX_TOLERANCE_PT for a, b in zip(g["bbox"], e["bbox"])), (where, g["bbox"], e["bbox"])
        assert g.get("caption") == e.get("caption"), where
        assert g.get("columns") == e.get("columns"), where
