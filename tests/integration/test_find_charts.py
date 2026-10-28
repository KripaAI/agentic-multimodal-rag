"""find_charts: exact queries on planet placements read off charts, against a real database (Phase 10)."""

from __future__ import annotations

import pytest

import mmrag.db as db
from mmrag.agent.ledger import EvidenceLedger
from mmrag.agent.tools import ToolContext, find_charts
from mmrag.config import PROJECT_ROOT, load_settings
from mmrag.index.chunk import Chunk
from mmrag.index.document import LoadedDoc
from mmrag.index.writer import write_document
from mmrag.ingest.models import Element

pytestmark = pytest.mark.integration
DOC = "c" * 16

# Chart 12 of Crux of Astrology (Lagna Aries) and two made-up ones.
CHART_12 = {"kind": "Rasi chart", "lagna": "Aries", "unread": [],
            "placements": {"Lagna": "Aries", "Ketu": "Pisces", "Sun": "Capricorn", "Mercury": "Capricorn",
                           "Jupiter": "Sagittarius", "Mars": "Sagittarius", "Venus": "Sagittarius",
                           "Saturn": "Sagittarius", "Moon": "Scorpio", "Rahu": "Virgo"}}
SAT_7TH = {"kind": "Rasi chart", "lagna": "Taurus", "unread": [],
           "placements": {"Lagna": "Taurus", "Saturn": "Scorpio", "Mars": "Leo"}}       # Saturn 7th, Mars 4th
NAVAMSA = {"kind": "Navamsa (D-9) chart", "lagna": "Taurus", "unread": [],
           "placements": {"Lagna": "Taurus", "Saturn": "Scorpio"}}                     # Saturn 7th, but D-9
NO_LAGNA = {"kind": "Rasi chart", "lagna": None, "unread": [], "placements": {"Saturn": "Scorpio"}}


@pytest.fixture
def ctx(fresh_db_url, tmp_path):
    s = load_settings(PROJECT_ROOT / "config.yaml", {"DATABASE_URL": fresh_db_url})
    db.migrate(s)
    els = [Element(element_id=f"{DOC}:p{p}:image:1", doc_id=DOC, source_file="crux.pdf", page=p,
                   bbox=(0, 0, 9, 9), type="image", content_hash=f"h{p}") for p in (29, 31, 33)]
    doc = LoadedDoc(doc_id=DOC, source_file="crux.pdf", content_hash="x", elements=els,
                    charts={els[0].element_id: [CHART_12], els[1].element_id: [SAT_7TH, NAVAMSA],
                            els[2].element_id: [NO_LAGNA]})
    chunks = [Chunk(chunk_id=f"{DOC}:figure:{n}", collection="figure", element_ids=[e.element_id],
                    dense_text="a chart", keyword_text="chart") for n, e in enumerate(els, 1)]
    write_document(s, doc, chunks, [[1.0] + [0.0] * (s.embed.dims - 1)] * 3, [])
    return ToolContext(settings=s, ledger=EvidenceLedger(), embed_query=lambda q: [0.0] * s.embed.dims,
                       chart_dir=tmp_path)


def test_saturn_in_the_7th_house(ctx):
    out = find_charts(ctx, [{"body": "Saturn", "houses": [7]}])
    assert out["total"] == 1  # the Navamsa and the Lagna-less chart do not count
    (c,) = out["charts"]
    assert c["id"] == f"{DOC}:p31:image:1" and c["page"] == 31 and c["chart"] == 1
    assert "Saturn in Scorpio (7th house)" in c["placements"]
    assert ctx.ledger.has(c["id"])  # citable, and showable as an image


def test_conditions_hold_together_in_one_chart(ctx):
    both = find_charts(ctx, [{"body": "Saturn", "houses": [7]}, {"body": "Mars", "signs": ["Leo"]}])
    assert both["total"] == 1
    assert find_charts(ctx, [{"body": "Saturn", "houses": [7]}, {"body": "Mars", "houses": [7]}])["total"] == 0


def test_chart_12_by_sign_and_house(ctx):
    out = find_charts(ctx, [{"body": "Saturn", "signs": ["Sagittarius"], "houses": [9]}, {"body": "Moon", "houses": [8]}])
    assert [c["id"] for c in out["charts"]] == [f"{DOC}:p29:image:1"]


def test_divisional_charts_on_request(ctx):
    assert find_charts(ctx, [{"body": "Saturn", "houses": [7]}], kind="Navamsa (D-9) chart")["total"] == 1
    assert find_charts(ctx, [{"body": "Saturn", "houses": [7]}], kind="any")["total"] == 2
    assert find_charts(ctx, [{"body": "Saturn", "signs": ["Scorpio"]}], kind="any")["total"] == 3  # signs need no Lagna


def test_placements_go_with_their_document(ctx):
    with db.connect(ctx.settings) as conn:
        conn.execute("DELETE FROM documents WHERE doc_id = %s", (DOC,))
        conn.commit()
        assert conn.execute("SELECT count(*) FROM chart_placements").fetchone()[0] == 0


def test_new_chart_readings_rewrite_an_otherwise_unchanged_document(ctx):
    """Placements live in their own table: reading charts again must reach it even when no search
    text changed (the build hash covers the readings)."""
    from mmrag.index.writer import build_hash

    els = [Element(element_id=f"{DOC}:p29:image:1", doc_id=DOC, source_file="crux.pdf", page=29,
                   bbox=(0, 0, 9, 9), type="image", content_hash="h29")]
    plain = LoadedDoc(doc_id=DOC, source_file="crux.pdf", content_hash="x", elements=els)
    with_chart = LoadedDoc(doc_id=DOC, source_file="crux.pdf", content_hash="x", elements=els,
                           charts={els[0].element_id: [CHART_12]})
    assert build_hash(plain, [], [], "m") != build_hash(with_chart, [], [], "m")


def test_any_of_several_houses_kendras(ctx):
    """Mars in a kendra (1, 4, 7, 10): one condition, not four searches."""
    out = find_charts(ctx, [{"body": "Saturn", "houses": [7]}, {"body": "Mars", "houses": [1, 4, 7, 10]}])
    assert out["total"] == 1 and out["charts"][0]["page"] == 31  # Mars in Leo is the 4th from Taurus
