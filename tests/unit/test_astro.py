"""Phase 11: birth charts computed by code, readings written by the fine-tuned model."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timezone

import pytest

from mmrag.astro import natal
from mmrag.astro.natal import (BirthData, ChartInputError, Dasha, antardashas, ascendant, lahiri_ayanamsa,
                               nakshatra_of, ordinal, parse_birth, vimshottari)
from mmrag.astro.reading import (Written, WriterNotConfigured, check_positions, gather_passages, get_writer,
                                 resolve_citations, write_reading, Passage)
from mmrag.config import PROJECT_ROOT, load_settings


def _ephemeris_ready() -> bool:
    return (natal._ephemeris_dir() / "de440s.bsp").is_file()


needs_ephemeris = pytest.mark.skipif(not _ephemeris_ready(), reason="JPL ephemeris not downloaded")


@pytest.fixture
def settings(monkeypatch):
    env = {"DATABASE_URL": "postgresql://u:p@localhost:5432/mmrag"}
    return load_settings(PROJECT_ROOT / "config.yaml", env)


# ---------------------------------------------------------------- the calculator

def test_lahiri_ayanamsa_at_j2000():
    assert lahiri_ayanamsa(2451545.0) == pytest.approx(23.857, abs=0.002)  # 23°51'25" (Swiss Ephemeris)


def test_ascendant_is_cancer_when_aries_culminates_at_the_equator():
    assert ascendant(0.0, 0.0, 0.0, 0.4091) == pytest.approx(90.0)


def test_ordinals_and_nakshatras():
    assert [ordinal(n) for n in (1, 2, 3, 4, 11, 12, 21)] == ["1st", "2nd", "3rd", "4th", "11th", "12th", "21st"]
    assert nakshatra_of(0.0) == ("Ashwini", 1)
    assert nakshatra_of(359.99) == ("Revati", 4)
    assert nakshatra_of(13.34) == ("Bharani", 1)


def test_vimshottari_starts_from_the_moons_nakshatra_lord():
    birth = datetime(2000, 1, 1, tzinfo=timezone.utc)
    dashas, balance = vimshottari(0.0, birth)  # start of Ashwini: all of Ketu's 7 years remain
    assert dashas[0].lord == "Ketu" and balance == pytest.approx(7)
    assert [d.lord for d in dashas] == natal.DASHA_ORDER
    dashas, balance = vimshottari(13.3333333 * 1.5, birth)  # middle of Bharani (Venus, 20 years)
    assert dashas[0].lord == "Venus" and balance == pytest.approx(10, abs=0.01)
    assert dashas[1].start == dashas[0].end


def test_antardashas_fill_the_mahadasha_and_start_with_its_lord():
    md = Dasha(lord="Jupiter", start=date(2000, 1, 1), end=date(2016, 1, 1))
    subs = antardashas(md)
    assert subs[0].sub == "Jupiter" and subs[1].sub == "Saturn" and len(subs) == 9
    assert abs((subs[-1].end - md.end).days) <= 1


def test_birth_time_to_utc():
    b = BirthData(when_local=datetime(1990, 5, 17, 14, 30), tz="+05:30", latitude=0, longitude=0)
    assert b.utc() == datetime(1990, 5, 17, 9, 0, tzinfo=timezone.utc)
    b.tz = "Asia/Kolkata"
    assert b.utc() == datetime(1990, 5, 17, 9, 0, tzinfo=timezone.utc)
    b.tz = "Mars/Olympus"
    with pytest.raises(ChartInputError):
        b.utc()
    with pytest.raises(ChartInputError):
        parse_birth("17/05/1990", "14:30", "+05:30", 0, 0)


@needs_ephemeris
def test_gandhis_chart_matches_the_published_one():
    chart = natal.compute_chart(parse_birth("1869-10-02", "07:12", "+04:38", 21.6417, 69.6293))
    signs = {b: p.sign for b, p in chart.placements.items()}
    assert chart.lagna.sign == "Libra"
    assert signs == {"Sun": "Virgo", "Moon": "Cancer", "Mars": "Libra", "Mercury": "Libra", "Jupiter": "Aries",
                     "Venus": "Libra", "Saturn": "Scorpio", "Rahu": "Cancer", "Ketu": "Capricorn"}
    assert chart.placements["Jupiter"].retrograde and chart.placements["Moon"].house == 10
    assert chart.placements["Venus"].dignity == "own sign" and chart.placements["Venus"].lord_of == [1, 8]
    assert chart.dashas[0].lord == "Mercury"  # Ashlesha
    facts = chart.facts(date(1900, 1, 1))
    assert "Lagna (Ascendant): Libra" in facts
    assert "- 1st house, Libra (lord Venus): Mars, Mercury, Venus" in facts and "- 3rd house, Sagittarius (lord Jupiter): empty" in facts


@needs_ephemeris
def test_out_of_range_births_are_refused():
    with pytest.raises(ChartInputError):
        natal.compute_chart(parse_birth("1800-01-01", "12:00", "+00:00", 0, 0))
    with pytest.raises(ChartInputError):
        natal.compute_chart(parse_birth("1990-01-01", "12:00", "+00:00", 70, 0))


# ---------------------------------------------------------------- checks

def _passages(n: int) -> list[Passage]:
    return [Passage(n=i, chunk_id=f"c{i}", source_file="book.pdf", pages=[i], text=f"text {i}", query="q")
            for i in range(1, n + 1)]


def test_citations_are_renumbered_and_unknown_ones_dropped():
    text, cited, notices = resolve_citations("A [P2]. B [P1, P3]. C [P9]. D [p2][P1].", _passages(3))
    assert text == "A [2]. B [1][3]. C. D [2][1]."
    assert [p.n for p in cited] == [2, 1, 3]
    assert notices and "P9" in notices[0]


@dataclass
class _P:
    sign: str
    house: int


@dataclass
class _Chart:
    lagna: _P
    placements: dict = field(default_factory=dict)


def test_position_check_flags_only_contradictions():
    chart = _Chart(lagna=_P("Libra", 1), placements={b: _P("Aries", 7) for b in natal.BODIES})
    chart.placements["Saturn"] = _P("Scorpio", 2)
    ok = "Saturn in the 2nd house and Saturn is in Scorpio; Jupiter in the 7th house. Libra Lagna."
    assert check_positions(ok, chart) == []
    bad = check_positions("Saturn is placed in the 7th house. Mars in Leo. The Ascendant is Leo.", chart)
    assert len(bad) == 3 and "Saturn in the 7th house" in bad[0] and "Lagna is Libra" in bad[2]


# ---------------------------------------------------------------- passages and writing

@dataclass
class _Hit:
    chunk_id: str
    dense_text: str
    source_file: str = "book.pdf"
    locations: list = field(default_factory=lambda: [{"page": 3}])


def _fake_search(results: dict[str, list[str]]):
    def search(settings, query, collection, k, embed_query=None):
        return [_Hit(cid, f"passage {cid}") for cid in results[query][:k]]
    return search


def test_passages_are_taken_round_robin_without_repeats(settings):
    search = _fake_search({"a": ["1", "2", "3"], "b": ["1", "4", "5"], "c": ["6", "7", "8"]})
    got = gather_passages(settings, ["a", "b", "c"], 5, search_fn=search)
    assert [p.chunk_id for p in got] == ["1", "6", "2", "4", "7"]
    assert [p.n for p in got] == [1, 2, 3, 4, 5]
    assert gather_passages(settings, ["a"], 0, search_fn=search) == []


def test_a_single_question_gets_more_than_three_passages(settings):
    search = _fake_search({"q": [str(i) for i in range(10)]})
    assert len(gather_passages(settings, ["q"], 6, search_fn=search)) == 6


@needs_ephemeris
def test_a_reading_gives_the_writer_facts_and_passages_then_checks_it(settings):
    chart = natal.compute_chart(parse_birth("1869-10-02", "07:12", "+04:38", 21.6417, 69.6293))
    seen = {}

    def write(messages):
        seen["messages"] = messages
        return Written(text="Libra Lagna [P1]. Saturn in the 7th house [P2][P40].", model="fake",
                       input_tokens=10, output_tokens=5)

    search = lambda s, q, c, k, embed_query=None: [_Hit(f"{q}-{i}", f"about {q}") for i in range(k)]  # noqa: E731
    r = write_reading(settings, chart, write=write, search_fn=search, on=date(1900, 1, 1))
    system, user = seen["messages"][0]["content"], seen["messages"][1]["content"]
    assert "Jyotishi" in system  # the training prompt
    assert "Lagna (Ascendant): Libra" in user and "[P1]" in user and "never recalculate" in user
    assert r.text == "Libra Lagna [1]. Saturn in the 7th house [2]."
    assert len(r.passages) == settings.writer.passages and [p.n for p in r.cited] == [1, 2]
    assert any("P40" in n for n in r.notices) and any("Saturn in the 7th house" in n for n in r.notices)


def test_writer_settings_come_from_the_environment(settings):
    assert settings.secrets.writer_base_url is None
    with pytest.raises(WriterNotConfigured):
        get_writer(settings, "astro")
    env = {"DATABASE_URL": "postgresql://u:p@localhost:5432/mmrag", "WRITER_BASE_URL": "https://x.modal.run/v1",
           "WRITER_API_KEY": "k"}
    s = load_settings(PROJECT_ROOT / "config.yaml", env)
    assert s.secrets.writer_base_url == "https://x.modal.run/v1" and s.secrets.writer_api_key.get_secret_value() == "k"
    assert callable(get_writer(s, "astro"))


def test_bare_citations_count_too():
    text, cited, _ = resolve_citations("This is covered in P5, and P2 agrees.", _passages(6))
    assert text == "This is covered in passage [5], and passage [2] agrees." and [p.n for p in cited] == [5, 2]


@needs_ephemeris
def test_the_checker_catches_the_errors_of_the_first_live_reading():
    """Sentences from the first reading by astro-v2-8b (2026-10-12), for the chart above."""
    chart = natal.compute_chart(parse_birth("1869-10-02", "07:12", "+04:38", 21.6417, 69.6293))
    text = ("Rahu and Ketu are also in the 1st, which can make the native restless. The presence of Rahu and Ketu "
            "in the 10th adds ambition. It is aspected by Saturn, lord of the 7th. Saturn is debilitated, though. "
            "The native is running Rahu mahadasha (1927-06-20 to 1940-05-14). Venus is flanked by Mercury (9th and "
            "12th lord) and Mars (2nd and 7th lord). The 10th lord Moon is in its own sign. Jupiter is retrograde "
            "and in Aries. Rahu mahadasha (1922-05-15 to 1940-05-14) with Saturn antardasha (1927-06-20 to 1930-04-26).")
    notes = " ".join(check_positions(text, chart))
    for wrong in ("Rahu in the 1st", "Ketu in the 1st", "Ketu in the 10th", "Saturn debilitated",
                  "Saturn lord of the 7th", "Rahu mahadasha as 1927-06-20"):
        assert wrong in notes, wrong
    assert len(check_positions(text, chart)) == 6  # the true statements raise nothing


@needs_ephemeris
def test_the_checker_catches_wrong_dasha_years_and_house_kinds():
    """From the second live reading (2026-10-12)."""
    chart = natal.compute_chart(parse_birth("1869-10-02", "07:12", "+04:38", 21.6417, 69.6293))
    text = ("The native is running Rahu mahadasha (1922-1940). The Sun mahadasha follows from 1940 to 1956. "
            "The 4th is a trikona (trine). The 7th house is a kendra.")
    notes = check_positions(text, chart)
    assert len(notes) == 2
    assert "Sun mahadasha as 1940-1956; the chart has 1899-1905" in notes[0] and "4th house a trikona" in notes[1]


def test_warm_up_loads_a_local_model_but_not_an_api_one(monkeypatch):
    from mmrag.index import embed

    loads = []
    monkeypatch.setattr(embed, "load_local_model", lambda model, dims: loads.append(model) or object())
    e = embed.Embedder.__new__(embed.Embedder)
    e.provider, e.model, e.dims, e._local = "local", "m", 768, None
    e.warm_up()
    e.warm_up()
    assert loads == ["m"]  # loaded once, kept
    e.provider, e._local = "openai", None
    e.warm_up()
    assert loads == ["m"]


def test_a_blocked_scikit_learn_is_replaced_by_a_stand_in(monkeypatch):
    import builtins
    import sys

    from mmrag.index import embed

    real_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name == "sklearn.metrics" or name.startswith("sklearn"):
            raise ImportError("DLL load failed: An Application Control policy has blocked this file.")
        return real_import(name, *args, **kwargs)

    saved = {k: v for k, v in sys.modules.items() if k == "sklearn" or k.startswith("sklearn.")}
    for k in saved:
        monkeypatch.delitem(sys.modules, k)
    monkeypatch.setattr(builtins, "__import__", blocked)
    try:
        embed._without_blocked_sklearn()
    finally:
        monkeypatch.setattr(builtins, "__import__", real_import)
    stand_in = sys.modules["sklearn.metrics"]
    assert callable(stand_in.pairwise_distances) and callable(stand_in.f1_score)
    with pytest.raises(AttributeError):
        stand_in.__file__  # noqa: B018
    with pytest.raises(RuntimeError):
        stand_in.roc_curve([1], [0])
    for k in [k for k in sys.modules if k == "sklearn" or k.startswith("sklearn.")]:
        monkeypatch.delitem(sys.modules, k)
    sys.modules.update(saved)


def test_questions_about_the_askers_own_chart_are_recognised():
    from mmrag.astro.reading import asks_about_own_chart

    assert asks_about_own_chart("My name is kripa shankar and my date of birth is 04-01-1983.. can u tell me "
                                "something about my future?")
    assert asks_about_own_chart("I was born 1990-05-17 at 14:30 in Delhi, what does my chart say?")
    assert asks_about_own_chart("what is my lagna")
    assert not asks_about_own_chart("What does Saturn in the 7th house give?")
    assert not asks_about_own_chart("Which charts in the books have Mars in Leo?")
