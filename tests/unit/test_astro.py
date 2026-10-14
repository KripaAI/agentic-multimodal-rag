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
def test_a_career_answer_gets_the_10th_house_facts_and_is_repaired_without_them(settings):
    """Review 2026-10-13: a career answer for a child built on the 5th house and never looked at the 10th."""
    chart = natal.compute_chart(parse_birth("2020-04-20", "16:46", "Asia/Kolkata", 28.61, 77.21))
    calls = []
    replies = ["**Answer**\nA bright, curious child.\n\n**Why the chart says this**\nThe 5th house is strong.",
               "**Answer**\nA bright, curious child.\n\n**Why the chart says this**\nThe 10th house is Gemini, "
               "with Rahu; its lord Mercury sits in the 7th."]

    def write(messages):
        calls.append(messages)
        return Written(text=replies[len(calls) - 1], model="fake")

    r = write_reading(settings, chart, question="How will his career be?", write=write, use_books=False,
                      on=date(2026, 10, 13))
    first_request = calls[0][-1]["content"]
    assert "**Career factors**" in first_request and "begin with the 10th house and its lord" in first_request
    assert "describe aptitudes and interests to notice and nurture" in first_request  # he is six
    assert len(calls) == 2 and "does not discuss the 10th house" in calls[1][-1]["content"]
    assert r.body.startswith("**Career factors** _(calculated by software)_") and "The 10th house is Gemini" in r.body
    assert not any("does not discuss the 10th house" in n for n in r.notices)


@needs_ephemeris
def test_a_reading_gives_the_writer_facts_and_passages_then_checks_and_repairs_it(settings):
    chart = natal.compute_chart(parse_birth("1869-10-02", "07:12", "+04:38", 21.6417, 69.6293))
    calls = []
    replies = ["**Chart overview**\nThe model's own overview.\n\n**Classical view**\nLibra Lagna [P1]. Saturn in the "
               "7th house [P2][P40].",
               "**Classical view**\nLibra Lagna [P1]. Saturn in the 2nd house [P2]."]

    def write(messages):
        calls.append(messages)
        return Written(text=replies[len(calls) - 1], model="fake", input_tokens=10, output_tokens=5)

    search = lambda s, q, c, k, embed_query=None: [_Hit(f"{q}-{i}", f"about {q}") for i in range(k)]  # noqa: E731
    r = write_reading(settings, chart, write=write, search_fn=search, on=date(1900, 1, 1))
    system, user = calls[0][0]["content"], calls[0][1]["content"]
    assert "Jyotishi" in system  # the training prompt
    assert "Lagna (Ascendant): Libra" in user and "[P1]" in user and "never recalculate" in user
    assert "Do not write those sections again" in user and "**Key factors**" in user
    # the repair turn saw the first answer and the flagged sentence
    assert len(calls) == 2 and calls[1][-2]["role"] == "assistant" and "Saturn in the 7th house" in calls[1][-1]["content"]
    assert r.text.startswith("**Chart overview** _(calculated by software)_")
    assert "The model's own overview" not in r.text            # the model's factual sections are dropped
    assert r.text.endswith("**Classical view**\nLibra Lagna [1]. Saturn in the 2nd house [2].")
    assert [p.n for p in r.cited] == [1, 2] and r.input_tokens == 20
    assert any("corrected 1 flagged sentence" in n for n in r.notices)
    assert not any("Saturn in the 7th house" in n for n in r.notices)


@needs_ephemeris
def test_the_repair_is_kept_only_when_it_has_fewer_problems(settings):
    chart = natal.compute_chart(parse_birth("1869-10-02", "07:12", "+04:38", 21.6417, 69.6293))
    calls = []

    def write(messages):
        calls.append(messages)
        return Written(text="Saturn in the 7th house." if len(calls) == 1 else "Saturn in the 9th house. Mars in Leo.",
                       model="fake")

    settings.writer.remove_flagged = False
    r = write_reading(settings, chart, write=write, search_fn=lambda *a, **k: [], on=date(1900, 1, 1))
    assert len(calls) == 2 and r.text.endswith("Saturn in the 7th house.")
    settings.writer.remove_flagged = True  # the wrong sentence is then deleted, with a notice
    calls.clear()
    r = write_reading(settings, chart, write=write, search_fn=lambda *a, **k: [], on=date(1900, 1, 1))
    assert "Saturn in the 7th house." not in r.text and any("were removed" in n for n in r.notices)
    settings.writer.repair = False
    calls.clear()
    write_reading(settings, chart, write=write, search_fn=lambda *a, **k: [], on=date(1900, 1, 1))
    assert len(calls) == 1


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


@needs_ephemeris
def test_birth_details_in_the_chat_get_a_chart_reading(settings):
    from mmrag.astro.chat import BirthDetails, chart_reply
    from mmrag.astro.reading import Reading

    full = BirthDetails(name="Kripa", date="1981-04-10", time="12:30", place="Vaishali, Bihar", latitude=25.68,
                        longitude=85.21, timezone="Asia/Kolkata", question="future")
    seen = {}

    def reading_fn(s, chart, question=None, writer="astro"):
        seen.update(chart=chart, question=question, writer=writer)
        return Reading(text="A reading [1].", model="fake", passages=[], cited=[], notices=["a note"], chart=chart)

    msg = "My name is kripa, my date of birth is 10-04-1981, born in vaishali at 12:30 PM. My future?"
    answer = chart_reply(settings, msg, extract=lambda s, m: full, reading_fn=reading_fn)
    text = "\n".join(b["content"]["markdown"] for b in answer["blocks"])
    assert "born 10 April 1981 at 12:30 in Vaishali, Bihar" in text and "| Lagna | Cancer |" in text
    assert "A reading [1]." in text and answer["notices"] == ["a note"]
    assert seen["question"] == "future" and seen["writer"] == "openai"  # no WRITER_BASE_URL in this config

    no_time = full.model_copy(update={"time": None})
    answer = chart_reply(settings, msg, extract=lambda s, m: no_time, reading_fn=reading_fn)
    assert "I also need the time of birth" in answer["blocks"][0]["content"]["markdown"]

    assert chart_reply(settings, "What does Saturn in the 7th house give?", extract=None) is None
    no_date = BirthDetails()
    assert chart_reply(settings, "what is my lagna", extract=lambda s, m: no_date) is None


@needs_ephemeris
def test_coming_periods_name_the_next_mahadasha():
    """Review 2026-10-13: the writer called the Mercury mahadasha from 2027 'the Mercury sub-period'."""
    from datetime import date

    from mmrag.astro.compose import coming_periods, factual_sections
    from mmrag.astro.natal import parse_birth

    chart = natal.compute_chart(parse_birth("1981-04-10", "12:30", "Asia/Kolkata", 25.68, 85.22))
    text = coming_periods(chart, date(2026, 10, 13))
    assert "in its Jupiter sub-period (antardasha) until March 2027" in text
    assert "The Jupiter sub-period is the last one of the Saturn mahadasha." in text
    assert "Next mahadasha: Mercury, March 2027 to March 2044 (17 years)" in text
    assert "Mercury rules the 3rd and 12th and sits in Pisces in the 9th house" in text
    assert "debilitated, with the debility cancelled" in text and "After that: Ketu mahadasha" in text
    early = coming_periods(chart, date(2015, 1, 1))  # mid-mahadasha: the sub-periods still to come are listed
    assert "in its Ketu sub-period" in early and "Sub-periods still to come in the Saturn mahadasha: Venus" in early
    shown = factual_sections(chart, date(2026, 10, 13))
    assert "**Coming periods**" in shown
    assert "- Note: Kemadruma is present but cancelled" in shown and "Strength: Kemadruma" not in shown


@needs_ephemeris
def test_checks_catch_lords_yoga_owners_pronoun_dignities_and_misnamed_periods():
    """The wrong sentences from two v4 answers for this chart (2026-10-13), and right ones that must pass."""
    from mmrag.astro.natal import parse_birth
    from mmrag.astro.reading import check_analysis, check_positions

    chart = natal.compute_chart(parse_birth("1981-04-10", "12:30", "Asia/Kolkata", 25.68, 85.22))

    def flags(t):
        return check_positions(t, chart) + check_analysis(t, chart)

    assert "lord of the 3rd; it rules the 7th and 8th" in " ".join(flags(
        "Saturn is the 3rd lord joined with Jupiter, the 9th lord."))
    assert "credits Venus with Gajakesari yoga" in " ".join(flags(
        "For your chart this fits Venus well: it is exalted there (Vargottama) and forms Gajakesari with Jupiter."))
    assert "calls Mercury in its own sign; in the chart it is debilitated" in " ".join(flags(
        "For a Cancer Lagna, Mercury rules the 3rd and 12th houses, so it sits in its own sign there."))
    assert "calls the Mercury mahadasha (March 2027 to March 2044) a sub-period" in " ".join(flags(
        "The Mercury antardasha follows from March 2027 until the end of its own period in 2044."))
    from mmrag.astro.reading import remove_flagged_sentences

    two = "The Sun is the natural karaka for authority. It sits in its own sign. Venus is exalted in Pisces."
    found = " ".join(flags(two))  # the claim is in the sentence after the name
    assert "calls Sun" in found and "own sign" in found
    kept, removed = remove_flagged_sentences(two, chart)
    assert kept == "The Sun is the natural karaka for authority. Venus is exalted in Pisces." and len(removed) == 1
    for ok in ("Saturn is the 7th and 8th lord, placed in the 3rd.", "Saturn is lord of the 7th and 8th.",
               "Jupiter in the 4th from the Moon forms Gajakesari.", "Venus is exalted in Pisces, and it is Vargottama.",
               "Mercury is debilitated, but it is exalted in the Navamsa.",
               "The Mercury mahadasha begins in March 2027 and runs until 2044.",
               "The Rahu sub-period (2021 to 2024) brought changes.", "Sun and Mercury form Budha-Aditya yoga."):
        assert flags(ok) == [], ok


def test_personal_follow_up_questions_are_recognised():
    from mmrag.astro.reading import asks_personal_followup

    for q in ("In which field i shouls make career", "Will I get married soon?", "what about marriage?",
              "and my health?", "Which gemstone should I wear?", "what is my lagna", "When will my job change?"):
        assert asks_personal_followup(q), q
    for q in ("What does Saturn in the 7th house give?", "Which charts in the books have Mars in Leo?",
              "Explain Gajakesari yoga", "thanks", "I see"):
        assert not asks_personal_followup(q), q


@needs_ephemeris
def test_follow_ups_use_the_chart_read_earlier_in_the_conversation(settings):
    from mmrag.astro.chat import BirthDetails, chart_reply, followup_reply
    from mmrag.astro.reading import Reading

    full = BirthDetails(name="Kripa", date="1981-04-10", time="12:30", place="Vaishali, Bihar", latitude=25.68,
                        longitude=85.21, timezone="Asia/Kolkata", question="future")
    seen = []

    def reading_fn(s, chart, question=None, writer="astro"):
        seen.append(question)
        return Reading(text="**Chart overview** _(calculated by software)_\nSame as before.\n\nAn answer.",
                       body="An answer.", model="fake", passages=[], cited=[], notices=[], chart=chart)

    kept = []
    msg = "My date of birth is 10-04-1981, born in vaishali at 12:30 PM. My future?"
    chart_reply(settings, msg, extract=lambda s, m: full, reading_fn=reading_fn, on_details=kept.append)
    assert kept == [full]  # the chat keeps these for the conversation

    no_time = full.model_copy(update={"time": None})  # no chart computed: nothing kept
    chart_reply(settings, msg, extract=lambda s, m: no_time, reading_fn=reading_fn, on_details=kept.append)
    assert kept == [full]

    answer = followup_reply(settings, "In which field should I make my career?", kept[0], reading_fn=reading_fn)
    text = "\n".join(b["content"]["markdown"] for b in answer["blocks"])
    assert seen[-1] == "In which field should I make my career?"
    assert "chart read earlier in this conversation" in text and "10 April 1981 at 12:30" in text
    assert "An answer." in text and "| Lagna |" not in text  # the chart table is not repeated
    assert "Chart overview" not in text and "Same as before." not in text  # nor the code-written sections

    first = chart_reply(settings, msg, extract=lambda s, m: full, reading_fn=reading_fn)  # the first reading shows them
    assert "Chart overview" in "\n".join(b["content"]["markdown"] for b in first["blocks"])
