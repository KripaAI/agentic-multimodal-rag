"""Phase 11: the astrologer's analysis computed by code, and the checks that hold the writer to it."""

from __future__ import annotations

from datetime import date

import pytest

from mmrag.astro import analysis, natal
from mmrag.astro.natal import parse_birth
from mmrag.astro.reading import check_analysis, check_positions, is_example_horoscope, reading_rules

needs_ephemeris = pytest.mark.skipif(not (natal._ephemeris_dir() / "de440s.bsp").is_file(),
                                     reason="JPL ephemeris not downloaded")


@pytest.fixture(scope="module")
def child():
    """Dikshant: 20 April 2020, 16:46, New Delhi. Virgo Lagna; Mars, Jupiter, Saturn in Capricorn."""
    return natal.compute_chart(parse_birth("2020-04-20", "16:46", "Asia/Kolkata", 28.61, 77.21))


ON = date(2026, 10, 12)


def test_life_stages():
    assert analysis.life_stage(date(2020, 4, 20), ON).age == 6
    assert analysis.life_stage(date(2020, 4, 20), ON).stage == "child"
    assert analysis.life_stage(date(2020, 10, 13), ON).age == 5  # birthday not yet reached
    assert analysis.life_stage(date(1990, 1, 1), ON).stage == "adult"
    assert "marriage" in analysis.life_stage(date(2020, 4, 20), ON).avoid


def test_navamsa_follows_the_movable_fixed_dual_rule():
    assert analysis.navamsa(0.0) == "Aries"            # Aries (movable) starts from itself
    assert analysis.navamsa(30.0) == "Capricorn"       # Taurus (fixed) starts from its 9th
    assert analysis.navamsa(60.0) == "Libra"           # Gemini (dual) starts from its 5th
    assert analysis.navamsa(29.99) == "Sagittarius"    # last part of Aries
    assert analysis.navamsa(270 + 2.2) == "Capricorn"  # Jupiter at Capricorn 2°12': Vargottama


def test_counting_from_a_sign_is_inclusive():
    assert analysis.house_from("Capricorn", "Capricorn") == 1
    assert analysis.house_from("Capricorn", "Taurus") == 5
    assert analysis.sign_at("Capricorn", 5) == "Taurus"


@needs_ephemeris
def test_aspects(child):
    asp = analysis.aspects(child)
    assert asp["Saturn"] == [2, 7, 11] and asp["Mars"] == [8, 11, 12] and asp["Jupiter"] == [1, 9, 11]
    assert set(analysis.aspected_by(child, 1)) == {"Moon", "Mercury", "Jupiter"}


@needs_ephemeris
def test_neecha_bhanga_separates_classical_and_later_rules(child):
    nbs = {nb.body: nb for nb in analysis.neecha_bhanga(child)}
    jup, merc = nbs["Jupiter"], nbs["Mercury"]
    assert any("Moon, the lord of Cancer" in r for r in jup.classical)       # exaltation-sign lord in a kendra
    assert any("Saturn" in r for r in jup.later) and any("Mars" in r for r in jup.later)
    assert merc.classical == []


@needs_ephemeris
def test_yogas_of_the_child_chart(child):
    names = [y.name for y in analysis.yogas(child)]
    assert "Gajakesari" not in names and "Budha-Aditya" not in names
    assert "Raja yoga" in names and "Dhana yoga" in names and "Neecha Bhanga Raja yoga" in names
    assert any(n.startswith("Vimala") for n in names)
    kem = next(y for y in analysis.yogas(child) if y.name == "Kemadruma")
    assert "cancelled by Mercury" in kem.condition


@needs_ephemeris
def test_gandhi_has_gajakesari_and_malavya():
    g = natal.compute_chart(parse_birth("1869-10-02", "07:12", "+04:38", 21.6417, 69.6293))
    names = [y.name for y in analysis.yogas(g)]
    assert "Gajakesari" in names and "Malavya" in names and "Sasa" not in names


@needs_ephemeris
def test_facts_carry_the_analysis(child):
    facts = analysis.analysis_facts(child, ON)
    assert "Age on 2026-10-12: 6 (child)." in facts
    assert "- Jupiter: Navamsa Capricorn (Vargottama) (debilitated in the Navamsa), debilitated" in facts
    assert "from Jupiter (Capricorn): Mars, Jupiter, Saturn in the 1st" in facts and "Venus in the 5th" in facts
    assert "Classical cancellation (kendra rules, Phaladeepika): not met." in facts  # Mercury


@needs_ephemeris
def test_the_checks_catch_the_errors_of_the_reading_for_a_child(child):
    """Sentences from the reading of 2026-10-12 for the six-year-old, and true statements."""
    stage = analysis.life_stage(child.birth.when_local.date(), ON)
    wrong = ("Here Jupiter is debilitated in the 5th, so the 5th from Jupiter is Capricorn, which holds Saturn. "
             "Penury follows. Saturn aspects the 4th house. Mars aspects the Moon. Mercury is Vargottama. "
             "This forms a Gajakesari yoga. The Sun is in his own sign.")
    notes = " ".join(check_analysis(wrong, child, stage) + check_positions(wrong, child))
    for expected in ("5th from Jupiter is Capricorn; it is Taurus", "Saturn aspects the 4th house",
                     "Mars aspects Moon", "Mercury Vargottama", "Gajakesari yoga", "penury",
                     "calls Sun own sign"):
        assert expected in notes, expected
    right = ("Venus is in the 5th from Jupiter. Rahu is in the 6th from Jupiter. Saturn aspects the 7th house. "
             "The Lagna is aspected by Jupiter. Jupiter is Vargottama. There is no Malavya yoga here. "
             "A Raja yoga is formed. Jupiter's debility is cancelled (Neecha Bhanga Raja yoga).")
    assert check_analysis(right, child, stage) == []


def test_example_horoscopes_are_recognised():
    assert is_example_horoscope("In Chart No. 97 Leo is aspected by Mars.")
    assert is_example_horoscope("The native's father died in April 1979 in Rahu Bhukti of Mercury Dasa.")
    assert not is_example_horoscope("Saturn in the 10th house gives fame and administrative positions.")


def test_rules_for_a_child_reading():
    stage = analysis.life_stage(date(2020, 4, 20), ON)
    rules = reading_rules(stage, with_passages=True)
    assert "6 years old (child)" in rules and "Do not discuss money, wealth or poverty, career, marriage" in rules
    assert "never count houses yourself" in rules and "[P3]" in rules
    assert "[P3]" not in reading_rules(stage, with_passages=False)


@needs_ephemeris
def test_a_cancellation_moved_to_the_wrong_planet_is_flagged(child):
    """The second reading for the child (2026-10-12) gave Jupiter's cancelled debility to Mercury."""
    stage = analysis.life_stage(child.birth.when_local.date(), ON)
    notes = check_analysis("Mercury is debilitated there, but its debility is cancelled by the classical rule. "
                           "Jupiter's debility is cancelled. Venus's debility is relieved.", child, stage)
    assert len(notes) == 2
    assert "Mercury's debility is cancelled; no rule cancels it" in notes[0]
    assert "Venus is not debilitated" in notes[1]
    rules = reading_rules(stage, True, child)
    assert "cancelled by a classical rule: Jupiter" in rules and "NOT cancelled: Mercury" in rules


@needs_ephemeris
def test_wrong_aspect_receipts_are_flagged(child):
    stage = analysis.life_stage(child.birth.when_local.date(), ON)
    notes = check_analysis("The Moon, the karaka of the mother, is in the 7th, a kendra, and receives aspects from "
                           "Jupiter. The 7th house receives the aspect of Saturn.", child, stage)
    assert len(notes) == 1 and "Moon receives the aspect of Jupiter; it does not" in notes[0]


def test_named_books_are_linked_to_the_passage_that_quotes_them():
    from mmrag.astro.reading import Passage, resolve_citations

    ps = [Passage(1, "c1", "bhrigu-sutras.pdf", [72], "Other Views Phaldeepika-The native will be long-lived. "
                  "Saravali-The native will be wealthy", "q"),
          Passage(2, "c2", "raman-how-to-judge-horoscope-2.pdf", [10], "text", "q")]
    out, cited, _ = resolve_citations("Phaldeepika describes a flatterer. Saravali describes conveyances. "
                                      "Raman notes this [P2]. Jataka Parijata agrees.", ps)
    assert out == ("Phaldeepika [1] describes a flatterer. Saravali [1] describes conveyances. Raman notes this [2]. "
                   "Jataka Parijata agrees.")
    assert [p.n for p in cited] == [2, 1]


@needs_ephemeris
def test_house_kinds_after_a_comma_and_wrong_conjunctions_are_flagged():
    """From the adult reading of 2026-10-12 (10 April 1981, 12:30, Vaishali)."""
    from datetime import date as d

    adult = natal.compute_chart(parse_birth("1981-04-10", "12:30", "Asia/Kolkata", 25.68, 85.20))
    stage = analysis.life_stage(d(1981, 4, 10), ON)
    text = ("The 9th lord Jupiter is in the 3rd, a trikona, and is retrograde. The Moon is joined by a benefic "
            "(Rahu). Venus is joined by Mars and Mercury. Jupiter is conjoined with Saturn.")
    notes = " ".join(check_positions(text, adult) + check_analysis(text, adult, stage))
    assert "3rd house a trikona" in notes and "Moon is joined by Rahu" in notes
    assert "Venus is joined by Mars" not in notes and "Jupiter is joined by Saturn" not in notes


@needs_ephemeris
def test_the_third_child_reading_errors_are_all_caught(child):
    """Every error in the reading of 2026-10-12 13:17 for the child, and true statements left alone."""
    stage = analysis.life_stage(child.birth.when_local.date(), ON)
    wrong = ("The Moon is in Pisces, its own sign, and the 11th lord Moon joins the 1st lord Mercury in the 11th. "
             "Mars is exalted in the 5th and also exalted in Navamsa Cancer. The 10th lord is in the 5th. "
             "It describes Virgo as a feminine sign, ruled by Mars. Mars is the significator of the mother, and the "
             "5th is the house of the mother. The Dhana yoga in the 11th suggests aptitude.")
    notes = " ".join(check_analysis(wrong, child, stage))
    for expected in ("Pisces Moon's own sign", "Mars exalted in the Navamsa", "10th lord is in the 5th",
                     "places Moon in the 11th", "Virgo is ruled by Mars", "Mars the significator of the mother",
                     "5th the house of mother", "Dhana yoga in the 11th"):
        assert expected in notes, expected
    right = ("Venus is in Taurus, its own sign. Mars is debilitated in the Navamsa. The 10th lord is in the 7th. "
             "Pisces is ruled by Jupiter. The Moon is the significator of the mother, and the 4th is the house of "
             "the mother. Mars is the significator of courage. The Dhana yoga in the 7th.")
    assert check_analysis(right, child, stage) == []


@needs_ephemeris
def test_dasha_facts_give_the_balance_at_birth_and_the_sub_periods(child):
    facts = child.facts(ON)
    assert "balance AT BIRTH of 12.40 years (it ends on 2032-09-11)" in facts
    assert "5.9 years left" in facts and "Next mahadasha: Mercury, 2032-09-11" in facts
    assert "Mars 2026-03-15 to 2027-04-24 (running); Rahu 2027-04-24 to 2030-02-28" in facts
    a = analysis.analysis_facts(child, ON)
    assert "- Mars: Navamsa Cancer (debilitated in the Navamsa), exalted" in a
    assert "10th lord Mercury in the 7th" in a and "in Pisces, the 7th house" in a


@needs_ephemeris
def test_the_fourth_child_reading(child):
    """From the reading of 2026-10-12 after the textbook checks: one false alarm, two misses."""
    stage = analysis.life_stage(child.birth.when_local.date(), ON)
    text = ("Saturn is in its own sign Capricorn, Jupiter is debilitated but its debility is cancelled, and Mars is "
            "exalted. Mercury is debilitated, but the Moon, the 2nd lord, is in the 7th with Mercury. The 7th house "
            "is also aspected by the Moon. Venus, the 9th lord, is in its own sign.")
    notes = " ".join(check_analysis(text, child, stage))
    assert "Saturn's debility" not in notes and "Jupiter's debility" not in notes
    assert "calls Moon the 2nd lord" in notes and "7th house is aspected by Moon" in notes
    assert "Venus the 9th lord" not in notes


@needs_ephemeris
def test_no_false_alarms_on_lists_of_planets(child):
    """From the first Level 1 reading (2026-10-12): two false alarms that would trigger needless repairs."""
    stage = analysis.life_stage(child.birth.when_local.date(), ON)
    text = ("It holds three planets: exalted Mars, Vargottama Jupiter and Saturn in its own sign. "
            "Mercury and Saturn in its own sign.")
    assert check_positions(text, child) == [] and check_analysis(text, child, stage) == []
    wrong = "Mercury is in its own sign. Mars is Vargottama."
    notes = " ".join(check_positions(wrong, child) + check_analysis(wrong, child, stage))
    assert "Mercury own sign" in notes and "Mars Vargottama" in notes
