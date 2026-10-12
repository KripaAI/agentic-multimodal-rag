"""What an astrologer reads off a chart before interpreting it, computed by code (Phase 11).

The writer (an 8B model) miscounted houses ("the 5th from Jupiter is Capricorn"), invented aspects
and Vargottama, missed a cancelled debility, and read poverty yogas to the parents of a six-year-old.
So everything that is arithmetic or a lookup is done here and handed to it as facts:

- age and life stage, with what a reading for that stage should and should not cover
- Navamsa (D9) signs and Vargottama
- Parashari aspects (all planets the 7th; Mars also 4th and 8th, Jupiter 5th and 9th, Saturn 3rd and
  10th); Rahu and Ketu are left out, as the schools disagree on their aspects
- house kinds (kendra, trikona, dusthana, upachaya) and the chart counted from the Moon and from
  every planet (the sign a planet occupies is the 1st from it)
- natural friendship of each planet with the lord of the sign it occupies
- neecha bhanga, separating the classical kendra rules (Phaladeepika 7.26) from later conjunction
  and Navamsa rules, so the model cannot overstate it
- yogas with their conditions: Gajakesari, Budha-Aditya, Chandra-Mangala, the five Mahapurusha
  yogas, Kemadruma (and its cancellation), Parashari Raja yoga, Dhana yoga, Vipareeta Raja yogas.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

from mmrag.astro.natal import BODIES, DEBILITATED, EXALTED, PLANETS, SIGN_LORD, NatalChart, ordinal
from mmrag.ingest.charts import SIGNS

KENDRA, TRIKONA, DUSTHANA, UPACHAYA = {1, 4, 7, 10}, {1, 5, 9}, {6, 8, 12}, {3, 6, 10, 11}
SPECIAL_ASPECTS = {"Mars": (4, 8), "Jupiter": (5, 9), "Saturn": (3, 10)}
FRIENDS = {"Sun": {"Moon", "Mars", "Jupiter"}, "Moon": {"Sun", "Mercury"}, "Mars": {"Sun", "Moon", "Jupiter"},
           "Mercury": {"Sun", "Venus"}, "Jupiter": {"Sun", "Moon", "Mars"}, "Venus": {"Mercury", "Saturn"},
           "Saturn": {"Mercury", "Venus"}}
ENEMIES = {"Sun": {"Venus", "Saturn"}, "Moon": set(), "Mars": {"Mercury"}, "Mercury": {"Moon"},
           "Jupiter": {"Mercury", "Venus"}, "Venus": {"Sun", "Moon"}, "Saturn": {"Sun", "Moon", "Mars"}}
# Moolatrikona sign and degree range (BPHS) where it differs from the exaltation sign.
MOOLATRIKONA = {"Sun": ("Leo", 0, 20), "Mars": ("Aries", 0, 12), "Jupiter": ("Sagittarius", 0, 10),
                "Venus": ("Libra", 0, 15), "Saturn": ("Aquarius", 0, 20)}
MAHAPURUSHA = {"Mars": "Ruchaka", "Mercury": "Bhadra", "Jupiter": "Hamsa", "Venus": "Malavya", "Saturn": "Sasa"}
YOGA_NAMES = ["Gajakesari", "Budha-Aditya", "Chandra-Mangala", "Ruchaka", "Bhadra", "Hamsa", "Malavya", "Sasa",
              "Kemadruma", "Raja yoga", "Dhana yoga", "Harsha", "Sarala", "Vimala", "Neecha Bhanga"]


@dataclass
class LifeStage:
    age: int
    stage: str
    focus: str
    avoid: str


def life_stage(birth: date, on: date) -> LifeStage:
    age = on.year - birth.year - ((on.month, on.day) < (birth.month, birth.day))
    if age < 14:
        return LifeStage(age, "child", "health and vitality, temperament, learning and education, creativity, "
                         "and how parents can support the child",
                         "money, wealth or poverty, career, marriage, longevity or death")
    if age < 25:
        return LifeStage(age, "young person", "education, temperament, health, the start of a career, "
                         "and relationships in general terms", "longevity or death")
    if age < 60:
        return LifeStage(age, "adult", "career, finances, relationships and family, health in general terms",
                         "longevity or death")
    return LifeStage(age, "elder", "health in general terms, family, spiritual life and contentment",
                     "longevity or death")


def house_from(sign_from: str, sign_to: str) -> int:
    """Counted inclusively: a sign is the 1st from itself."""
    return (SIGNS.index(sign_to) - SIGNS.index(sign_from)) % 12 + 1


def sign_at(sign_from: str, n: int) -> str:
    return SIGNS[(SIGNS.index(sign_from) + n - 1) % 12]


def navamsa(longitude: float) -> str:
    """The D9 sign: movable signs start their nine parts from themselves, fixed signs from the 9th,
    dual signs from the 5th."""
    s = int(longitude // 30) % 12
    part = int((longitude % 30) // (30 / 9))
    start = s if s % 3 == 0 else (s + 8 if s % 3 == 1 else s + 4)
    return SIGNS[(start + part) % 12]


def navamsa_dignity(body: str, d9: str) -> str | None:
    """Exalted, debilitated or own sign in the Navamsa (planets only)."""
    if body not in EXALTED:
        return None
    return ("exalted" if d9 == EXALTED[body] else "debilitated" if d9 == DEBILITATED[body]
            else "own sign" if SIGN_LORD[d9] == body else None)


def aspects(chart: NatalChart) -> dict[str, list[int]]:
    """Houses each planet aspects (Rahu and Ketu left out)."""
    out = {}
    for b in PLANETS:
        h = chart.placements[b].house
        out[b] = sorted(((h + n - 2) % 12) + 1 for n in (7, *SPECIAL_ASPECTS.get(b, ())))
    return out


def aspected_by(chart: NatalChart, house: int) -> list[str]:
    return [b for b, hs in aspects(chart).items() if house in hs]


def relation_to_sign_lord(body: str, sign: str) -> str | None:
    lord = SIGN_LORD[sign]
    if body not in FRIENDS or lord == body:
        return None
    return "friend's sign" if lord in FRIENDS[body] else "enemy's sign" if lord in ENEMIES[body] else "neutral sign"


def _kendra_from_lagna_or_moon(chart: NatalChart, body: str) -> str | None:
    p, moon = chart.placements[body], chart.placements["Moon"]
    if p.house in KENDRA:
        return f"{body} is in the {ordinal(p.house)} house, a kendra from the Lagna"
    if body != "Moon" and house_from(moon.sign, p.sign) in KENDRA:
        return f"{body} is in a kendra from the Moon (the {ordinal(house_from(moon.sign, p.sign))})"
    return None


@dataclass
class NeechaBhanga:
    body: str
    classical: list[str] = field(default_factory=list)  # Phaladeepika 7.26: kendra rules
    later: list[str] = field(default_factory=list)      # conjunction and Navamsa rules of later texts

    def line(self) -> str:
        sign = DEBILITATED[self.body]
        head = f"{self.body} is debilitated in {sign}."
        cl = ("Classical cancellation (kendra rules, Phaladeepika): " + "; ".join(self.classical) + "."
              if self.classical else "Classical cancellation (kendra rules, Phaladeepika): not met.")
        lt = (" Later rules also cited for cancellation: " + "; ".join(self.later) + "." if self.later else "")
        return f"{head} {cl}{lt}"


def neecha_bhanga(chart: NatalChart) -> list[NeechaBhanga]:
    out = []
    exalted_in = {s: b for b, s in EXALTED.items()}  # sign -> planet exalted there
    for b in PLANETS:
        p = chart.placements[b]
        if p.dignity != "debilitated":
            continue
        nb = NeechaBhanga(b)
        dispositor, exalt_lord, exalted_here = SIGN_LORD[p.sign], SIGN_LORD[EXALTED[b]], exalted_in.get(p.sign)
        for who, role in ((dispositor, f"the lord of {p.sign}"), (exalt_lord, f"the lord of {EXALTED[b]}, "
                          f"{b}'s exaltation sign"), (exalted_here, f"the planet exalted in {p.sign}")):
            if who and who != b:
                where = _kendra_from_lagna_or_moon(chart, who)
                if where:
                    nb.classical.append(f"{who}, {role}: {where}")
        for who, role in ((dispositor, f"the lord of {p.sign}"), (exalted_here, f"the planet exalted in {p.sign}")):
            if who and who != b and chart.placements[who].sign == p.sign:
                nb.later.append(f"joined by {who}, {role}")
        if navamsa(p.longitude) == EXALTED[b]:
            nb.later.append(f"{b} is exalted in the Navamsa ({EXALTED[b]})")
        out.append(nb)
    return out


@dataclass
class Yoga:
    name: str
    condition: str
    present: bool = True
    houses: set[int] = field(default_factory=set)  # where it forms, when it is a conjunction in one house


def yogas(chart: NatalChart) -> list[Yoga]:
    p, lagna, moon = chart.placements, chart.lagna.sign, chart.placements["Moon"]
    lord_of = {h: SIGN_LORD[sign_at(lagna, h)] for h in range(1, 13)}
    out: list[Yoga] = []
    jup_from_moon = house_from(moon.sign, p["Jupiter"].sign)
    if jup_from_moon in KENDRA:
        out.append(Yoga("Gajakesari", f"Jupiter in the {ordinal(jup_from_moon)} from the Moon, a kendra"))
    if p["Sun"].sign == p["Mercury"].sign:
        out.append(Yoga("Budha-Aditya", f"Sun and Mercury together in {p['Sun'].sign}", houses={p["Sun"].house}))
    if p["Moon"].sign == p["Mars"].sign:
        out.append(Yoga("Chandra-Mangala", f"Moon and Mars together in {moon.sign}", houses={moon.house}))
    for b, name in MAHAPURUSHA.items():
        if p[b].dignity in ("exalted", "own sign") and p[b].house in KENDRA:
            out.append(Yoga(name, f"{b} {p[b].dignity} in the {ordinal(p[b].house)} house, a kendra"))
    beside = [b for b in PLANETS if b not in ("Sun", "Moon")
              and house_from(moon.sign, p[b].sign) in (2, 12)]
    if not beside:
        cancel = [b for b in PLANETS if b != "Moon" and (p[b].house in KENDRA
                  or house_from(moon.sign, p[b].sign) in KENDRA)]
        out.append(Yoga("Kemadruma", "no planet (other than the Sun) in the 2nd or 12th from the Moon"
                        + (f"; cancelled by {', '.join(cancel)} in a kendra from the Lagna or the Moon" if cancel
                           else "; not cancelled")))
    kendra_lords, trikona_lords = {lord_of[h] for h in (1, 4, 7, 10)}, {lord_of[h] for h in (1, 5, 9)}
    for b in PLANETS:
        houses = [h for h, l in lord_of.items() if l == b]
        if any(h in (4, 7, 10) for h in houses) and any(h in (5, 9) for h in houses):
            out.append(Yoga("Raja yoga", f"{b} rules both a kendra and a trikona ("
                            f"{' and '.join(ordinal(h) for h in houses)}): a yogakaraka"))
    pairs = set()
    for a in kendra_lords:
        for b in trikona_lords:
            if a != b and p[a].sign == p[b].sign and (b, a) not in pairs:
                pairs.add((a, b))
                ka = [ordinal(h) for h in (1, 4, 7, 10) if lord_of[h] == a]
                tb = [ordinal(h) for h in (1, 5, 9) if lord_of[h] == b]
                out.append(Yoga("Raja yoga", f"{a} (lord of the {' and '.join(ka)}) joins {b} (lord of the "
                                f"{' and '.join(tb)}) in {p[a].sign}, the {ordinal(p[a].house)} house",
                                houses={p[a].house}))
    wealth_lords, luck_lords = {lord_of[2], lord_of[11]}, {lord_of[1], lord_of[5], lord_of[9]}
    seen = set()
    for a in wealth_lords:
        for b in luck_lords:
            if a != b and p[a].sign == p[b].sign and frozenset((a, b)) not in seen:
                seen.add(frozenset((a, b)))
                out.append(Yoga("Dhana yoga", f"{a} ({', '.join(ordinal(h) for h in (2, 11) if lord_of[h] == a)} "
                                f"lord) joins {b} ({', '.join(ordinal(h) for h in (1, 5, 9) if lord_of[h] == b)} "
                                f"lord) in {p[a].sign}, the {ordinal(p[a].house)} house", houses={p[a].house}))
    for h, name in ((6, "Harsha"), (8, "Sarala"), (12, "Vimala")):
        lord = lord_of[h]
        if p[lord].house in DUSTHANA:
            out.append(Yoga(f"{name} (Vipareeta Raja yoga)", f"{lord}, lord of the {ordinal(h)}, is in the "
                            f"{ordinal(p[lord].house)}, a dusthana"))
    for nb in neecha_bhanga(chart):
        if nb.classical:
            out.append(Yoga("Neecha Bhanga Raja yoga", f"{nb.body}'s debility is cancelled by a classical rule"))
    return out


def counted_from(chart: NatalChart, ref_sign: str) -> str:
    groups: dict[int, list[str]] = {}
    for b in BODIES:
        groups.setdefault(house_from(ref_sign, chart.placements[b].sign), []).append(b)
    return "; ".join(f"{', '.join(bs)} in the {ordinal(h)}" for h, bs in sorted(groups.items()))


def analysis_facts(chart: NatalChart, on: date | None = None) -> str:
    """The computed analysis, as plain lines appended to NatalChart.facts()."""
    on = on or date.today()
    st = life_stage(chart.birth.when_local.date(), on)
    p, lagna = chart.placements, chart.lagna.sign
    lines = [f"Age on {on:%Y-%m-%d}: {st.age} ({st.stage}).", "",
             "House kinds: kendras 1, 4, 7, 10; trikonas 1, 5, 9; dusthanas 6, 8, 12; upachayas 3, 6, 10, 11.",
             "Dignity and Navamsa:"]
    for b in BODIES:
        pl = p[b]
        d9 = navamsa(pl.longitude)
        d9_dig = navamsa_dignity(b, d9)
        parts = [f"Navamsa {d9}" + (" (Vargottama)" if d9 == pl.sign else "")
                 + (f" ({d9_dig} in the Navamsa)" if d9_dig else "")]
        if b in MOOLATRIKONA and pl.sign == MOOLATRIKONA[b][0] and MOOLATRIKONA[b][1] <= pl.degree < MOOLATRIKONA[b][2]:
            parts.append("in its moolatrikona range")
        if not pl.dignity and (rel := relation_to_sign_lord(b, pl.sign)):
            parts.append(f"in a {rel} ({SIGN_LORD[pl.sign]})")
        elif pl.dignity:
            parts.append(pl.dignity)
        lines.append(f"- {b}: {', '.join(parts)}")
    d9_lagna = navamsa(chart.lagna.longitude)
    lines.append(f"- Lagna: Navamsa {d9_lagna}" + (" (Vargottama)" if d9_lagna == lagna else ""))
    lines.append("House lords and where they sit: " + "; ".join(
        f"{ordinal(h)} lord {SIGN_LORD[sign_at(lagna, h)]} in the {ordinal(p[SIGN_LORD[sign_at(lagna, h)]].house)}"
        for h in range(1, 13)) + ".")
    lines.append("Aspects (Parashari; Rahu and Ketu not counted):")
    asp = aspects(chart)
    for b in PLANETS:
        lines.append(f"- {b} (in the {ordinal(p[b].house)}) aspects the "
                     f"{', '.join(ordinal(h) for h in asp[b])} house" + ("s" if len(asp[b]) > 1 else ""))
    received = [f"{ordinal(h)}: {', '.join(aspected_by(chart, h))}" for h in range(1, 13) if aspected_by(chart, h)]
    lines.append("Houses receiving aspects: " + "; ".join(received) + ".")
    moon_sign = p["Moon"].sign
    lines.append("Houses counted from the Moon (Chandra Lagna), with their signs and occupants: " + "; ".join(
        f"{ordinal(h)} {sign_at(moon_sign, h)}"
        + (f" ({', '.join(b for b in BODIES if p[b].sign == sign_at(moon_sign, h))})"
           if any(p[b].sign == sign_at(moon_sign, h) for b in BODIES) else "")
        for h in range(1, 13)) + ".")
    lines.append("Sign rulers (fixed): " + "; ".join(f"{s} - {SIGN_LORD[s]}" for s in SIGNS)
                 + ". A planet's own signs are only the signs it rules.")
    lines.append("Natural significators (karakas): Sun - father, soul, authority; Moon - mother, mind; Mars - siblings, "
                 "courage, property; Mercury - intellect, speech, learning; Jupiter - children, wisdom, teachers, the "
                 "husband; Venus - the wife, marriage, arts, comfort; Saturn - longevity, discipline, service.")
    lines.append("Counted from each planet (the sign a planet occupies is the 1st from it):")
    for b in PLANETS:
        lines.append(f"- from {b} ({p[b].sign}): {counted_from(chart, p[b].sign)}")
    nbs = neecha_bhanga(chart)
    if nbs:
        lines.append("Debility and its cancellation:")
        lines += [f"- {nb.line()}" for nb in nbs]
    ys = yogas(chart)
    lines.append("Yogas found by the software (name no other yoga):"
                 if ys else "Yogas: none of the yogas the software checks are present (name no yoga).")
    lines += [f"- {y.name}: {y.condition}" for y in ys]
    return "\n".join(lines)
