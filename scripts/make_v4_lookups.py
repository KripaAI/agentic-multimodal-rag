"""Training examples for astro-v4: questions answered by looking up the chart facts the app gives the writer.

The writer (astro-v2-8b) was trained on the jyotish-lora chart text, but this app sends a different facts block
(mmrag.astro.natal.facts + mmrag.astro.analysis.analysis_facts). Given correct facts it still mis-copied them
("Ketu in the 2nd", "the 3rd from Pisces is Cancer", sub-periods called mahadashas). Each example here is a random
birth chart in exactly the app's prompt format, one question, and an answer composed by code from the computed
chart, so every answer is correct by construction. No language model writes any of it.

    python scripts/make_v4_lookups.py --n 650 --out C:/Users/krish/Projects/jyotish-lora/data/sft_v4/lookups.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from datetime import date
from pathlib import Path

from mmrag.astro.analysis import (KENDRA, analysis_facts, aspected_by, house_from, navamsa, navamsa_dignity,
                                  neecha_bhanga, sign_at, yogas)
from mmrag.astro.natal import BODIES, PLANETS, SIGN_LORD, antardashas, compute_chart, ordinal, parse_birth
from mmrag.astro.reading import reading_messages
from mmrag.config import PROJECT_ROOT, get_settings

ON = date(2026, 10, 12)
CITIES = [("New Delhi", 28.61, 77.21), ("Mumbai", 19.08, 72.88), ("Kolkata", 22.57, 88.36), ("Chennai", 13.08, 80.27),
          ("Bengaluru", 12.97, 77.59), ("Hyderabad", 17.39, 78.49), ("Patna", 25.59, 85.14), ("Jaipur", 26.91, 75.79),
          ("Lucknow", 26.85, 80.95), ("Ahmedabad", 23.02, 72.57), ("Pune", 18.52, 73.86), ("Varanasi", 25.32, 82.97),
          ("Bhopal", 23.26, 77.41), ("Guwahati", 26.14, 91.74), ("Thiruvananthapuram", 8.52, 76.94), ("Vaishali", 25.68, 85.20)]


def join(xs: list[str]) -> str:
    return xs[0] if len(xs) == 1 else ", ".join(xs[:-1]) + " and " + xs[-1] if xs else "no planet"


def questions(chart, rng: random.Random) -> list[tuple[str, str, str]]:
    """(kind, question, answer) for one chart."""
    p, lagna, moon = chart.placements, chart.lagna.sign, chart.placements["Moon"]
    out = []
    b = rng.choice(BODIES)
    n = house_from(moon.sign, p[b].sign)
    out.append(("from_moon", f"In which house from the Moon is {b}?",
                f"{b} is in {p[b].sign}. Counting from the Moon's sign, {moon.sign}, as the 1st, {p[b].sign} is the "
                f"{ordinal(n)}. So {b} is in the {ordinal(n)} from the Moon."))
    k = rng.randint(2, 12)
    s = sign_at(moon.sign, k)
    occ = [x for x in BODIES if p[x].sign == s]
    out.append(("moon_house_sign", f"Which sign is the {ordinal(k)} house from the Moon, and what is in it?",
                f"The Moon is in {moon.sign}, so the {ordinal(k)} from the Moon is {s}. It holds {join(occ)}."))
    h = rng.randint(1, 12)
    lord = SIGN_LORD[sign_at(lagna, h)]
    out.append(("house_lord", f"Who rules the {ordinal(h)} house, and where is that lord placed?",
                f"The {ordinal(h)} house is {sign_at(lagna, h)}, ruled by {lord}. {lord} sits in {p[lord].sign}, the "
                f"{ordinal(p[lord].house)} house."))
    pl = rng.choice(PLANETS)
    d = p[pl].dignity
    out.append(("dignity", f"Is {pl} exalted, debilitated or in its own sign?",
                f"{pl} is in {p[pl].sign}" + (f", where it is {d}." if d else f", which is ruled by {SIGN_LORD[p[pl].sign]}: "
                f"{pl} is neither exalted, debilitated nor in its own sign there.")))
    pl = rng.choice(PLANETS)
    d9 = navamsa(p[pl].longitude)
    dig9 = navamsa_dignity(pl, d9)
    out.append(("navamsa", f"What is {pl}'s Navamsa sign, and how is it placed there?",
                f"{pl}'s Navamsa is {d9}" + (f", where it is {dig9}." if dig9 else ", where it is neither exalted, debilitated nor "
                f"in its own sign.") + (" It is Vargottama: the same sign in the birth chart and the Navamsa." if d9 == p[pl].sign else "")))
    h = rng.randint(1, 12)
    by = aspected_by(chart, h)
    out.append(("aspects", f"Which planets aspect the {ordinal(h)} house?",
                (f"The {ordinal(h)} house receives the aspects of {join(by)}." if by else
                 f"No planet aspects the {ordinal(h)} house (Rahu and Ketu are not counted).")))
    h = rng.randint(1, 12)
    occ = [x for x in BODIES if p[x].house == h]
    out.append(("occupants", f"Which planets are in the {ordinal(h)} house?",
                f"The {ordinal(h)} house ({sign_at(lagna, h)}) " + (f"holds {join(occ)}." if occ else "is empty.")))
    md, ad = chart.current(ON)
    if md and ad:
        subs = [a for a in antardashas(md) if a.end > md.start]
        nxt = next((a for a in subs if a.start >= ad.end), None)
        nxt_md = next((x for x in chart.dashas if x.start >= md.end), None)
        out.append(("dasha", "Which dasha period is running now, and what comes next?",
                    f"{md.lord} mahadasha runs until {md.end:%B %Y}. Within it, the {ad.sub} antardasha (sub-period) runs until "
                    f"{ad.end:%B %Y}" + (f"; the next sub-period is {nxt.sub}, from {nxt.start:%B %Y}." if nxt else ".")
                    + (f" The next mahadasha is {nxt_md.lord}, from {nxt_md.start:%B %Y}." if nxt_md else "")))
    nbs = neecha_bhanga(chart)
    if nbs:
        nb = rng.choice(nbs)
        out.append(("neecha_bhanga", f"Is {nb.body}'s debility cancelled?",
                    (f"Yes, by a classical rule: {nb.classical[0]}." if nb.classical else
                     "Not by the classical kendra rules." + (f" Later rules are cited: {'; '.join(nb.later)}." if nb.later else ""))))
    ys = yogas(chart)
    out.append(("yogas", "Which yogas does the software find in this chart?",
                ("The software lists: " + "; ".join(f"{y.name} ({y.condition})" for y in ys) + ". No other yoga should be named.")
                if ys else "The software finds none of the yogas it checks, so no yoga should be named."))
    h = rng.choice([x for x in range(1, 13)])
    kinds = [k for k, hs in (("a kendra", KENDRA), ("a trikona", {1, 5, 9}), ("a dusthana", {6, 8, 12})) if h in hs]
    out.append(("house_kind", f"Is the {ordinal(h)} house a kendra, a trikona or a dusthana?",
                f"The {ordinal(h)} house is " + (join(kinds) + "." if kinds else "none of these: kendras are the 1st, 4th, 7th and "
                "10th, trikonas the 1st, 5th and 9th, dusthanas the 6th, 8th and 12th.")))
    return out


# Facts blocks (first line prefixes) in the order the app writes them. The full block runs ~2,800 tokens, too long
# for Kaggle's T4 at 3,072 with the system prompt (memory test 2026-10-12: 116/150 lookups over). Each example keeps
# the header, the planets, the block(s) its question needs and two other blocks as distractors, in the app's order.
BLOCKS = ["Planets:", "House by house", "Moon sign", "Vimshottari", "Running on", "Sub-periods", "Next mahadasha",
          "Mahadashas", "Age on", "House kinds", "Dignity and Navamsa", "House lords", "Aspects (", "Houses receiving",
          "Houses counted from the Moon", "Sign rulers", "Natural significators", "Counted from each planet",
          "Debility and its cancellation", "Yogas"]
NEEDS = {"from_moon": ["Houses counted from the Moon"], "moon_house_sign": ["Houses counted from the Moon"],
         "house_lord": ["House lords", "House by house"], "dignity": ["Dignity and Navamsa"],
         "navamsa": ["Dignity and Navamsa"], "aspects": ["Aspects (", "Houses receiving"],
         "occupants": ["House by house"], "dasha": ["Running on", "Sub-periods", "Next mahadasha", "Mahadashas"],
         "neecha_bhanga": ["Debility and its cancellation"], "yogas": ["Yogas"], "house_kind": ["House kinds"]}


def split_blocks(facts: str) -> list[tuple[str, list[str]]]:
    """[(block key, lines)]; lines before the first known block go under 'head'."""
    out: list[tuple[str, list[str]]] = [("head", [])]
    for line in facts.splitlines():
        key = next((b for b in BLOCKS if line.startswith(b)), None)
        if key:
            out.append((key, [line]))
        else:
            out[-1][1].append(line)
    return out


def facts_subset(facts: str, kind: str, rng: random.Random) -> str:
    blocks = split_blocks(facts)
    keys = [k for k, _ in blocks]
    want = {"head", "Planets:", *NEEDS[kind]}
    others = [k for k in keys if k not in want and k not in ("Counted from each planet",)]
    want |= set(rng.sample(others, min(2, len(others))))
    return "\n".join("\n".join(lines) for k, lines in blocks if k in want).strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=650)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=21)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    settings = get_settings()
    rows, kinds = [], Counter()
    while len(rows) < a.n:
        born = date(rng.randint(1945, 2023), rng.randint(1, 12), rng.randint(1, 28))
        city, lat, lon = rng.choice(CITIES)
        chart = compute_chart(parse_birth(born.isoformat(), f"{rng.randint(0, 23):02d}:{rng.randint(0, 59):02d}",
                                          "Asia/Kolkata", lat, lon, None, city))
        facts = chart.facts(ON) + "\n\n" + analysis_facts(chart, ON)
        for kind, q, ans in rng.sample(questions(chart, rng), 2):  # two questions per chart keeps charts varied
            sub = facts_subset(facts, kind, rng)
            msgs = reading_messages(settings, sub, [], f"{q} Answer from the chart facts above.")
            msgs[0]["content"] = msgs[0]["content"].replace("\r\n", "\n").strip()
            rows.append({"id": f"lookup-{len(rows):04d}", "task": "lookup_v4", "category": kind,
                         "messages": msgs + [{"role": "assistant", "content": ans}]})
            kinds[kind] += 1
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows[:a.n]) + "\n", encoding="utf-8")
    longest = max(sum(len(m["content"]) for m in r["messages"]) for r in rows)
    print(f"wrote {min(len(rows), a.n)} lookup examples -> {out}; longest {longest} chars (~{longest // 3.5:.0f} tokens)")
    print(dict(kinds))


if __name__ == "__main__":
    main()
