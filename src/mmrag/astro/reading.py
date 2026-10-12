"""Birth-chart readings and plain questions written by the fine-tuned model (Phase 11).

    birth data -> chart computed by code (natal.py) -> library passages for its key factors
               -> the writer (astro-v2-8b on Modal, or the agent's OpenAI model to compare)
               -> checks in code: citations point at given passages; positions match the chart

The writer gets its training system prompt, the chart facts, and numbered passages it cites as
[P3]. It is never asked to calculate anything. The checks never rewrite the text: they drop
citations to passages that were not given and report every problem as a notice.
"""

from __future__ import annotations

import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import date
from typing import Callable, Literal

from mmrag.astro.natal import BODIES, NatalChart, ordinal
from mmrag.config import Settings
from mmrag.ingest.charts import SIGNS

WriterName = Literal["astro", "openai"]


class WriterNotConfigured(RuntimeError):
    pass


@dataclass
class Passage:
    n: int                    # cited as [P<n>]
    chunk_id: str
    source_file: str
    pages: list[int]
    text: str
    query: str                # the chart factor it was found for


@dataclass
class Written:
    text: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class Reading:
    text: str                                   # markdown; [P3] markers replaced by [3]
    model: str
    passages: list[Passage]                     # given to the writer
    cited: list[Passage]                        # cited in the text, in first-cited order
    notices: list[str] = field(default_factory=list)
    chart: NatalChart | None = None
    facts: str = ""
    seconds: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0


_DATE = re.compile(r"\b(?:\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}|\d{4}-\d{1,2}-\d{1,2})\b")
_PERSONAL = re.compile(r"\b(?:date of birth|d\.?o\.?b|born on|i was born|my birth|my (?:future|chart|kundli|horoscope|"
                       r"rashi|lagna|dasha)|birth (?:time|place|details))\b", re.I)


def asks_about_own_chart(question: str) -> bool:
    """A chat question about the asker's own birth chart ('my date of birth is 04-01-1983 ... my future'):
    the document chat cannot compute charts, so the UI points to the birth-chart reading page."""
    return bool(_PERSONAL.search(question)) or (bool(_DATE.search(question)) and
                                                bool(re.search(r"\b(?:my|me|i)\b", question, re.I)))


# ---------------------------------------------------------------- writers

def _openai_like_call(client, model: str, messages: list[dict], max_tokens: int, temperature: float,
                      reasoning: bool) -> Written:
    if reasoning:  # gpt-5.x: no temperature; a little reasoning keeps the comparison fair on speed
        r = client.chat.completions.create(model=model, messages=messages, max_completion_tokens=max_tokens * 3,
                                           reasoning_effort="low")
    else:
        r = client.chat.completions.create(model=model, messages=messages, max_tokens=max_tokens,
                                           temperature=temperature)
    text = (r.choices[0].message.content or "").strip()
    text = re.sub(r"^<think>.*?</think>\s*", "", text, flags=re.S)  # in case the server leaves thinking on
    usage = getattr(r, "usage", None)
    return Written(text=text, model=model, input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
                   output_tokens=getattr(usage, "completion_tokens", 0) or 0)


def get_writer(settings: Settings, which: WriterName = "astro") -> Callable[[list[dict]], Written]:
    """A function messages -> Written. `astro`: the fine-tuned model at WRITER_BASE_URL; `openai`:
    the agent's model (agent.model), for side-by-side comparisons."""
    from openai import OpenAI

    w = settings.writer
    if which == "astro":
        if not settings.secrets.writer_base_url or settings.secrets.writer_api_key is None:
            raise WriterNotConfigured("WRITER_BASE_URL and WRITER_API_KEY are not set in .env (see "
                                      "deploy/modal/astro_llm.py for the server)")
        client = OpenAI(base_url=settings.secrets.writer_base_url.rstrip("/"),
                        api_key=settings.secrets.writer_api_key.get_secret_value(), timeout=w.timeout_s,
                        max_retries=1)
        return lambda messages: _openai_like_call(client, w.model, messages, w.max_tokens, w.temperature, False)
    from mmrag.llm import REASONING_PREFIXES, get_client

    model = settings.agent.model
    client = get_client(settings)
    return lambda messages: _openai_like_call(client, model, messages, w.max_tokens, w.temperature,
                                              model.startswith(REASONING_PREFIXES))


def system_prompt(settings: Settings) -> str:
    return settings.resolve(settings.writer.system_prompt_file).read_text(encoding="utf-8").strip()


# ---------------------------------------------------------------- passages

def chart_queries(chart: NatalChart, on: date | None = None) -> list[str]:
    """What to look up in the books for this chart, most important first. The Lagna query names its
    characteristics: "Virgo ascendant native" alone matched an example horoscope in a chapter on
    poverty yogas."""
    from mmrag.astro.analysis import life_stage, yogas

    on = on or date.today()
    p, lagna = chart.placements, chart.lagna
    moon = p["Moon"]
    md, ad = chart.current(on)
    queries = [f"{lagna.sign} Lagna: appearance, temperament and character of persons born with {lagna.sign} rising",
               f"Moon in {moon.sign} in the {ordinal(moon.house)} house",
               f"born in {moon.nakshatra} nakshatra: character and temperament"]
    if life_stage(chart.birth.when_local.date(), on).stage == "child":
        queries.append("education, intelligence and learning: the 5th house and Mercury")
    queries += [f"{y.name.split(' (')[0]} yoga: definition and results" for y in yogas(chart)
                if y.name != "Kemadruma"][:3]
    if md:
        queries.append(f"results of {md.lord} mahadasha" + (f" and {ad.sub} antardasha" if ad else ""))
    for body in BODIES:
        if body != "Moon":
            queries.append(f"{body} in the {ordinal(p[body].house)} house")
    return queries


_EXAMPLE_CHART = re.compile(r"\b(?:chart|horoscope)\s*(?:no\.?|number|#)\s*\d+|\bstandard horoscope\b|"
                            r"\bnative (?:was|is) born\b|\bborn on \d|\bthe native(?:'s)? (?:father|mother|wife) died\b",
                            re.I)


def is_example_horoscope(text: str) -> bool:
    """A passage discussing a particular person's chart from the book (Raman's "Chart No. 97")."""
    return bool(_EXAMPLE_CHART.search(text))


def gather_passages(settings: Settings, queries: list[str], limit: int,
                    embed_query: Callable[[str], list[float]] | None = None,
                    search_fn: Callable | None = None, exclude: re.Pattern | None = None) -> list[Passage]:
    """Library passages for the queries: the best new one per query in turn, then the second best,
    until `limit`. Round-robin, so every chart factor gets a passage before any gets two."""
    if limit <= 0 or not queries:
        return []
    if search_fn is None:
        from mmrag.retrieval.hybrid import _default_embedder, batch_embed_queries
        from mmrag.retrieval.hybrid import search as search_fn

        if embed_query is None:  # one batch for all queries, then the searches run in parallel
            embed_query = _default_embedder(settings)
            batch_embed_queries(settings, queries)
    k = 3 if len(queries) > 1 else limit
    with ThreadPoolExecutor(max_workers=min(4, len(queries))) as pool:
        per_query = list(pool.map(lambda q: search_fn(settings, q, "text", k, embed_query=embed_query), queries))
    out, seen = [], set()
    for rank in range(k):
        for q, hits in zip(queries, per_query):
            if rank < len(hits) and hits[rank].chunk_id not in seen and len(out) < limit:
                h = hits[rank]
                if exclude is not None and exclude.search(h.dense_text):
                    seen.add(h.chunk_id)
                    continue
                seen.add(h.chunk_id)
                pages = sorted({loc["page"] for loc in h.locations})
                text = " ".join(h.dense_text.split())[: settings.writer.passage_chars]
                out.append(Passage(n=len(out) + 1, chunk_id=h.chunk_id, source_file=h.source_file, pages=pages,
                                   text=text, query=q))
    return out


def _passages_block(passages: list[Passage]) -> str:
    if not passages:
        return ""
    lines = ["Passages from the books in the library. When you use one, cite it right after the sentence with "
             "its number in square brackets, for example [P3]. Use only these passages for classical views, and "
             "do not cite a number that is not listed here. If they say nothing about a factor, say so instead "
             "of inventing a source.", ""]
    for p in passages:
        pages = ", ".join(str(n) for n in p.pages)
        label = " [example horoscope of another person; never apply its details to this native]" \
            if is_example_horoscope(p.text) else ""
        lines.append(f"[P{p.n}] ({p.source_file}, page {pages}){label} {p.text}")
    return "\n".join(lines)


def reading_messages(settings: Settings, facts: str, passages: list[Passage], request: str) -> list[dict]:
    parts = ["Chart facts, computed by software. Use them exactly as given; never recalculate positions, "
             "houses or dasha dates.", facts, "", _passages_block(passages), "", f"Request: {request}"]
    return [{"role": "system", "content": system_prompt(settings)},
            {"role": "user", "content": "\n".join(x for x in parts if x is not None).strip()}]


def question_messages(settings: Settings, question: str, passages: list[Passage]) -> list[dict]:
    content = question if not passages else f"{_passages_block(passages)}\n\nQuestion: {question}"
    return [{"role": "system", "content": system_prompt(settings)}, {"role": "user", "content": content}]


# ---------------------------------------------------------------- checks

_CITE = re.compile(r"\[\s*P\s*(\d+)(?:\s*[,;]\s*P?\s*(\d+))*\s*\]", re.I)
_BARE_CITE = re.compile(r"(?<![\w\[])P(\d{1,2})\b")  # "covered in P5": a citation without brackets
_NUM = re.compile(r"P?\s*(\d+)", re.I)


BOOK_NAMES = {"Phaladeepika": r"Phala\s*-?\s*deepika|Phaldeepika|Phaladipika", "Saravali": r"Saravali",
              "Bhrigu Sutras": r"Bhrigu", "Brihat Jataka": r"Brihat\s+Jatak", "Jataka Parijata": r"Jataka\s+Parijata",
              "Raman": r"\bRaman\b|How\s+to\s+Judge\s+a\s+Horoscope", "Sanketa Nidhi": r"Sanketa\s*nidhi",
              "Chamatkar Chintamani": r"Chamatkar", "Uttara Kalamrita": r"Uttara\s+Kalamrita",
              "Brihat Parashara Hora Shastra": r"Parashara\s+Hora|\bBPHS\b", "Saturn: Friend and Guide":
              r"Saturn[-\s]+Friend"}


def link_named_books(text: str, passages: list[Passage], cited: list[Passage]) -> tuple[str, list[Passage]]:
    """The writer often names a book ("Phaldeepika describes...") instead of citing [P3]. Each named
    book is linked to the first given passage that is from it or quotes it, so the source can be
    checked; a book that no passage contains stays unlinked."""
    cited = list(cited)
    for pattern in BOOK_NAMES.values():
        m = re.search(rf"({pattern})(?!\s*\[)", text, re.I)
        if not m:
            continue
        hit = next((p for p in passages if re.search(pattern, p.text, re.I)
                    or re.search(pattern, p.source_file.replace("-", " "), re.I)), None)
        if hit is None:
            continue
        end = text.find(".", m.end())
        if f"[{hit.n}]" in text[m.end():end if end >= 0 else len(text)]:  # the sentence already cites it
            if hit not in cited:
                cited.append(hit)
            continue
        text = text[:m.end()] + f" [{hit.n}]" + text[m.end():]
        if hit not in cited:
            cited.append(hit)
    return text, cited


def resolve_citations(text: str, passages: list[Passage]) -> tuple[str, list[Passage], list[str]]:
    """[P3] (or a bare P3) -> [3] for passages that were given; unknown numbers are removed with a notice."""
    by_n = {p.n: p for p in passages}
    cited: list[Passage] = []
    unknown: set[int] = set()

    def keep(nums: list[int]) -> str:
        good = []
        for n in nums:
            if n in by_n:
                good.append(n)
                if by_n[n] not in cited:
                    cited.append(by_n[n])
            else:
                unknown.add(n)
        return "".join(f"[{n}]" for n in good)

    out = _CITE.sub(lambda m: keep([int(x) for x in _NUM.findall(m.group(0))]), text)
    out = _BARE_CITE.sub(lambda m: f"passage {keep([int(m.group(1))])}" if int(m.group(1)) in by_n
                         else keep([int(m.group(1))]), out)
    out = re.sub(r" +([.,;:])", r"\1", out)
    out, cited = link_named_books(out, passages, cited)
    notices = [f"Removed citation(s) to passage(s) the model was not given: {', '.join(f'P{n}' for n in sorted(unknown))}."] \
        if unknown else []
    return out, cited, notices


_B = r"(?:Sun|Moon|Mars|Mercury|Jupiter|Venus|Saturn|Rahu|Ketu)"
_BODIES = rf"(?P<bodies>{_B}(?:\s*(?:,|and|&)\s*{_B})*)"   # "Saturn", "Rahu and Ketu", "Venus, Mars and Mercury"
_LINK = r"(?:\s+(?:is|are|also|both|placed|posited|sits|sit|located|situated|together|here|retrograde|and|still))*"
_ORD = r"(\d{1,2})(?:st|nd|rd|th)"
_SIGN = "(" + "|".join(SIGNS) + ")"
_HOUSE_RE = re.compile(_BODIES + _LINK + r"\s+in\s+the\s+" + _ORD + r"\b(?!\s+(?:from|lord|to|and)\b)(?:\s+house)?")
_SIGN_RE = re.compile(_BODIES + _LINK + r"\s+in\s+" + _SIGN + r"\b")
_LAGNA_RE = re.compile(_SIGN + r"\s+(?:Lagna|Ascendant|ascendant|lagna|rising)|(?:Lagna|Ascendant|ascendant)\s+(?:is\s+|in\s+)?" + _SIGN)
_DIGNITY_RE = re.compile(_BODIES + _LINK + r"(?:\s*,)?\s+(exalted|debilitated|in\s+(?:its|his|her|their)\s+own\s+sign)\b")
_LORD_AFTER_RE = re.compile(rf"(?P<body>{_B}),?\s+(?:the\s+)?(?:lord|ruler)\s+of\s+the\s+{_ORD}(?:\s*(?:,|and)\s*(?:the\s+)?{_ORD})?")
_LORD_PAREN_RE = re.compile(rf"(?P<body>{_B})\s*\(\s*{_ORD}(?:\s*(?:,|and)\s*{_ORD})?\s+lords?")
_LORD_BEFORE_RE = re.compile(rf"\b{_ORD}\s+lord,?\s+(?:the\s+)?(?P<body>{_B})\b")
_DASHA_RE = re.compile(rf"(?P<body>{_B})\s+(?P<kind>maha|antar)\s*dasha[^()\n]{{0,30}}\((\d{{4}}-\d\d-\d\d)\s+(?:to|-|–)\s+(\d{{4}}-\d\d-\d\d)\)",
                       re.I)

_DASHA_YEARS_RE = re.compile(rf"(?P<body>{_B})\s+maha\s*dasha[^.()\n]{{0,25}}?\(?(?:from\s+)?(\d{{4}})\s*(?:-|–|to)\s*(\d{{4}})\b", re.I)
_HOUSE_KIND_RE = re.compile(rf"\b(?:the\s+)?{_ORD}(?:\s+house)?(?:\s+(?:is|being)|\s*,|\s*\()\s*(?:a|an|the)\s+"
                            r"(kendra|trikona|trine|angular|dusthana|upachaya)\b", re.I)
_HOUSE_KINDS = {"kendra": {1, 4, 7, 10}, "angular": {1, 4, 7, 10}, "trikona": {1, 5, 9}, "trine": {1, 5, 9},
                "dusthana": {6, 8, 12}, "upachaya": {3, 6, 10, 11}}


def _bodies(group: str) -> list[str]:
    return re.findall(_B, group)


def check_positions(text: str, chart: NatalChart) -> list[str]:
    """Every chart fact the text states that the computed chart contradicts: a planet's house or sign
    ('Saturn in the 7th house', 'Rahu and Ketu in the 1st', 'Mars in Leo'), the Lagna, a dignity
    ('Saturn is debilitated'), a lordship ('Saturn, lord of the 7th') and dasha dates. A book's
    general rule quoted in the text is flagged too; the notice says so. Aspects are not checked."""
    problems: list[str] = []
    pl = chart.placements
    for m in _HOUSE_RE.finditer(text):
        house = int(m.group(2))
        for body in _bodies(m.group("bodies")):
            if pl[body].house != house:
                problems.append(f"says {body} in the {ordinal(house)} house; the chart has it in the "
                                f"{ordinal(pl[body].house)}")
    for m in _SIGN_RE.finditer(text):
        sign = m.group(2)
        for body in _bodies(m.group("bodies")):
            if pl[body].sign != sign:
                problems.append(f"says {body} in {sign}; the chart has it in {pl[body].sign}")
    for m in _LAGNA_RE.finditer(text):
        sign = m.group(1) or m.group(2)
        if sign != chart.lagna.sign:
            problems.append(f"says {sign} Lagna; the chart's Lagna is {chart.lagna.sign}")
    for m in _DIGNITY_RE.finditer(text):
        claim = "own sign" if m.group(2).startswith("in") else m.group(2)
        for body in _bodies(m.group("bodies")):
            actual = getattr(pl[body], "dignity", None)
            if body not in ("Rahu", "Ketu") and actual != claim:
                problems.append(f"calls {body} {claim}; in the chart it is {actual or 'neither exalted, debilitated nor in its own sign'}")
    lord_claims = [(m.group("body"), [int(x) for x in m.groups()[1:] if x and x.isdigit()])
                   for r in (_LORD_AFTER_RE, _LORD_PAREN_RE) for m in r.finditer(text)]
    lord_claims += [(m.group("body"), [int(m.group(1))]) for m in _LORD_BEFORE_RE.finditer(text)]
    for body, houses in lord_claims:
        rules = getattr(pl[body], "lord_of", None)
        if rules is None or body in ("Rahu", "Ketu"):
            continue
        for h in houses:
            if h not in rules:
                problems.append(f"calls {body} lord of the {ordinal(h)}; it rules the "
                                f"{' and '.join(ordinal(x) for x in rules) or 'no house'}")
    dashas = getattr(chart, "dashas", None)
    if dashas:
        from datetime import date as _date

        from mmrag.astro.natal import antardashas

        periods = {("maha", d.lord, d.start, d.end) for d in dashas}
        periods |= {("antar", a.sub, a.start, a.end) for d in dashas for a in antardashas(d)}
        for m in _DASHA_RE.finditer(text):
            kind, body = m.group("kind").lower(), m.group("body")
            start, end = _date.fromisoformat(m.group(3)), _date.fromisoformat(m.group(4))
            if not any(k == kind and b == body and abs((s - start).days) <= 1 and abs((e - end).days) <= 1
                       for k, b, s, e in periods):
                problems.append(f"gives {body} {kind}dasha as {start} to {end}, which are not the chart's dates")
        for m in _DASHA_YEARS_RE.finditer(text):
            body, y1, y2 = m.group("body"), int(m.group(2)), int(m.group(3))
            if not any(d.lord == body and d.start.year == y1 and d.end.year == y2 for d in dashas):
                actual = next((f"{d.start.year}-{d.end.year}" for d in dashas if d.lord == body), "not in the sequence")
                problems.append(f"gives {body} mahadasha as {y1}-{y2}; the chart has {actual}")
    for m in _HOUSE_KIND_RE.finditer(text):
        house, kind = int(m.group(1)), m.group(2).lower()
        if house not in _HOUSE_KINDS[kind]:
            problems.append(f"calls the {ordinal(house)} house a {kind}, which it is not")
    unique = list(dict.fromkeys(problems))
    return [f"Position check: the text {p} (unless it is quoting a general rule)." for p in unique]


_CHILD_EXCLUDE = re.compile(r"\b(?:poverty|penury|daridra|beggar|widow|divorce|death|dies|die|longevity|maraka|"
                            r"adultery|prostitut)", re.I)


def reading_rules(stage, with_passages: bool, chart: NatalChart | None = None) -> str:
    """The rules, repeated at the end of the request, where the small model follows them best."""
    from mmrag.astro.analysis import neecha_bhanga
    rules = [f"The native is {stage.age} years old ({stage.stage}). Focus on {stage.focus}. Do not discuss "
             f"{stage.avoid}.",
             "Use only the dignities, aspects, yogas, Navamsa and cancellations listed in the facts. Do not name a "
             "yoga, an aspect, Vargottama or a cancellation that is not listed there.",
             "Counting: the sign a planet occupies is the 1st from it. Take every 'Nth from' position from the "
             "'Counted from' lines; never count houses yourself.",
             "Begin with the chart's strongest factors (exalted and own-sign planets, yogas, cancelled debility), "
             "then the challenges. Weigh them evenly.",
             "When a debilitated planet's debility is cancelled, say so and say by which rule; do not call it "
             "simply weak."]
    if chart is not None:
        nbs = neecha_bhanga(chart)
        cancelled = [nb.body for nb in nbs if nb.classical]
        later_only = [nb.body for nb in nbs if not nb.classical and nb.later]
        not_cancelled = [nb.body for nb in nbs if not nb.classical and not nb.later]
        parts = ([f"cancelled by a classical rule: {', '.join(cancelled)}"] if cancelled else []) + \
                ([f"cancelled only by later rules: {', '.join(later_only)}"] if later_only else []) + \
                ([f"NOT cancelled: {', '.join(not_cancelled)}"] if not_cancelled else [])
        if parts:
            rules.append("Debilitated planets in this chart: " + "; ".join(parts) + ". Do not move a cancellation "
                         "from one planet to another.")
    if with_passages:
        rules += ["Apply a rule from a passage only when every one of its conditions is shown in the facts; "
                  "otherwise leave the rule out entirely. Never list rules that do not apply.",
                  "Passages marked as an example horoscope describe other people; never apply their details to "
                  "this native.",
                  "Whenever you use a passage, put its number in square brackets right after the sentence, for "
                  "example [P3]; do not name the book instead."]
    return "Rules for this reading:\n" + "\n".join(f"{n}. {r}" for n, r in enumerate(rules, 1))


_FROM_SIGN_RE = re.compile(rf"\bthe\s+{_ORD}\s+(?:house\s+|sign\s+)?from\s+(?:the\s+)?(?P<ref>{_B}|Lagna|Ascendant)"
                           rf"(?:\s*\([^)]*\))?\s*(?:,\s*)?(?:is|=|falls in|being)\s+(?:in\s+)?{_SIGN}")
_IN_FROM_RE = re.compile(_BODIES + _LINK + rf"\s+in\s+the\s+{_ORD}\s+(?:house\s+)?from\s+(?:the\s+)?"
                         rf"(?P<ref>{_B}|Lagna|Ascendant)\b")
_ASPECT_HOUSE_RE = re.compile(rf"(?P<a>{_B})(?:\s+also)?\s+(?:aspects|fully aspects|casts\s+(?:its|his|her)\s+"
                              rf"(?:\w+\s+)?aspect\s+on)\s+the\s+{_ORD}(?:\s+house)?\b")
_ASPECT_BODY_RE = re.compile(rf"(?P<a>{_B})(?:\s+also)?\s+aspects\s+(?:the\s+)?(?P<b>{_B}|Lagna|Ascendant)\b")
_ASPECTED_RE = re.compile(rf"(?P<b>{_B}|Lagna|Ascendant|the\s+{_ORD}(?:\s+house)?)\s+(?:is|are)\s+(?:also\s+)?"
                          rf"aspected\s+by\s+(?P<a>{_B})\b")
_CANCELLED_RE = re.compile(rf"(?P<a>{_B})(?:'s)?(?:[^.]{{0,60}}?\bits)?\s+(?:debility|debilitation)\s+(?:is\s+)?"
                           rf"(?:also\s+|fully\s+|partly\s+)?(?:cancelled|canceled|relieved|removed)", re.I)
_CANCELLED_OF_RE = re.compile(rf"(?:cancelled|canceled)\s+debility\s+(?:of|for)\s+(?P<a>{_B})"
                              rf"|neecha\s+bhanga(?:\s+raja\s+yoga)?\s+(?:of|for)\s+(?P<b>{_B})", re.I)
_RECEIVES_RE = re.compile(rf"(?P<t>\bthe\s+{_ORD}(?:\s+house)?|\b{_B}|\bLagna)\b[^.]{{0,100}}?\breceives?\s+"
                          rf"(?:the\s+)?(?:full\s+)?aspects?\s+(?:of|from)\s+(?P<from>{_B}(?:\s*(?:,|and|&)\s*{_B})*)")
_JOINED_RE = re.compile(rf"(?P<a>{_B})\b(?:\s+is|\s+are|\s*,)?(?:\s+also)?\s+(?:joined|conjoined|accompanied)\s+"
                        rf"(?:by|with)\s+(?:a\s+\w+\s+\(?)?(?:the\s+)?(?P<with>{_B}(?:\s*(?:,|and|&)\s*(?:the\s+)?{_B})*)")
_VARGOTTAMA_RE = re.compile(rf"(?P<a>{_B})(?:\s+is|\s*,|\s+being)?(?:\s+also)?\s+vargottama", re.I)
_CHILD_TOPICS = re.compile(r"\b(poverty|penury|daridra|wealth|riches|career|profession|marriage|spouse|wife|"
                           r"husband|divorce|death|longevity)\b", re.I)
_NEGATED = re.compile(r"(?:\bno\b|\bnot\b|\bnor\b|\bwithout\b|\babsence\b|\bisn't\b|\bdoes not\b)[^.]{0,40}$", re.I)


def _house_of(ref: str, chart: NatalChart) -> str:
    return chart.lagna.sign if ref in ("Lagna", "Ascendant") else chart.placements[ref].sign


def check_analysis(text: str, chart: NatalChart, stage=None) -> list[str]:
    """Claims the computed analysis decides: positions counted from a planet, aspects, Vargottama,
    yogas the chart does not have, and adult topics in a child's reading."""
    from mmrag.astro.analysis import YOGA_NAMES, aspects, house_from, navamsa, sign_at, yogas

    problems: list[str] = []
    for m in _FROM_SIGN_RE.finditer(text):
        n, ref, sign = int(m.group(1)), m.group("ref"), m.group(len(m.groups()))
        actual = sign_at(_house_of(ref, chart), n)
        if sign != actual:
            problems.append(f"says the {ordinal(n)} from {ref} is {sign}; it is {actual}")
    for m in _IN_FROM_RE.finditer(text):
        n, ref = int(m.group(len(m.groups()) - 1)), m.group("ref")
        for body in _bodies(m.group("bodies")):
            actual = house_from(_house_of(ref, chart), chart.placements[body].sign)
            if body != ref and actual != n:
                problems.append(f"says {body} is in the {ordinal(n)} from {ref}; it is in the {ordinal(actual)}")
    asp = aspects(chart)
    def house_of(x: str) -> int:
        return 1 if x in ("Lagna", "Ascendant") else chart.placements[x].house
    for m in _ASPECT_HOUSE_RE.finditer(text):
        a, h = m.group("a"), int(m.group(2))
        if a in asp and h not in asp[a]:
            problems.append(f"says {a} aspects the {ordinal(h)} house; it aspects the "
                            f"{', '.join(ordinal(x) for x in asp[a])}")
    for m in _ASPECT_BODY_RE.finditer(text):
        a, b = m.group("a"), m.group("b")
        if a in asp and a != b and house_of(b) not in asp[a]:
            problems.append(f"says {a} aspects {b}; it does not (it aspects the "
                            f"{', '.join(ordinal(x) for x in asp[a])})")
    for m in _ASPECTED_RE.finditer(text):
        a, target = m.group("a"), m.group("b")
        h = int(re.search(r"\d+", target).group()) if target.startswith("the") else house_of(target)
        if a in asp and h not in asp[a]:
            problems.append(f"says {target} is aspected by {a}; it is not")
    from mmrag.astro.analysis import neecha_bhanga

    nbs = {nb.body: nb for nb in neecha_bhanga(chart)}
    claims = [m.group("a") for m in _CANCELLED_RE.finditer(text)]
    claims += [m.group("a") or m.group("b") for m in _CANCELLED_OF_RE.finditer(text)]
    for a in dict.fromkeys(claims):
        nb = nbs.get(a)
        if nb is None:
            problems.append(f"says {a}'s debility is cancelled; {a} is not debilitated")
        elif not nb.classical and not nb.later:
            problems.append(f"says {a}'s debility is cancelled; no rule cancels it in this chart")
    for m in _RECEIVES_RE.finditer(text):
        target = m.group("t")
        digits = re.search(r"\d+", target)
        h = int(digits.group()) if digits else house_of(target.strip())
        for a in _bodies(m.group("from")):
            if a in asp and h not in asp[a]:
                problems.append(f"says {target.strip()} receives the aspect of {a}; it does not "
                                f"({a} aspects the {', '.join(ordinal(x) for x in asp[a])})")
    for m in _JOINED_RE.finditer(text):
        a = m.group("a")
        for b in _bodies(m.group("with")):
            if b != a and chart.placements[b].sign != chart.placements[a].sign:
                problems.append(f"says {a} is joined by {b}; {a} is in {chart.placements[a].sign} and {b} in "
                                f"{chart.placements[b].sign}")
    for m in _VARGOTTAMA_RE.finditer(text):
        a = m.group("a")
        if navamsa(chart.placements[a].longitude) != chart.placements[a].sign:
            problems.append(f"calls {a} Vargottama; it is not (Navamsa "
                            f"{navamsa(chart.placements[a].longitude)})")
    present = {y.name.split(" (")[0].lower() for y in yogas(chart)}
    present |= {"vipareeta raja yoga"} if any("Vipareeta" in y.name for y in yogas(chart)) else set()
    for name in YOGA_NAMES:
        for m in re.finditer(rf"\b{re.escape(name)}(?:\s+(?:raja\s+)?yoga)?\b", text, re.I):
            if _NEGATED.search(text[max(0, m.start() - 60):m.start()]):
                continue
            if not any(name.lower() in y for y in present):
                problems.append(f"names {name} yoga, which this chart does not have")
                break
    if stage is not None and stage.stage == "child":
        topics = sorted({m.group(1).lower() for m in _CHILD_TOPICS.finditer(text)
                         if not _NEGATED.search(text[max(0, m.start() - 60):m.start()])})
        if topics:
            problems.append(f"discusses {', '.join(topics)} in a reading for a {stage.age}-year-old")
    unique = list(dict.fromkeys(problems))
    return [f"Analysis check: the text {p}." for p in unique]


# ---------------------------------------------------------------- the two entry points

def write_reading(settings: Settings, chart: NatalChart, question: str | None = None,
                  writer: WriterName = "astro", use_books: bool = True, on: date | None = None,
                  write: Callable[[list[dict]], Written] | None = None,
                  embed_query: Callable[[str], list[float]] | None = None,
                  search_fn: Callable | None = None) -> Reading:
    """A reading of `chart`, or an answer to `question` about it, from chart facts and library passages."""
    from mmrag.astro.analysis import analysis_facts, life_stage

    started = time.monotonic()
    on = on or date.today()
    stage = life_stage(chart.birth.when_local.date(), on)
    facts = chart.facts(on) + "\n\n" + analysis_facts(chart, on)
    queries = ([question] if question else []) + chart_queries(chart, on)
    passages = gather_passages(settings, queries, settings.writer.passages if use_books else 0, embed_query,
                               search_fn, exclude=_CHILD_EXCLUDE if stage.stage == "child" else None)
    request = (f"{question.strip()} Answer from this chart." if question else
               "Give a full reading of this chart.") + "\n\n" + reading_rules(stage, bool(passages), chart)
    written = (write or get_writer(settings, writer))(reading_messages(settings, facts, passages, request))
    text, cited, notices = resolve_citations(written.text, passages)
    notices += check_positions(text, chart)
    notices += check_analysis(text, chart, stage)
    if passages and not cited:
        notices.append("The reading cites none of the library passages it was given.")
    if not text:
        notices.append("The writer returned no text.")
    return Reading(text=text, model=written.model, passages=passages, cited=cited, notices=notices, chart=chart,
                   facts=facts, seconds=time.monotonic() - started, input_tokens=written.input_tokens,
                   output_tokens=written.output_tokens)


def answer_question(settings: Settings, question: str, writer: WriterName = "astro", use_books: bool = True,
                    write: Callable[[list[dict]], Written] | None = None,
                    embed_query: Callable[[str], list[float]] | None = None,
                    search_fn: Callable | None = None) -> Reading:
    """A plain astrology question, with library passages (use_books) or from the model alone."""
    started = time.monotonic()
    passages = gather_passages(settings, [question], min(settings.writer.passages, 6) if use_books else 0,
                               embed_query, search_fn)
    written = (write or get_writer(settings, writer))(question_messages(settings, question, passages))
    text, cited, notices = resolve_citations(written.text, passages)
    if passages and not cited:
        notices.append("The answer cites none of the library passages it was given.")
    return Reading(text=text, model=written.model, passages=passages, cited=cited, notices=notices,
                   seconds=time.monotonic() - started, input_tokens=written.input_tokens,
                   output_tokens=written.output_tokens)


def sources_markdown(reading: Reading) -> str:
    """The cited passages as a numbered source list."""
    return "\n".join(f"[{p.n}] {p.source_file}, page {', '.join(str(x) for x in p.pages)}" for p in reading.cited)
