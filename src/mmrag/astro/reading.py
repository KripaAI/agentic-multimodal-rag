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
    """What to look up in the books for this chart, most important first."""
    p, lagna = chart.placements, chart.lagna
    moon = p["Moon"]
    md, ad = chart.current(on or date.today())
    queries = [f"{lagna.sign} ascendant (Lagna) native", f"Moon in {moon.sign} in the {ordinal(moon.house)} house",
               f"born in {moon.nakshatra} nakshatra"]
    if md:
        queries.append(f"results of {md.lord} mahadasha" + (f" and {ad.sub} antardasha" if ad else ""))
    for body in BODIES:
        if body != "Moon":
            queries.append(f"{body} in the {ordinal(p[body].house)} house")
    return queries


def gather_passages(settings: Settings, queries: list[str], limit: int,
                    embed_query: Callable[[str], list[float]] | None = None,
                    search_fn: Callable | None = None) -> list[Passage]:
    """Library passages for the queries: the best new one per query in turn, then the second best,
    until `limit`. Round-robin, so every chart factor gets a passage before any gets two."""
    if limit <= 0 or not queries:
        return []
    if search_fn is None:
        from mmrag.retrieval.hybrid import search as search_fn
    k = 3 if len(queries) > 1 else limit
    per_query = [search_fn(settings, q, "text", k, embed_query=embed_query) for q in queries]
    out, seen = [], set()
    for rank in range(k):
        for q, hits in zip(queries, per_query):
            if rank < len(hits) and hits[rank].chunk_id not in seen and len(out) < limit:
                h = hits[rank]
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
        lines.append(f"[P{p.n}] ({p.source_file}, page {pages}) {p.text}")
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
_NUM = re.compile(r"P?\s*(\d+)", re.I)


def resolve_citations(text: str, passages: list[Passage]) -> tuple[str, list[Passage], list[str]]:
    """[P3] -> [3] for passages that were given; unknown numbers are removed with a notice."""
    by_n = {p.n: p for p in passages}
    cited: list[Passage] = []
    unknown: set[int] = set()

    def repl(m: re.Match) -> str:
        nums = [int(x) for x in _NUM.findall(m.group(0))]
        good = []
        for n in nums:
            if n in by_n:
                good.append(n)
                if by_n[n] not in cited:
                    cited.append(by_n[n])
            else:
                unknown.add(n)
        return "".join(f"[{n}]" for n in good)

    out = re.sub(r" +([.,;:])", r"\1", _CITE.sub(repl, text))
    notices = [f"Removed citation(s) to passage(s) the model was not given: {', '.join(f'P{n}' for n in sorted(unknown))}."] \
        if unknown else []
    return out, cited, notices


_BODY = r"(Sun|Moon|Mars|Mercury|Jupiter|Venus|Saturn|Rahu|Ketu)"
_SIGN = "(" + "|".join(SIGNS) + ")"
_HOUSE_RE = re.compile(_BODY + r"(?:\s+(?:is|placed|posited|sits|located|situated)){0,2}\s+in\s+(?:the\s+)?(\d{1,2})"
                       r"(?:st|nd|rd|th)\s+house")
_SIGN_RE = re.compile(_BODY + r"(?:\s+(?:is|placed|posited|sits|located|situated)){0,2}\s+in\s+" + _SIGN + r"\b")
_LAGNA_RE = re.compile(_SIGN + r"\s+(?:Lagna|Ascendant|ascendant|lagna|rising)|(?:Lagna|Ascendant|ascendant)\s+(?:is\s+|in\s+)?" + _SIGN)


def check_positions(text: str, chart: NatalChart) -> list[str]:
    """Every 'Saturn in the 7th house' / 'Mars in Leo' / 'Leo Lagna' the text states, compared with
    the computed chart. A book's general rule quoted in the text is flagged too; the notice says so."""
    problems: list[str] = []
    for m in _HOUSE_RE.finditer(text):
        body, house = m.group(1), int(m.group(2))
        if chart.placements[body].house != house:
            problems.append(f"says {body} in the {ordinal(house)} house; the chart has it in the "
                            f"{ordinal(chart.placements[body].house)}")
    for m in _SIGN_RE.finditer(text):
        body, sign = m.group(1), m.group(2)
        if chart.placements[body].sign != sign:
            problems.append(f"says {body} in {sign}; the chart has it in {chart.placements[body].sign}")
    for m in _LAGNA_RE.finditer(text):
        sign = m.group(1) or m.group(2)
        if sign != chart.lagna.sign:
            problems.append(f"says {sign} Lagna; the chart's Lagna is {chart.lagna.sign}")
    unique = list(dict.fromkeys(problems))
    return [f"Position check: the text {p} (unless it is quoting a general rule)." for p in unique]


# ---------------------------------------------------------------- the two entry points

def write_reading(settings: Settings, chart: NatalChart, question: str | None = None,
                  writer: WriterName = "astro", use_books: bool = True, on: date | None = None,
                  write: Callable[[list[dict]], Written] | None = None,
                  embed_query: Callable[[str], list[float]] | None = None,
                  search_fn: Callable | None = None) -> Reading:
    """A reading of `chart`, or an answer to `question` about it, from chart facts and library passages."""
    started = time.monotonic()
    facts = chart.facts(on)
    queries = ([question] if question else []) + chart_queries(chart, on)
    passages = gather_passages(settings, queries, settings.writer.passages if use_books else 0, embed_query,
                               search_fn)
    request = (f"{question.strip()} Answer from this chart." if question else
               "Give a full reading of this chart.")
    written = (write or get_writer(settings, writer))(reading_messages(settings, facts, passages, request))
    text, cited, notices = resolve_citations(written.text, passages)
    notices += check_positions(text, chart)
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
