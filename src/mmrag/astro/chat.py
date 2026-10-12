"""Birth-chart readings from the chat (Phase 11).

People type their birth details into the chat ("my date of birth is 10-04-1981, born in Vaishali at
12:30 PM, tell me my future"). The document agent cannot calculate a chart, so it answered "no
evidence" every time. Here such a message is recognised, the details are read out of it by
`agent.model` (structured output: date, time, place, its coordinates and time zone), the chart is
computed by code, and the fine-tuned model writes the reading. The answer states exactly which
details were used, so a misread date or place is visible at once.
"""

from __future__ import annotations

from typing import Callable

from pydantic import BaseModel, ConfigDict, Field

from mmrag.astro.natal import BODIES, ChartInputError, compute_chart, dms, parse_birth
from mmrag.astro.reading import Reading, asks_about_own_chart, write_reading
from mmrag.config import Settings

EXTRACT_PROMPT = """Read birth details out of the user's message for a Vedic birth chart.
- date: the birth date as YYYY-MM-DD. Users in India write day first: 04-01-1983 is 1983-01-04 and
  10-04-1981 is 1981-04-10. Null if no birth date is given.
- time: the local birth time as HH:MM on the 24-hour clock (12:30 PM -> 12:30, 04:46 PM -> 16:46).
  Null if no time is given; never guess a time.
- place: the birth place as written, with its state or country if you know it. Null if none.
- latitude and longitude: the place's coordinates in decimal degrees (north and east positive), to
  two decimals. Null if there is no place or you do not know where it is.
- timezone: the IANA time zone of the place, e.g. Asia/Kolkata. Null if there is no place.
- name: the person's name, if given.
- question: what the person wants to know, in a few words (e.g. "career", "future in general")."""


class BirthDetails(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str | None = None
    date: str | None = None
    time: str | None = None
    place: str | None = None
    latitude: float | None = Field(default=None, ge=-90, le=90)
    longitude: float | None = Field(default=None, ge=-180, le=180)
    timezone: str | None = None
    question: str | None = None


def extract_birth_details(settings: Settings, message: str) -> BirthDetails:
    """The birth details in a chat message, read by agent.model."""
    from mmrag.agent.llm import OpenAILLM
    from mmrag.llm import get_client

    llm = OpenAILLM(get_client(settings), settings.agent.model, settings.agent.effort, max_tokens=800)
    details, _ = llm.structured([{"role": "system", "content": EXTRACT_PROMPT},
                                 {"role": "user", "content": message}], BirthDetails)
    return details


def _text(markdown: str) -> dict:
    return {"type": "text", "content": {"markdown": markdown}, "citations": []}


def _chart_table(reading: Reading) -> str:
    c = reading.chart
    rows = [f"| Lagna | {c.lagna.sign} | {dms(c.lagna.degree)} | 1 | {c.lagna.nakshatra} {c.lagna.pada} |"]
    for b in BODIES:
        p = c.placements[b]
        flags = ", ".join(x for x in ("R" if p.retrograde and b not in ("Rahu", "Ketu") else "", p.dignity or "",
                                      "combust" if p.combust else "") if x)
        rows.append(f"| {b} | {p.sign} | {dms(p.degree)} | {p.house} | {p.nakshatra} {p.pada} |"
                    + (f" {flags}" if flags else ""))
    return "| | sign | degree | house | nakshatra |\n|---|---|---|---|---|\n" + "\n".join(rows)


def chart_reply(settings: Settings, message: str,
                extract: Callable[[Settings, str], BirthDetails] | None = None,
                reading_fn: Callable[..., Reading] | None = None) -> dict | None:
    """An answer (the chat's answer format) for a message carrying the asker's birth details, or None
    when the message is not about their own chart and the document agent should answer it."""
    if not asks_about_own_chart(message):
        return None
    d = (extract or extract_birth_details)(settings, message)
    if not d.date:
        return None  # e.g. "what is my lagna" with no data: the agent (and the UI hint) handle it
    missing = [label for label, value in (("the time of birth (as exact as you know it)", d.time),
                                          ("the place of birth", d.latitude is not None and d.longitude is not None
                                           and d.timezone)) if not value]
    if missing:
        return {"blocks": [_text(
            f"To read your chart I also need {' and '.join(missing)}. The birth time decides the Lagna and the "
            "houses, so a reading without it would be wrong. Please send all of it in one message, for example: "
            "*born 10-04-1981 at 12:30 PM in Vaishali, Bihar*.")], "notices": [], "sources": []}
    try:
        chart = compute_chart(parse_birth(d.date, d.time, d.timezone, d.latitude, d.longitude, d.name, d.place))
    except ChartInputError as e:
        return {"blocks": [_text(f"I could not calculate the chart: {e}.")], "notices": [], "sources": []}
    writer = "astro" if settings.secrets.writer_base_url and settings.secrets.writer_api_key else "openai"
    reading = (reading_fn or write_reading)(settings, chart, question=d.question, writer=writer)
    who = f"**{d.name}**, born" if d.name else "Born"
    used = (f"{who} {chart.birth.when_local:%d %B %Y at %H:%M} in {d.place} ({d.latitude:.2f}°, "
            f"{d.longitude:.2f}°, {d.timezone}). If any of this is wrong, send all the details again, corrected.")
    blocks = [_text(used), _text(_chart_table(reading)), _text(reading.text or "_(the model returned no text)_")]
    if reading.cited:
        blocks.append(_text("**Sources**  \n" + "  \n".join(
            f"[{p.n}] {p.source_file}, page {', '.join(str(x) for x in p.pages)}" for p in reading.cited)))
    blocks.append(_text(f"_Chart calculated by software; reading written by {reading.model} in "
                        f"{reading.seconds:.0f} s._"))
    return {"blocks": blocks, "notices": list(reading.notices), "sources": []}
