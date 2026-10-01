"""What is worth remembering about a user (Phase 9, LLD §5.11).

A low-cost model (`memory.model`) reads one finished turn and returns short statements about
the **user** — never about the documents. The prompts say so, and `_clean` enforces what a
prompt cannot: length, count, and that nothing carrying a citation id is stored, so a memory
can never smuggle document content back in as if it were evidence (P13).
"""

from __future__ import annotations

import re
from typing import Iterable

from pydantic import BaseModel, ConfigDict, Field

from mmrag.agent.llm import LLM
from mmrag.obs import get_logger

_log = get_logger("mmrag.memory")

MAX_STATEMENTS = 5
_ID = re.compile(r"\b[\w-]+:[\w-]+(?::[\w-]+)*\b")  # chunk and element ids: d:text:12, p3:table:1

SEMANTIC_PROMPT = """You keep a small set of durable notes about one user of a document assistant.

From the exchange below, return statements about the USER that will still be useful weeks from now:
- what they are working on or studying, their role, their level of expertise;
- how they like answers ("prefers charts", "wants short answers", "asks for the maths").

Rules:
- Nothing about the documents, and no facts, numbers, figures or quotations from them. Those are not
  memories; they are evidence, and they are stored elsewhere.
- Nothing about this one question unless it says something lasting about the user.
- No names, addresses, passwords or other personal details the user did not volunteer as a preference.
- One short sentence per statement, written in the third person ("Prefers charts to tables.").
- `subject` is a short lower-case key for what the statement is about ("output_format", "topic_focus",
  "expertise"), so a later statement on the same subject replaces this one.
- Return an empty list when the exchange says nothing lasting about the user. That is the normal case.

Existing notes (update a subject only when the exchange genuinely changes it):
{existing}

Question: {question}

Answer given: {answer}
"""

EPISODE_PROMPT = """Summarise one finished conversation between a user and a document assistant, so the
assistant can pick it up again weeks later.

Two or three sentences: what the user asked about, what was found, and what was left open or unanswered.
Write it as a record of the conversation ("The user asked about ..."), not as an answer. Do not repeat
figures, numbers or quotations from the documents, and do not include citation ids: this is a reminder of
what happened, never a source.

Conversation:
{turns}
"""


class Statement(BaseModel):
    model_config = ConfigDict(extra="forbid")

    subject: str = Field(description="short lower-case key, e.g. output_format")
    statement: str = Field(description="one short sentence about the user")


class Statements(BaseModel):
    model_config = ConfigDict(extra="forbid")

    statements: list[Statement]


class Episode(BaseModel):
    model_config = ConfigDict(extra="forbid")

    summary: str


def _clean(text: str, max_chars: int) -> str | None:
    """A statement the rules allow, or None. Citation ids are a sign of document content."""
    text = " ".join(text.split())
    if not text or len(text) > max_chars or _ID.search(text):
        return None
    return text


def extract_semantic(llm: LLM, question: str, answer_text: str, existing: Iterable[str] = (),
                     max_chars: int = 200) -> list[tuple[str, str]]:
    """(subject, statement) pairs worth keeping about the user. Never raises: a failed
    extraction costs a memory, not the answer the user already has."""
    prompt = SEMANTIC_PROMPT.format(existing="\n".join(f"- {e}" for e in existing) or "(none yet)",
                                    question=question.strip()[:2000], answer=answer_text.strip()[:4000])
    try:
        out, _ = llm.structured([{"role": "user", "content": prompt}], Statements)
    except Exception as e:  # noqa: BLE001 - memory is optional
        _log.warning("memory extraction failed (%s); nothing stored", e)
        return []
    pairs = []
    for s in out.statements[:MAX_STATEMENTS]:
        text = _clean(s.statement, max_chars)
        subject = " ".join(s.subject.split())[:64].lower()
        if text and subject:
            pairs.append((subject, text))
    return pairs


def summarize_episode(llm: LLM, turns: Iterable[tuple[str, str]], max_chars: int = 1000) -> str:
    """A few sentences about one finished conversation, from its (question, answer) turns."""
    text = "\n\n".join(f"Q: {q.strip()}\nA: {(a or '').strip()[:800]}" for q, a in turns)
    if not text.strip():
        return ""
    try:
        out, _ = llm.structured([{"role": "user", "content": EPISODE_PROMPT.format(turns=text[:8000])}], Episode)
    except Exception as e:  # noqa: BLE001 - memory is optional
        _log.warning("episode summary failed (%s); nothing stored", e)
        return ""
    return _ID.sub("", " ".join(out.summary.split()))[:max_chars].strip()
