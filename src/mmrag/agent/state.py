"""The LangGraph state carried between nodes and checkpointed per thread (LLD §5.2).

Only plain data lives here. `messages` accumulates over the whole thread (so follow-ups see
earlier turns) and so does `ledger`; the other fields are reset at the start of each question.
"""

from __future__ import annotations

import operator
from typing import Annotated, Literal, TypedDict

QuestionType = Literal["conceptual", "visual", "quantitative", "mixed", "multi_part"]


class AgentState(TypedDict, total=False):
    messages: Annotated[list[dict], operator.add]  # OpenAI-format messages, incl. tool calls and results
    question: str
    qtype: QuestionType
    round: int
    round_limit: int
    ledger: dict  # EvidenceLedger.to_dict()
    charts: dict  # chart_id -> serialised ChartResult
    answer: dict | None  # Answer (model form)
    validation: dict | None  # {"ok", "failures", "failed_blocks", "answer" (hydrated)}
    repaired: bool
    notices: list[str]
    tool_log: list[dict]  # [{"round", "name", "arguments", "error"}] for query_log
    tokens_in: int
    tokens_out: int
    memories: list[str]  # Phase 9: recalled user memories (context only, never evidence)
