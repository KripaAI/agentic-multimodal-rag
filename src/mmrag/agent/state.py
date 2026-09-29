"""The LangGraph state carried between nodes and checkpointed per thread (LLD §5.2)."""

from __future__ import annotations

from typing import Annotated, Literal, TypedDict

from langgraph.graph.message import add_messages

QuestionType = Literal["conceptual", "visual", "quantitative", "mixed", "multi_part"]


class AgentState(TypedDict, total=False):
    messages: Annotated[list, add_messages]  # the conversation, incl. tool calls and results
    question: str
    qtype: QuestionType
    round: int
    round_limit: int
    ledger: dict  # EvidenceLedger, serialised (the checkpointer stores plain data)
    charts: dict  # chart engine results by chart_id
    answer: dict | None  # Answer (model form), serialised
    validation: dict | None  # ValidationResult, serialised
    repaired: bool
    memories: list[str]  # Phase 9: recalled user memories (context only, never evidence)
