"""The agent as a LangGraph state graph (D3, spec §7.2, LLD §5.2).

    START → plan → agent ⇄ tools → compose → validate → END
                                               ↳ repair → validate (once)

LangGraph is used for orchestration only: every node calls the OpenAI SDK directly (no
LangChain chat models or prebuilt agents); tools are plain functions from agent/tools.py.
The Postgres checkpointer keeps each conversation as a thread (FR-23). Phase 9 adds the
`recall_memory` and `remember` nodes around this graph.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from mmrag.agent.answer import HydratedAnswer
from mmrag.agent.state import AgentState
from mmrag.config import Settings

# ---------------------------------------------------------------- nodes


def plan(state: AgentState, config) -> dict:
    """OpenAI: classify the question (conceptual · visual · quantitative · mixed · multi_part)
    and set round_limit (agent.rounds_default, or agent.rounds_multi for multi-part)."""
    raise NotImplementedError


def agent(state: AgentState, config) -> dict:
    """OpenAI with TOOL_SPECS: the next tool calls (possibly several at once), or none."""
    raise NotImplementedError


def tools(state: AgentState, config) -> dict:
    """Run the pending tool calls with tools.dispatch; add results to messages and the ledger;
    round += 1."""
    raise NotImplementedError


def compose(state: AgentState, config) -> dict:
    """OpenAI structured output: the Answer (model form). At the round limit, the prompt says
    to answer with what was found and state what is missing."""
    raise NotImplementedError


def validate_node(state: AgentState, config) -> dict:
    """validator.validate; stores the ValidationResult."""
    raise NotImplementedError


def repair(state: AgentState, config) -> dict:
    """OpenAI structured output again, given the validator's failures; repaired = True."""
    raise NotImplementedError


# ---------------------------------------------------------------- edges


def after_agent(state: AgentState) -> Literal["tools", "compose"]:
    """tools while the model asks for tools and round < round_limit; otherwise compose."""
    raise NotImplementedError


def after_validate(state: AgentState) -> Literal["repair", "__end__"]:
    """END when valid; repair if invalid and not yet repaired; END otherwise (failing blocks
    dropped with a notice)."""
    raise NotImplementedError


# ---------------------------------------------------------------- build and run


def build_graph(settings: Settings, checkpointer=None):
    """Compile the StateGraph with the nodes and edges above and the given checkpointer."""
    raise NotImplementedError


@dataclass
class QueryRun:
    answer: HydratedAnswer
    thread_id: str
    trace_id: str
    rounds: int
    tool_calls: list[dict]
    input_tokens: int
    output_tokens: int
    cost_usd: float | None  # None when the model has no price in config
    latency_ms: int
    validator_result: Literal["ok", "repaired", "dropped_blocks"]


def run_query(question: str, settings: Settings, thread_id: str | None = None,
              model: str | None = None) -> QueryRun:
    """Answer one question in a (new or existing) thread: one OpenTelemetry trace, one
    query_log row. `model` overrides agent.model (used by the model comparison)."""
    raise NotImplementedError
