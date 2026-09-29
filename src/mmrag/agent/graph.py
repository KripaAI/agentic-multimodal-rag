"""The agent as a LangGraph state graph (D3, spec §7.2, LLD §5.2).

    START → plan → agent ⇄ tools → compose → validate ─ok──────────────→ finish → END
                                               ├─fail─→ repair → validate
                                               └─fail after repair → give_up → finish

LangGraph is used for orchestration only: every OpenAI call goes through agent/llm.py (no
LangChain chat models or prebuilt agents); tools are plain functions from agent/tools.py.
Runtime objects (model client, database access) travel in the graph's *context*; only plain
data is in the state, which the checkpointer saves per thread (FR-23). Phase 9 adds
`recall_memory` and `remember` around this graph.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Literal

from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime
from pydantic import BaseModel, ConfigDict

from mmrag.agent.answer import Answer, HydratedAnswer
from mmrag.agent.ledger import EvidenceLedger
from mmrag.agent.llm import LLM, LLMReply
from mmrag.agent.state import AgentState, QuestionType
from mmrag.agent.tools import TOOL_SPECS, ToolCall, ToolContext, dispatch
from mmrag.agent.validator import Store, drop_failing_blocks, validate
from mmrag.charts.engine import ChartResult
from mmrag.config import PROJECT_ROOT, Settings
from mmrag.obs import get_tracer

PROMPTS_DIR = PROJECT_ROOT / "src" / "mmrag" / "agent" / "prompts"
LIMIT_NOTE = ("You reached the round limit for this question. Answer with what the evidence supports, and say "
              "clearly in the text and in `missing` what could not be found.")


class PlanOut(BaseModel):
    model_config = ConfigDict(extra="forbid")

    qtype: QuestionType
    note: str


@dataclass
class AgentDeps:
    """Runtime context for one question (never checkpointed)."""

    settings: Settings
    llm: LLM
    store: Store
    embed_query: Callable[[str], list[float]]
    chart_dir: Path
    source_note: Callable[[list[str]], str] | None = None

    def prompt(self, name: str) -> str:
        return (PROMPTS_DIR / f"{name}_{self.settings.agent.prompt_version}.md").read_text(encoding="utf-8")


# ---------------------------------------------------------------- helpers

def _system(deps: AgentDeps) -> dict:
    return {"role": "system", "content": deps.prompt("system")}


def _usage(state: AgentState, reply: LLMReply) -> dict:
    return {"tokens_in": state.get("tokens_in", 0) + reply.input_tokens,
            "tokens_out": state.get("tokens_out", 0) + reply.output_tokens}


def _charts_in(state: AgentState) -> dict[str, ChartResult]:
    out = {}
    for cid, d in (state.get("charts") or {}).items():
        out[cid] = ChartResult(**{**d, "png_path": Path(d["png_path"]) if d.get("png_path") else None})
    return out


def _charts_out(charts: dict[str, ChartResult]) -> dict:
    return {cid: {**asdict(c), "png_path": str(c.png_path) if c.png_path else None} for cid, c in charts.items()}


def _answered(messages: list[dict]) -> list[dict]:
    """Close any tool calls left without a result (OpenAI rejects them)."""
    done = {m["tool_call_id"] for m in messages if m.get("role") == "tool"}
    extra = [{"role": "tool", "tool_call_id": c["id"], "content": json.dumps({"error": "not run: round limit"})}
             for m in messages if m.get("tool_calls") for c in m["tool_calls"] if c["id"] not in done]
    return messages + extra


def _traced(name: str):
    def wrap(fn):
        def node(state, runtime):
            with get_tracer("mmrag.agent").start_as_current_span(name) as span:
                span.set_attribute("mmrag.round", state.get("round", 0))
                return fn(state, runtime)
        node.__name__ = fn.__name__
        return node
    return wrap


# ---------------------------------------------------------------- nodes

@_traced("agent.plan")
def plan(state: AgentState, runtime: Runtime[AgentDeps]) -> dict:
    """Classify the question and set the round limit (3, or 5 for multi-part)."""
    deps = runtime.context
    out, reply = deps.llm.structured([_system(deps), {"role": "system", "content": deps.prompt("plan")},
                                      *state["messages"]], PlanOut)
    a = deps.settings.agent
    return {"qtype": out.qtype, "round_limit": a.rounds_multi if out.qtype == "multi_part" else a.rounds_default,
            **_usage(state, reply)}


@_traced("agent.round")
def agent(state: AgentState, runtime: Runtime[AgentDeps]) -> dict:
    """The next tool calls (possibly several at once), or none when the evidence is enough."""
    deps = runtime.context
    reply = deps.llm.chat([_system(deps), *state["messages"]], TOOL_SPECS)
    update = _usage(state, reply)
    if reply.tool_calls:
        update["messages"] = [{"role": "assistant", "content": reply.content, "tool_calls": [
            {"id": c["id"], "type": "function", "function": {"name": c["name"], "arguments": json.dumps(c["arguments"])}}
            for c in reply.tool_calls]}]
    return update


@_traced("agent.tools")
def tools(state: AgentState, runtime: Runtime[AgentDeps]) -> dict:
    """Run the pending tool calls; record results in the messages and the ledger; round += 1."""
    deps = runtime.context
    last = state["messages"][-1]
    calls = [ToolCall(c["id"], c["function"]["name"], json.loads(c["function"]["arguments"])) for c in last["tool_calls"]]
    ledger = EvidenceLedger.from_dict(state.get("ledger"))
    ctx = ToolContext(settings=deps.settings, ledger=ledger, embed_query=deps.embed_query, chart_dir=deps.chart_dir,
                      charts=_charts_in(state), computed=sum(1 for k in ledger.items if k.startswith("compute:")),
                      source_note=deps.source_note)
    results = dispatch(calls, ctx)
    round_no = state.get("round", 0) + 1
    messages = [{"role": "tool", "tool_call_id": r.call_id, "content": r.content} for r in results]
    images = [r for r in results if r.image_path]
    if images:  # tool messages carry text only; images follow as one user message
        messages.append({"role": "user", "content": [
            {"type": "text", "text": "Images requested by " + ", ".join(r.name for r in images) + ":"},
            *[{"type": "image_path", "path": r.image_path} for r in images]]})
    log = [{"round": round_no, "name": c.name, "arguments": c.arguments, "error": r.is_error}
           for c, r in zip(calls, results)]
    return {"messages": messages, "round": round_no, "ledger": ledger.to_dict(), "charts": _charts_out(ctx.charts),
            "tool_log": state.get("tool_log", []) + log}


def _compose_messages(state: AgentState, deps: AgentDeps) -> list[dict]:
    at_limit = state.get("round", 0) >= state.get("round_limit", 1)
    instructions = deps.prompt("compose").replace("{limit_note}", LIMIT_NOTE if at_limit else "")
    return [_system(deps), *_answered(state["messages"]), {"role": "user", "content": instructions}]


@_traced("agent.compose")
def compose(state: AgentState, runtime: Runtime[AgentDeps]) -> dict:
    """Structured output: the Answer (model form), citing ids only."""
    deps = runtime.context
    answer, reply = deps.llm.structured(_compose_messages(state, deps), Answer, TOOL_SPECS)
    return {"answer": answer.model_dump(), **_usage(state, reply)}


def _validation(answer: Answer, state: AgentState, deps: AgentDeps) -> dict:
    r = validate(answer, EvidenceLedger.from_dict(state.get("ledger")), _charts_in(state), deps.store)
    return {"ok": r.ok, "failures": r.failures, "failed_blocks": sorted(r.failed_blocks),
            "answer": r.answer.model_dump() if r.answer else None}


@_traced("validator.validate")
def validate_node(state: AgentState, runtime: Runtime[AgentDeps]) -> dict:
    return {"validation": _validation(Answer.model_validate(state["answer"]), state, runtime.context)}


@_traced("agent.repair")
def repair(state: AgentState, runtime: Runtime[AgentDeps]) -> dict:
    """One more structured-output turn, given the validator's failures."""
    deps = runtime.context
    text = deps.prompt("repair").replace("{failures}", "\n".join(f"- {f}" for f in state["validation"]["failures"])) \
        .replace("{answer}", json.dumps(state["answer"], ensure_ascii=False))
    answer, reply = deps.llm.structured(_compose_messages(state, deps) + [{"role": "user", "content": text}],
                                        Answer, TOOL_SPECS)
    return {"answer": answer.model_dump(), "repaired": True, **_usage(state, reply)}


@_traced("agent.give_up")
def give_up(state: AgentState, runtime: Runtime[AgentDeps]) -> dict:
    """The repair failed too: drop the blocks that still break a rule, with a notice."""
    kept, notices = drop_failing_blocks(Answer.model_validate(state["answer"]),
                                        set(state["validation"]["failed_blocks"]))
    return {"answer": kept.model_dump(), "validation": _validation(kept, state, runtime.context),
            "notices": state.get("notices", []) + notices}


def finish(state: AgentState, runtime: Runtime[AgentDeps]) -> dict:
    """Add the answer to the conversation, so follow-up questions can refer to it."""
    answer = Answer.model_validate(state["answer"])
    text = "\n\n".join(b.markdown for b in answer.blocks if b.type == "text")
    shown = [f"{b.type} {getattr(b, 'element_id', None) or getattr(b, 'chart_id', '')}"
             for b in answer.blocks if b.type != "text"]
    return {"messages": [{"role": "assistant", "content": text + (f"\n\n(Shown: {', '.join(shown)})" if shown else "")}]}


# ---------------------------------------------------------------- edges

def after_agent(state: AgentState) -> Literal["tools", "compose"]:
    """tools while the model asks for tools and rounds remain; otherwise compose."""
    last = state["messages"][-1]
    wants_tools = last.get("role") == "assistant" and bool(last.get("tool_calls"))
    return "tools" if wants_tools and state.get("round", 0) < state.get("round_limit", 1) else "compose"


def after_tools(state: AgentState) -> Literal["agent", "compose"]:
    return "agent" if state.get("round", 0) < state.get("round_limit", 1) else "compose"


def after_validate(state: AgentState) -> Literal["finish", "repair", "give_up"]:
    if state["validation"]["ok"]:
        return "finish"
    return "give_up" if state.get("repaired") else "repair"


# ---------------------------------------------------------------- build and run

def build_graph(settings: Settings, checkpointer=None):
    """Compile the StateGraph with the nodes and edges above and the given checkpointer."""
    g = StateGraph(AgentState, context_schema=AgentDeps)
    for name, fn in [("plan", plan), ("agent", agent), ("tools", tools), ("compose", compose),
                     ("validate", validate_node), ("repair", repair), ("give_up", give_up), ("finish", finish)]:
        g.add_node(name, fn)
    g.add_edge(START, "plan")
    g.add_edge("plan", "agent")
    g.add_conditional_edges("agent", after_agent)
    g.add_conditional_edges("tools", after_tools)
    g.add_edge("compose", "validate")
    g.add_conditional_edges("validate", after_validate)
    g.add_edge("repair", "validate")
    g.add_edge("give_up", "finish")
    g.add_edge("finish", END)
    return g.compile(checkpointer=checkpointer)


def answer_question(app, deps: AgentDeps, question: str, thread_id: str) -> AgentState:
    """Run one question in a thread; per-question fields are reset, messages and ledger carry over."""
    start = {"messages": [{"role": "user", "content": question}], "question": question, "round": 0,
             "answer": None, "validation": None, "repaired": False, "notices": [], "tool_log": [],
             "charts": {}, "tokens_in": 0, "tokens_out": 0}
    return app.invoke(start, {"configurable": {"thread_id": thread_id}, "recursion_limit": 60}, context=deps)


@dataclass
class QueryRun:
    answer: HydratedAnswer
    thread_id: str
    trace_id: str
    model: str
    qtype: str
    rounds: int
    tool_calls: list[dict]
    input_tokens: int
    output_tokens: int
    cost_usd: float | None  # None when the model has no price in config
    latency_ms: int
    validator_result: Literal["ok", "repaired", "dropped_blocks"]


def run_query(question: str, settings: Settings, thread_id: str | None = None,
              model: str | None = None, llm: LLM | None = None,
              embed_query: Callable[[str], list[float]] | None = None) -> QueryRun:
    """Answer one question in a (new or existing) thread: one OpenTelemetry trace, one
    query_log row. `model` overrides agent.model (used by the model comparison); `llm` and
    `embed_query` replace the OpenAI clients in tests."""
    from langgraph.checkpoint.postgres import PostgresSaver

    from mmrag.agent.llm import OpenAILLM
    from mmrag.agent.validator import DbStore
    from mmrag.llm import get_client
    from mmrag.obs.querylog import cost_usd, log_query
    from mmrag.retrieval.hybrid import _default_embedder

    thread_id = thread_id or uuid.uuid4().hex[:12]
    model = model or (llm.model if llm else settings.agent.model)
    run_dir = settings.resolve(settings.paths.data_dir) / "answers" / "charts" / thread_id
    deps = AgentDeps(settings=settings, llm=llm or OpenAILLM(get_client(settings), model, settings.agent.effort),
                     store=DbStore(settings), embed_query=embed_query or _default_embedder(settings), chart_dir=run_dir)
    started = time.monotonic()
    with get_tracer("mmrag.agent").start_as_current_span("query") as span, \
            PostgresSaver.from_conn_string(settings.secrets.database_url.get_secret_value()) as saver:
        saver.setup()
        trace_id = format(span.get_span_context().trace_id, "032x")
        span.set_attribute("mmrag.thread_id", thread_id)
        span.set_attribute("gen_ai.request.model", model)
        state = answer_question(build_graph(settings, saver), deps, question, thread_id)
        span.set_attribute("mmrag.qtype", state.get("qtype", ""))
        span.set_attribute("mmrag.rounds_used", state.get("round", 0))
    result = "dropped_blocks" if state.get("notices") else "repaired" if state.get("repaired") else "ok"
    answer = HydratedAnswer.model_validate(state["validation"]["answer"])
    answer.notices = list(state.get("notices", []))
    run = QueryRun(answer=answer, thread_id=thread_id, trace_id=trace_id, model=model, qtype=state.get("qtype", ""),
                   rounds=state.get("round", 0), tool_calls=state.get("tool_log", []),
                   input_tokens=state.get("tokens_in", 0), output_tokens=state.get("tokens_out", 0),
                   cost_usd=cost_usd(model, state.get("tokens_in", 0), state.get("tokens_out", 0), settings),
                   latency_ms=int((time.monotonic() - started) * 1000), validator_result=result)
    log_query(question, run, settings)
    return run
