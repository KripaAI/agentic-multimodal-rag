"""The LangGraph agent with a scripted fake model: no API calls (plan Phase 4 task 8). Test-first."""

from __future__ import annotations

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from mmrag.agent import tools
from mmrag.agent.answer import Answer
from mmrag.agent.graph import AgentDeps, PlanOut, answer_question, build_graph
from mmrag.agent.ledger import Evidence
from mmrag.agent.llm import LLMReply

pytestmark = pytest.mark.unit


class FakeStore:
    def locations(self, ids):
        return {i: [{"element_id": i, "source_file": "t.pdf", "page": 3, "bbox": (0, 0, 1, 1)}]
                for i in ids if i.startswith("d:")}

    def element_info(self, ids):
        return {}


class FakeLLM:
    model = "fake-model"

    def __init__(self, qtype="conceptual", chats=(), answers=()):
        self.qtype, self.chats, self.answers = qtype, list(chats), list(answers)
        self.calls = []  # (kind, messages)

    def chat(self, messages, tools_spec):
        self.calls.append(("chat", messages))
        return self.chats.pop(0) if self.chats else LLMReply(content="enough", input_tokens=10, output_tokens=2)

    def structured(self, messages, schema, tools_spec=None):
        self.calls.append((schema.__name__, messages))
        if schema is PlanOut:
            return PlanOut(qtype=self.qtype, note="search text"), LLMReply(None, input_tokens=5, output_tokens=1)
        return Answer.model_validate(self.answers.pop(0)), LLMReply(None, input_tokens=20, output_tokens=8)


def _call(n, query="loop"):
    return LLMReply(content=None, tool_calls=[{"id": f"c{n}", "name": "search_text", "arguments": {"query": query}}],
                    input_tokens=10, output_tokens=3)


def _cited(*ids):
    return {"blocks": [{"type": "text", "markdown": "An answer.", "citations": [{"id": i} for i in ids]}]}


@pytest.fixture(autouse=True)
def fake_search(monkeypatch):
    def search_text(ctx, query, k=8):
        ctx.ledger.add(Evidence(id="d:text:1", kind="chunk", text="four steps"))
        return [{"id": "d:text:1", "text": "four steps"}]

    monkeypatch.setitem(tools.TOOLS, "search_text", search_text)


def _run(llm, parse_settings, tmp_path, question="How does generation work?", thread="t1", app=None):
    app = app or build_graph(parse_settings, InMemorySaver())
    deps = AgentDeps(settings=parse_settings, llm=llm, store=FakeStore(), embed_query=lambda q: [0.0],
                     chart_dir=tmp_path, source_note=lambda ids: "")
    return app, answer_question(app, deps, question, thread)


def test_plan_search_compose_validate(parse_settings, tmp_path):
    llm = FakeLLM(chats=[_call(1)], answers=[_cited("d:text:1")])
    _, state = _run(llm, parse_settings, tmp_path)
    assert state["validation"]["ok"] and state["round"] == 1 and state["qtype"] == "conceptual"
    assert [t["name"] for t in state["tool_log"]] == ["search_text"]
    assert state["tokens_in"] == 5 + 10 + 10 + 20 and state["tokens_out"] == 1 + 3 + 2 + 8
    assert state["validation"]["answer"]["blocks"][0]["citations"][0]["locations"][0]["page"] == 3


def test_round_limit_forces_compose_with_every_tool_call_answered(parse_settings, tmp_path):
    llm = FakeLLM(chats=[_call(n) for n in range(1, 10)], answers=[_cited("d:text:1")])
    _, state = _run(llm, parse_settings, tmp_path)
    assert state["round"] == parse_settings.agent.rounds_default
    compose_msgs = next(m for kind, m in llm.calls if kind == "Answer")
    assert "round limit" in compose_msgs[0]["content"].lower() + compose_msgs[-1]["content"].lower()
    asked = {c["id"] for m in compose_msgs if m.get("tool_calls") for c in m["tool_calls"]}
    answered = {m["tool_call_id"] for m in compose_msgs if m.get("role") == "tool"}
    assert asked == answered  # OpenAI rejects unanswered tool calls


def test_multi_part_questions_get_more_rounds(parse_settings, tmp_path):
    llm = FakeLLM(qtype="multi_part", chats=[_call(n) for n in range(1, 10)], answers=[_cited("d:text:1")])
    _, state = _run(llm, parse_settings, tmp_path)
    assert state["round_limit"] == parse_settings.agent.rounds_multi == state["round"]


def test_a_failed_answer_is_repaired_once(parse_settings, tmp_path):
    llm = FakeLLM(chats=[_call(1)], answers=[_cited("d:p99:text:9"), _cited("d:text:1")])
    _, state = _run(llm, parse_settings, tmp_path)
    assert state["repaired"] and state["validation"]["ok"]
    repair_msgs = [m for kind, m in llm.calls if kind == "Answer"][1]
    assert "d:p99:text:9" in repair_msgs[-1]["content"]  # the validator's failure is shown to the model


def test_after_a_failed_repair_the_bad_blocks_are_dropped(parse_settings, tmp_path):
    two_blocks = {"blocks": [_cited("d:text:1")["blocks"][0], _cited("d:p99:text:9")["blocks"][0]]}
    llm = FakeLLM(chats=[_call(1)], answers=[two_blocks, two_blocks])
    _, state = _run(llm, parse_settings, tmp_path)
    assert state["validation"]["ok"] and len(state["validation"]["answer"]["blocks"]) == 1
    assert state["notices"]


def test_a_follow_up_in_the_same_thread_sees_the_first_turn(parse_settings, tmp_path):
    llm = FakeLLM(chats=[_call(1)], answers=[_cited("d:text:1"), _cited("d:text:1")])
    app, _ = _run(llm, parse_settings, tmp_path, question="How does generation work?")
    llm.calls.clear()
    _, state = _run(llm, parse_settings, tmp_path, question="And the second step?", app=app)
    first_chat = next(m for kind, m in llm.calls if kind == "chat")
    text = " ".join(str(m.get("content")) for m in first_chat)
    assert "How does generation work?" in text and "And the second step?" in text
    assert state["round"] == 0 and state["question"] == "And the second step?"  # per-question fields reset
    assert "d:text:1" in state["ledger"]  # evidence carries over within the thread


def test_a_new_thread_starts_fresh(parse_settings, tmp_path):
    llm = FakeLLM(chats=[_call(1)], answers=[_cited("d:text:1"), _cited("d:text:1")])
    app, _ = _run(llm, parse_settings, tmp_path, question="How does generation work?", thread="a")
    llm.calls.clear()
    llm.chats.append(_call(2))  # evidence is per thread: the new thread must search for its own
    _, state = _run(llm, parse_settings, tmp_path, question="Something else?", thread="b", app=app)
    assert state["validation"]["ok"] and not state["repaired"]
    first_chat = next(m for kind, m in llm.calls if kind == "chat")
    assert "How does generation work?" not in " ".join(str(m.get("content")) for m in first_chat)
