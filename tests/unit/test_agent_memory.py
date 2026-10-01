"""The memory nodes inside the agent graph (plan Phase 9 tasks 4, 5, 9). Test-first.

A scripted model and a fake memory session: no API calls, no database. What is checked here
is the shape of the deal between memory and the agent — recalled notes reach the model as
context about the *user*, and an answer that treats one as evidence does not survive the
validator (P13).
"""

from __future__ import annotations

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from mmrag.agent import tools
from mmrag.agent.answer import Answer
from mmrag.agent.graph import MEMORY_HEADER, AgentDeps, PlanOut, answer_question, build_graph
from mmrag.agent.ledger import Evidence
from mmrag.agent.llm import LLMReply
from mmrag.memory import Memory

pytestmark = pytest.mark.unit


class FakeStore:
    def locations(self, ids):
        return {i: [{"element_id": i, "source_file": "t.pdf", "page": 3, "bbox": (0, 0, 1, 1)}]
                for i in ids if i.startswith("d:")}

    def element_info(self, ids):
        return {}


class FakeLLM:
    model = "fake-model"

    def __init__(self, answers):
        self.answers, self.calls, self.searched = list(answers), [], False

    def chat(self, messages, tools_spec):
        self.calls.append(("chat", messages))
        if self.searched:
            return LLMReply(content="enough", input_tokens=1, output_tokens=1)
        self.searched = True  # one search, so the ledger holds the evidence the answers cite
        return LLMReply(content=None, input_tokens=1, output_tokens=1,
                        tool_calls=[{"id": "c1", "name": "search_text", "arguments": {"query": "generation"}}])

    def structured(self, messages, schema, tools_spec=None):
        self.calls.append((schema.__name__, messages))
        if schema is PlanOut:
            return PlanOut(qtype="conceptual", note="search text"), LLMReply(None)
        return Answer.model_validate(self.answers.pop(0)), LLMReply(None)


class FakeMemory:
    """Stands in for mmrag.memory.MemorySession."""

    def __init__(self, recalled=()):
        self.recalled = list(recalled)
        self.written: list[tuple[str, str]] = []

    def recall(self, question):
        return self.recalled

    def remember(self, question, answer_text):
        self.written.append((question, answer_text))
        return []


@pytest.fixture(autouse=True)
def fake_search(monkeypatch):
    def search_text(ctx, query, k=8):
        ctx.ledger.add(Evidence(id="d:text:1", kind="chunk", text="four steps"))
        return [{"id": "d:text:1", "text": "four steps"}]

    monkeypatch.setitem(tools.TOOLS, "search_text", search_text)


def _answer(*ids, not_found=False):
    return {"not_found": not_found,
            "blocks": [{"type": "text", "markdown": "An answer.", "citations": [{"id": i} for i in ids]}]}


def _run(parse_settings, tmp_path, llm, memory=None, question="How does generation work?"):
    app = build_graph(parse_settings, InMemorySaver())
    deps = AgentDeps(settings=parse_settings, llm=llm, store=FakeStore(), embed_query=lambda q: [0.0],
                     chart_dir=tmp_path, source_note=lambda ids: "", memory=memory)
    return answer_question(app, deps, question, "t1")


def _prompts(llm, kind):
    return [m for k, msgs in llm.calls if k == kind for m in msgs if m.get("role") == "system"]


def test_recalled_memories_reach_the_planner_as_context_about_the_user(parse_settings, tmp_path):
    llm = FakeLLM([_answer("d:text:1")])
    memory = FakeMemory([Memory(key="output_format", kind="semantic", text="Prefers charts to tables.")])
    state = _run(parse_settings, tmp_path, llm, memory)
    assert state["memories"] == ["memory:semantic:output_format: Prefers charts to tables."]
    planner = "\n".join(m["content"] for m in _prompts(llm, "PlanOut"))
    assert "Prefers charts to tables." in planner and MEMORY_HEADER in planner


def test_the_memory_context_says_it_is_not_evidence(parse_settings, tmp_path):
    llm = FakeLLM([_answer("d:text:1")])
    _run(parse_settings, tmp_path, llm, FakeMemory([Memory(key="k", kind="semantic", text="Studies RLHF.")]))
    composer = "\n".join(m["content"] for m in _prompts(llm, "Answer"))
    assert "not evidence" in composer and "never cite them" in composer


def test_without_memories_the_prompts_are_unchanged(parse_settings, tmp_path):
    llm = FakeLLM([_answer("d:text:1")])
    state = _run(parse_settings, tmp_path, llm, FakeMemory([]))
    assert state["memories"] == []
    assert all(MEMORY_HEADER not in m["content"] for m in _prompts(llm, "PlanOut"))


def test_the_graph_runs_normally_with_no_memory_session_at_all(parse_settings, tmp_path):
    llm = FakeLLM([_answer("d:text:1")])
    state = _run(parse_settings, tmp_path, llm, memory=None)
    assert state["validation"]["ok"] and state.get("memories") == []


def test_a_validated_answer_is_offered_to_memory(parse_settings, tmp_path):
    llm, memory = FakeLLM([_answer("d:text:1")]), FakeMemory()
    _run(parse_settings, tmp_path, llm, memory, question="Chart the sizes")
    assert memory.written == [("Chart the sizes", "An answer.")]


def test_nothing_is_remembered_from_an_answer_with_removed_parts(parse_settings, tmp_path):
    """Both the first answer and the repair cite a memory, so the blocks are dropped."""
    llm = FakeLLM([_answer("memory:semantic:k"), _answer("memory:semantic:k")])
    memory = FakeMemory([Memory(key="k", kind="semantic", text="Prefers charts.")])
    state = _run(parse_settings, tmp_path, llm, memory)
    assert state["notices"] and memory.written == []


def test_nothing_is_remembered_from_a_not_found_answer(parse_settings, tmp_path):
    not_found = {"not_found": True,
                 "blocks": [{"type": "text", "markdown": "The documents do not cover this.", "citations": []}]}
    llm, memory = FakeLLM([not_found]), FakeMemory()
    state = _run(parse_settings, tmp_path, llm, memory)
    assert state["validation"]["ok"] and memory.written == []


def test_citing_a_recalled_memory_fails_validation(parse_settings, tmp_path):
    """P13 end to end: the model is given memory ids, and using one as a source is refused."""
    llm = FakeLLM([_answer("memory:semantic:k"), _answer("d:text:1")])
    memory = FakeMemory([Memory(key="k", kind="semantic", text="Prefers charts.")])
    state = _run(parse_settings, tmp_path, llm, memory)
    assert state["repaired"] and state["validation"]["ok"]  # repaired into citing the documents


def test_a_failing_memory_never_fails_the_question(parse_settings, tmp_path):
    """Whatever breaks in memory, the user still gets their answer."""

    class Broken(FakeMemory):
        def recall(self, question):
            raise RuntimeError("store is down")

        def remember(self, question, answer_text):
            raise RuntimeError("store is down")

    state = _run(parse_settings, tmp_path, FakeLLM([_answer("d:text:1")]), Broken())
    assert state["validation"]["ok"] and state["memories"] == []


def test_the_progress_label_appears_only_when_something_was_recalled(parse_settings, tmp_path):
    app = build_graph(parse_settings, InMemorySaver())

    def steps_with(memory):
        deps = AgentDeps(settings=parse_settings, llm=FakeLLM([_answer("d:text:1")]), store=FakeStore(),
                         embed_query=lambda q: [0.0], chart_dir=tmp_path, source_note=lambda ids: "", memory=memory)
        steps: list[str] = []
        answer_question(app, deps, "How does generation work?", f"t-{id(memory)}", on_step=steps.append)
        return steps

    assert "Recalling earlier conversations" in steps_with(FakeMemory([Memory(key="k", kind="semantic", text="x")]))
    assert "Recalling earlier conversations" not in steps_with(FakeMemory([]))
