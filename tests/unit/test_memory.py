"""Long-term memory: namespaces, update rules, extraction and deletion (plan Phase 9 task 9).

Test-first, with an in-memory store and a scripted model: no database and no API calls. The
rule these tests exist for is P13 — memory personalises, documents inform — so most of them
are about what must *not* end up in a memory, and about one user never seeing another's.
"""

from __future__ import annotations

import hashlib

import pytest
from langgraph.store.memory import InMemoryStore

from mmrag import memory
from mmrag.memory.extract import Episode, Statement, Statements, extract_semantic, summarize_episode

pytestmark = pytest.mark.unit

ALICE, BOB = "11111111-1111-1111-1111-111111111111", "22222222-2222-2222-2222-222222222222"


def _embed(texts: list[str]) -> list[list[float]]:
    """A deterministic stand-in for the OpenAI embedder: similar words, similar vectors."""
    out = []
    for text in texts:
        vector = [0.0] * 16
        for word in text.lower().split():
            vector[int(hashlib.sha256(word.encode()).hexdigest(), 16) % 16] += 1.0
        out.append(vector)
    return out


@pytest.fixture
def store():
    return InMemoryStore(index={"dims": 16, "embed": _embed, "fields": ["text"]})


class FakeLLM:
    """Returns the scripted structured outputs; records the prompts it was given."""

    model = "fake-memory-model"

    def __init__(self, *outputs):
        self.outputs, self.prompts = list(outputs), []

    def chat(self, messages, tools):  # pragma: no cover - memory never calls chat
        raise AssertionError("memory must not use the tool-calling model")

    def structured(self, messages, schema, tools=None):
        self.prompts.append(messages[-1]["content"])
        if not self.outputs:
            raise AssertionError("no scripted output left")
        return self.outputs.pop(0), object()


def _statements(*pairs):
    return Statements(statements=[Statement(subject=s, statement=t) for s, t in pairs])


# ---------------------------------------------------------------- namespaces (NFR-13)

def test_namespace_is_per_user_and_kind():
    assert memory.namespace(ALICE, "semantic") == ("memories", ALICE, "semantic")
    with pytest.raises(ValueError):
        memory.namespace("", "semantic")
    with pytest.raises(ValueError):
        memory.namespace(ALICE, "everything")


def test_one_user_never_sees_another_users_memories(store):
    memory.remember_statements(store, ALICE, [("topic_focus", "Works on RLHF.")])
    memory.remember_statements(store, BOB, [("topic_focus", "Works on tokenizers.")])
    assert [m.text for m in memory.list_memories(store, ALICE)] == ["Works on RLHF."]
    assert [m.text for m in memory.recall(store, BOB, "what am I working on?", 5)] == ["Works on tokenizers."]


def test_forgetting_one_user_leaves_the_other_untouched(store):
    memory.remember_statements(store, ALICE, [("a", "Alice note.")])
    memory.remember_statements(store, BOB, [("b", "Bob note.")])
    assert memory.forget_all(store, ALICE) == 1
    assert memory.list_memories(store, ALICE) == []
    assert len(memory.list_memories(store, BOB)) == 1


# ---------------------------------------------------------------- the update rule

def test_a_new_statement_on_the_same_subject_replaces_the_old_one(store):
    memory.remember_statements(store, ALICE, [("output_format", "Prefers tables.")])
    first = memory.list_memories(store, ALICE)[0]
    memory.remember_statements(store, ALICE, [("output_format", "Prefers charts.")])
    items = memory.list_memories(store, ALICE)
    assert [m.text for m in items] == ["Prefers charts."]
    assert items[0].created_at == first.created_at  # the same note, updated: not a second one


def test_different_subjects_are_kept_side_by_side(store):
    memory.remember_statements(store, ALICE, [("output_format", "Prefers charts."), ("topic_focus", "Studies RLHF.")])
    assert len(memory.list_memories(store, ALICE)) == 2


def test_subjects_become_stable_keys(store):
    memory.remember_statements(store, ALICE, [("Output Format", "Prefers charts.")])
    memory.remember_statements(store, ALICE, [("output format", "Prefers charts and tables.")])
    assert [m.key for m in memory.list_memories(store, ALICE)] == ["output_format"]


def test_episodes_are_keyed_by_thread_so_a_sweep_can_be_re_run(store):
    memory.remember_episode(store, ALICE, "thread-1", "The user asked about attention.")
    memory.remember_episode(store, ALICE, "thread-1", "The user asked about attention and KV caches.")
    items = memory.list_memories(store, ALICE, memory.EPISODIC)
    assert len(items) == 1 and "KV caches" in items[0].text


def test_an_empty_summary_is_not_stored(store):
    assert memory.remember_episode(store, ALICE, "thread-1", "   ") is None
    assert memory.list_memories(store, ALICE) == []


# ---------------------------------------------------------------- recall

def test_recall_prefers_the_memories_that_match_the_question(store):
    memory.remember_statements(store, ALICE, [("output_format", "Prefers charts to tables."),
                                              ("topic_focus", "Studies reinforcement learning.")])
    assert "charts" in memory.recall(store, ALICE, "prefers charts", 1)[0].text


def test_recall_gives_every_memory_an_id_the_validator_refuses(store):
    memory.remember_statements(store, ALICE, [("output_format", "Prefers charts.")])
    assert memory.recall(store, ALICE, "charts", 3)[0].id.startswith("memory:")


def test_a_broken_store_costs_a_memory_not_an_answer():
    class Broken:
        def search(self, *a, **k):
            raise RuntimeError("database is down")

    assert memory.recall(Broken(), ALICE, "anything", 3) == []


def test_without_a_store_every_entry_point_is_a_no_op():
    assert memory.recall(None, ALICE, "q", 3) == [] and memory.list_memories(None, ALICE) == []
    assert memory.remember_statements(None, ALICE, [("a", "b")]) == [] and memory.forget_all(None, ALICE) == 0


# ---------------------------------------------------------------- extraction (P13)

def test_extraction_returns_subject_and_statement_pairs():
    llm = FakeLLM(_statements(("output_format", "Prefers charts to tables.")))
    assert extract_semantic(llm, "Chart the sizes", "Here is a chart.") == [("output_format",
                                                                             "Prefers charts to tables.")]


def test_a_statement_carrying_a_citation_id_is_never_stored():
    """Document content must not re-enter as a memory, whatever the model returns (P13)."""
    llm = FakeLLM(_statements(("finding", "The paper says d:text:12 reports 40%."),
                              ("output_format", "Prefers charts.")))
    assert extract_semantic(llm, "q", "a") == [("output_format", "Prefers charts.")]


def test_over_long_statements_are_dropped():
    llm = FakeLLM(_statements(("rambling", "x" * 500), ("ok", "Short enough.")))
    assert extract_semantic(llm, "q", "a", max_chars=200) == [("ok", "Short enough.")]


def test_nothing_worth_remembering_stores_nothing():
    assert extract_semantic(FakeLLM(_statements()), "q", "a") == []


def test_existing_notes_are_shown_to_the_extractor_so_it_can_update_them():
    llm = FakeLLM(_statements())
    extract_semantic(llm, "q", "a", existing=["Prefers charts."])
    assert "Prefers charts." in llm.prompts[0]


def test_a_failing_extractor_is_not_an_error():
    class Failing(FakeLLM):
        def structured(self, messages, schema, tools=None):
            raise RuntimeError("rate limited")

    assert extract_semantic(Failing(), "q", "a") == []


def test_episode_summaries_drop_citation_ids():
    llm = FakeLLM(Episode(summary="The user asked about KV caches (d:text:9) and got sizes."))
    summary = summarize_episode(llm, [("How big is the KV cache?", "About 2 GB.")])
    assert "d:text:9" not in summary and "KV caches" in summary


def test_an_empty_conversation_is_not_summarised():
    assert summarize_episode(FakeLLM(), []) == ""


# ---------------------------------------------------------------- the session the agent holds

def test_the_extractor_is_built_only_when_something_is_written(store):
    built = []

    def factory():
        built.append(1)
        return FakeLLM(_statements(("output_format", "Prefers charts.")))

    session = memory.MemorySession(store=store, user_id=ALICE, llm_factory=factory)
    session.recall("anything")
    assert built == []  # recalling, listing and deleting never need the extractor
    session.remember("Chart it", "Here is a chart.")
    assert built == [1] and len(memory.list_memories(store, ALICE)) == 1


def test_an_extractor_that_cannot_be_built_costs_a_memory_not_an_answer(store):
    def factory():
        raise RuntimeError("OPENAI_API_KEY is not set")

    session = memory.MemorySession(store=store, user_id=ALICE, llm_factory=factory)
    assert session.remember("q", "a") == []
