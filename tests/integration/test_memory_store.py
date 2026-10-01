"""Long-term memory against a real PostgreSQL store (plan Phase 9 task 9).

The embedder is a local stand-in, so these tests need the database but no API key. What they
prove is what only the database can: namespaces really isolate users in one shared table,
deletion really removes rows, retention really expires them, and deleting an account takes
its memories with it (FR-25, NFR-13).
"""

from __future__ import annotations

import hashlib

import pytest

import mmrag.db as db
from mmrag import memory
from mmrag.auth import service
from mmrag.config import load_settings
from mmrag.memory.episodes import summarize_idle
from mmrag.memory.extract import Episode
from mmrag.obs.ops import cleanup

pytestmark = pytest.mark.integration


def _embed(texts: list[str]) -> list[list[float]]:
    out = []
    for text in texts:
        vector = [0.0] * 1536
        for word in text.lower().split():
            vector[int(hashlib.sha256(word.encode()).hexdigest(), 16) % 1536] += 1.0
        out.append(vector)
    return out


class FakeLLM:
    model = "fake-memory-model"

    def __init__(self, summary="The user asked about attention and got an answer."):
        self.summary = summary

    def chat(self, messages, tools):  # pragma: no cover
        raise AssertionError("memory must not use the tool-calling model")

    def structured(self, messages, schema, tools=None):
        return Episode(summary=self.summary), object()


@pytest.fixture
def settings(fresh_db_url, base_config, write_config):
    base_config["observability"].update(enabled=False, log_file=None)
    s = load_settings(write_config(base_config), {"DATABASE_URL": fresh_db_url})
    db.migrate(s)
    return s


@pytest.fixture
def store(settings):
    with memory.open_store(settings, embed=_embed) as opened:
        yield opened


def _user(settings, email="a@example.com") -> str:
    service.add_user(settings, email)
    with db.connect(settings) as conn:
        return str(conn.execute("SELECT user_id FROM users WHERE email = %s", (email,)).fetchone()[0])


def _log_question(settings, user_id, thread, question="How does attention work?", age_minutes=0):
    with db.connect(settings) as conn:
        conn.execute(
            "INSERT INTO query_log (user_id, thread_id, trace_id, question, model, rounds, tokens_in, tokens_out, "
            "latency_ms, validator_result, answer_json, created_at) VALUES (%s, %s, 'x', %s, 'm', 1, 1, 1, 1, 'ok', "
            '\'{"blocks": [{"type": "text", "content": {"markdown": "An answer."}}]}\', '
            "now() - make_interval(mins => %s))", (user_id, thread, question, age_minutes))
        conn.commit()


def _age_episode(settings, thread_id: str, minutes: int) -> None:
    """Backdate a stored summary, so a later turn counts as new since it was written."""
    with db.connect(settings) as conn:
        conn.execute("UPDATE store SET value = jsonb_set(value, '{updated_at}', "
                     "to_jsonb(to_char(now() - make_interval(mins => %s), 'YYYY-MM-DD\"T\"HH24:MI:SSOF:00'))) "
                     "WHERE key = %s", (minutes, thread_id))
        conn.commit()


# ---------------------------------------------------------------- the store

def test_the_migration_adds_the_per_user_switch_defaulting_to_on(settings):
    user_id = _user(settings)
    assert memory.is_enabled(settings, user_id)
    memory.set_enabled(settings, user_id, False)
    assert not memory.is_enabled(settings, user_id)


def test_memory_off_in_config_overrides_every_users_switch(settings, base_config, write_config, fresh_db_url):
    user_id = _user(settings)
    base_config["memory"]["enabled"] = False
    off = load_settings(write_config(base_config), {"DATABASE_URL": fresh_db_url})
    assert not memory.is_enabled(off, user_id)
    with memory.open_store(off) as store:
        assert store is None  # nothing is opened, so nothing can be stored or recalled


def test_statements_survive_a_round_trip_through_postgres(settings, store):
    user_id = _user(settings)
    memory.remember_statements(store, user_id, [("output_format", "Prefers charts to tables.")])
    assert [m.text for m in memory.recall(store, user_id, "charts", 5)] == ["Prefers charts to tables."]


def test_two_users_share_the_table_but_never_each_others_memories(settings, store):
    alice, bob = _user(settings, "alice@example.com"), _user(settings, "bob@example.com")
    memory.remember_statements(store, alice, [("topic_focus", "Studies RLHF.")])
    memory.remember_statements(store, bob, [("topic_focus", "Studies tokenizers.")])
    with db.connect(settings) as conn:
        assert conn.execute("SELECT count(*) FROM store").fetchone()[0] == 2  # one shared table
    assert [m.text for m in memory.recall(store, alice, "what do I study?", 5)] == ["Studies RLHF."]
    assert [m.text for m in memory.list_memories(store, bob)] == ["Studies tokenizers."]


def test_a_deleted_memory_is_no_longer_recalled(settings, store):
    user_id = _user(settings)
    memory.remember_statements(store, user_id, [("output_format", "Prefers charts.")])
    memory.delete(store, user_id, memory.SEMANTIC, "output_format")
    assert memory.recall(store, user_id, "charts", 5) == []


# ---------------------------------------------------------------- retention and deletion

def test_retention_cleanup_removes_only_expired_memories(settings, store):
    user_id = _user(settings)
    memory.remember_statements(store, user_id, [("old", "An old note."), ("new", "A fresh note.")])
    with db.connect(settings) as conn:
        conn.execute("UPDATE store SET updated_at = now() - interval '400 days' WHERE key = 'old'")
        conn.commit()
    assert cleanup(settings)["memories"] == 1
    assert [m.key for m in memory.list_memories(store, user_id)] == ["new"]


def test_deleting_an_account_deletes_its_memories_and_leaves_the_others(settings, store):
    alice, bob = _user(settings, "alice@example.com"), _user(settings, "bob@example.com")
    memory.remember_statements(store, alice, [("a", "Alice note.")])
    memory.remember_statements(store, bob, [("b", "Bob note.")])
    _log_question(settings, alice, "t1")

    assert service.delete_user(settings, "alice@example.com") == {"memories": 1}
    assert memory.list_memories(store, alice) == []
    assert [m.text for m in memory.list_memories(store, bob)] == ["Bob note."]
    with db.connect(settings) as conn:
        assert conn.execute("SELECT count(*) FROM users").fetchone()[0] == 1
        # the question stays for the cost history, with no one attached to it
        assert conn.execute("SELECT user_id FROM query_log").fetchone()[0] is None
        assert conn.execute("SELECT count(*) FROM auth_events WHERE event_type = 'user_deleted'").fetchone()[0] == 1


# ---------------------------------------------------------------- episodes

def test_a_finished_conversation_is_summarised_once(settings, store):
    user_id = _user(settings)
    _log_question(settings, user_id, "thread-1", age_minutes=120)
    _log_question(settings, user_id, "thread-live", age_minutes=0)

    written = summarize_idle(settings, store, FakeLLM(), idle_minutes=60)
    assert [m.key for m in written] == ["thread-1"]  # the live conversation is left alone
    assert summarize_idle(settings, store, FakeLLM(), idle_minutes=60) == []  # the sweep can be re-run


def test_a_conversation_taken_up_again_is_summarised_again(settings, store):
    user_id = _user(settings)
    _log_question(settings, user_id, "thread-1", age_minutes=120)
    summarize_idle(settings, store, FakeLLM("First summary."), idle_minutes=60)
    _age_episode(settings, "thread-1", minutes=300)  # the summary was written yesterday...
    _log_question(settings, user_id, "thread-1", question="And the KV cache?", age_minutes=90)  # ...this came after

    written = summarize_idle(settings, store, FakeLLM("Second summary."), idle_minutes=60)
    assert [m.text for m in written] == ["Second summary."]
    assert len(memory.list_memories(store, user_id, memory.EPISODIC)) == 1


def test_a_user_with_memory_switched_off_is_never_summarised(settings, store):
    user_id = _user(settings)
    memory.set_enabled(settings, user_id, False)
    _log_question(settings, user_id, "thread-1", age_minutes=120)
    assert summarize_idle(settings, store, FakeLLM(), idle_minutes=60) == []
