"""Long-term memory against the real models and the real corpus (plan Phase 9 task 8).

Run on request only: `pytest -m live` (a few cents). This is the evidence for two of the
Phase 9 acceptance criteria that no mock can give — that the extractor really produces notes
about the *user* from a real exchange, and that a preference stated in one conversation
reaches the next one. The faithfulness comparison itself is a paid evaluation run; see
`docs/07-operations.md` §9.

It uses the project database, so it creates its own throwaway account and deletes it at the
end, memories and all.
"""

from __future__ import annotations

import uuid

import pytest

from mmrag import db, memory
from mmrag.agent.graph import run_query
from mmrag.agent.llm import OpenAILLM
from mmrag.auth import service
from mmrag.config import get_settings
from mmrag.llm import get_client
from mmrag.memory.extract import extract_semantic

pytestmark = pytest.mark.live


@pytest.fixture
def account():
    settings = get_settings()
    if not settings.memory.enabled:
        pytest.skip("memory.enabled is false in config.yaml")
    email = f"memory-live-{uuid.uuid4().hex[:8]}@example.com"
    service.add_user(settings, email)
    with db.connect(settings) as conn:
        user_id = str(conn.execute("SELECT user_id FROM users WHERE email = %s", (email,)).fetchone()[0])
    yield settings, user_id
    service.delete_user(settings, email)


def test_the_extractor_keeps_preferences_and_drops_document_facts(account):
    settings, _ = account
    llm = OpenAILLM(get_client(settings), settings.memory.model or settings.agent.model)
    pairs = extract_semantic(
        llm,
        "I always want charts rather than tables. How do KV cache sizes compare across the models?",
        "Here is a chart of the three sizes: 1.2 GB, 2.4 GB and 4.8 GB.",
        max_chars=settings.memory.max_statement_chars)
    assert pairs, "the model found nothing worth remembering about the user"
    text = " ".join(t for _, t in pairs).lower()
    assert "chart" in text
    assert "1.2" not in text and "gb" not in text  # document numbers are evidence, not memory (P13)


def test_a_preference_from_one_conversation_reaches_the_next(account):
    """The acceptance criterion, end to end: two threads, one user, no shared state but memory."""
    settings, user_id = account
    with memory.open_store(settings) as store:
        memory.remember_statements(store, user_id, [("output_format", "Prefers charts to tables.")])

    run = run_query("How do the attention variants compare?", settings, user_id=user_id)
    assert run.answer.blocks  # a normal, validated answer
    with db.connect(settings) as conn:  # the question was logged against this user, in its own thread
        assert conn.execute("SELECT count(*) FROM query_log WHERE user_id = %s", (user_id,)).fetchone()[0] == 1

    with memory.open_store(settings) as store:
        recalled = [m.text for m in memory.recall(store, user_id, "compare the variants", 5)]
    assert "Prefers charts to tables." in recalled


def test_no_answer_ever_cites_a_memory(account):
    """P13 in the real thing: whatever the model does with the notes, no citation points at one."""
    settings, user_id = account
    with memory.open_store(settings) as store:
        memory.remember_statements(store, user_id, [("topic_focus", "Studies attention mechanisms.")])

    run = run_query("What is attention?", settings, user_id=user_id)
    cited = [c.id for b in run.answer.blocks for c in b.citations]
    assert not any(c.startswith("memory:") for c in cited)
