"""Chat history per user and the admin metrics (plan Phase 7 tasks 1, 9, 10). Test-first."""

from __future__ import annotations

import pytest

import mmrag.db as db
from mmrag.auth import service
from mmrag.config import load_settings
from mmrag.ui.data import admin_metrics, thread_turns, user_threads

pytestmark = pytest.mark.integration


@pytest.fixture
def settings(fresh_db_url, base_config, write_config):
    base_config["observability"].update(enabled=False, log_file=None)
    s = load_settings(write_config(base_config), {"DATABASE_URL": fresh_db_url})
    db.migrate(s)
    return s


def _log(conn, uid, thread, q, cost=0.01, latency=5000, result="ok"):
    conn.execute("INSERT INTO query_log (user_id, thread_id, trace_id, question, model, rounds, tokens_in, tokens_out, "
                 "cost_usd, latency_ms, validator_result, answer_json) VALUES (%s,%s,%s,%s,'m',1,10,2,%s,%s,%s,%s)",
                 (uid, thread, "f" * 32, q, cost, latency, result, '{"blocks": [], "sources": []}'))


def test_each_user_sees_only_their_own_conversations(settings):
    service.add_user(settings, "a@example.com")
    service.add_user(settings, "b@example.com")
    with db.connect(settings) as conn:
        a, b = [r[0] for r in conn.execute("SELECT user_id FROM users ORDER BY email")]
        _log(conn, a, "t1", "first question")
        _log(conn, a, "t1", "follow-up")
        _log(conn, b, "t2", "b's secret question")
        conn.commit()
    threads = user_threads(settings, str(a))
    assert [(t.thread_id, t.title, t.turns) for t in threads] == [("t1", "first question", 2)]
    assert [t.question for t in thread_turns(settings, str(a), "t1")] == ["first question", "follow-up"]
    assert thread_turns(settings, str(a), "t2") == []  # someone else's thread


def test_admin_metrics(settings):
    service.add_user(settings, "a@example.com")
    service.login(settings, "a@example.com", "wrong password here", "10.1.1.1")
    with db.connect(settings) as conn:
        a = conn.execute("SELECT user_id FROM users").fetchone()[0]
        _log(conn, a, "t1", "fast", cost=0.02, latency=3000)
        _log(conn, a, "t1", "slow", cost=0.03, latency=40000, result="dropped_blocks")
        conn.commit()
    m = admin_metrics(settings)
    assert m["questions_per_day"][0][1] == 2
    assert m["slowest"][0][0] == "slow" and m["slowest"][0][2] == "f" * 32
    assert m["cost_per_user"][0][0] == "a@example.com" and m["cost_per_user"][0][2] == pytest.approx(0.05)
    assert m["failed_sign_ins_per_day"][0][1] == 1
    assert m["answers_with_removed_parts"] == pytest.approx(0.5)
