"""Retention cleanup and alerts (plan Phase 8 task 6). Test-first."""

from __future__ import annotations

import pytest

import mmrag.db as db
from mmrag.auth import service
from mmrag.config import load_settings
from mmrag.obs.ops import check_alerts, cleanup

pytestmark = pytest.mark.integration


@pytest.fixture
def settings(fresh_db_url, base_config, write_config):
    base_config["observability"].update(enabled=False, log_file=None)
    base_config["alerts"] = {"daily_cost_usd": 1.0, "dropped_rate": 0.3, "min_questions": 3,
                             "failed_sign_ins_per_hour": 3}
    s = load_settings(write_config(base_config), {"DATABASE_URL": fresh_db_url})
    db.migrate(s)
    return s


def _q(conn, age_days=0, cost=0.01, result="ok", thread="t"):
    conn.execute("INSERT INTO query_log (thread_id, trace_id, question, model, rounds, tokens_in, tokens_out, cost_usd, "
                 "latency_ms, validator_result, answer_json, created_at) VALUES (%s,'x','q','m',1,1,1,%s,1,%s,'{}', "
                 "now() - make_interval(days => %s))", (thread, cost, result, age_days))


def test_cleanup_applies_the_retention_periods(settings):
    service.add_user(settings, "a@example.com")
    with db.connect(settings) as conn:
        _q(conn, age_days=200, thread="old")
        _q(conn, age_days=1, thread="new")
        conn.execute("INSERT INTO auth_events (event_type, created_at) VALUES ('login_failure', now() - interval '400 days')")
        uid = conn.execute("SELECT user_id FROM users").fetchone()[0]
        conn.execute("INSERT INTO sessions (user_id, token_hash, expires_at, created_at) "
                     "VALUES (%s, 'h', now() - interval '40 days', now() - interval '47 days')", (uid,))
        conn.commit()
    counts = cleanup(settings)
    assert counts["query_log"] == 1 and counts["auth_events"] == 1 and counts["sessions"] == 1
    with db.connect(settings) as conn:
        assert conn.execute("SELECT thread_id FROM query_log").fetchall() == [("new",)]
        assert conn.execute("SELECT count(*) FROM auth_events WHERE event_type = 'user_created'").fetchone()[0] == 1


def test_alerts_for_cost_dropped_answers_and_failed_sign_ins(settings):
    assert check_alerts(settings) == []
    with db.connect(settings) as conn:
        for _ in range(3):
            _q(conn, cost=0.5, result="dropped_blocks")
        conn.commit()
    for _ in range(4):
        service.login(settings, "nobody@example.com", "wrong password here", "10.2.2.2")
    alerts = " | ".join(check_alerts(settings))
    assert "cost today" in alerts and "removed parts" in alerts and "failed sign-ins" in alerts
