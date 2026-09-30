"""Chat history (each user sees only their own) and the admin metrics page (FR-22)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from mmrag import db
from mmrag.config import Settings


@dataclass
class Thread:
    thread_id: str
    title: str  # the first question
    turns: int
    last_at: datetime


@dataclass
class Turn:
    question: str
    answer: dict  # hydrated answer (query_log.answer_json)
    created_at: datetime
    cost_usd: float | None
    latency_ms: int
    trace_id: str


def user_threads(settings: Settings, user_id: str, limit: int = 50) -> list[Thread]:
    with db.connect(settings) as conn:
        rows = conn.execute(
            "SELECT thread_id, (array_agg(question ORDER BY created_at))[1], count(*), max(created_at) "
            "FROM query_log WHERE user_id = %s GROUP BY thread_id ORDER BY max(created_at) DESC LIMIT %s",
            (user_id, limit)).fetchall()
    return [Thread(*r) for r in rows]


def thread_turns(settings: Settings, user_id: str, thread_id: str) -> list[Turn]:
    """The turns of one thread, only if it belongs to this user."""
    with db.connect(settings) as conn:
        rows = conn.execute(
            "SELECT question, answer_json, created_at, cost_usd, latency_ms, trace_id FROM query_log "
            "WHERE user_id = %s AND thread_id = %s ORDER BY created_at", (user_id, thread_id)).fetchall()
    return [Turn(q, a, t, float(c) if c is not None else None, lat, tr) for q, a, t, c, lat, tr in rows]


def admin_metrics(settings: Settings, days: int = 14) -> dict:
    with db.connect(settings) as conn:
        since = "now() - make_interval(days => %s)"
        per_day = conn.execute(f"SELECT created_at::date, count(*) FROM query_log WHERE created_at > {since} "
                               "GROUP BY 1 ORDER BY 1 DESC", (days,)).fetchall()
        removed = conn.execute(f"SELECT avg((validator_result = 'dropped_blocks')::int) FROM query_log "
                               f"WHERE created_at > {since}", (days,)).fetchone()[0]
        slowest = conn.execute(f"SELECT question, latency_ms, trace_id, created_at FROM query_log "
                               f"WHERE created_at > {since} ORDER BY latency_ms DESC LIMIT 10", (days,)).fetchall()
        per_user = conn.execute(f"SELECT coalesce(u.email, '(no user)'), count(*), coalesce(sum(q.cost_usd), 0) "
                                f"FROM query_log q LEFT JOIN users u USING (user_id) WHERE q.created_at > {since} "
                                "GROUP BY 1 ORDER BY 3 DESC", (days,)).fetchall()
        failed = conn.execute(f"SELECT created_at::date, count(*) FROM auth_events WHERE event_type = 'login_failure' "
                              f"AND created_at > {since} GROUP BY 1 ORDER BY 1 DESC", (days,)).fetchall()
    return {"questions_per_day": per_day,
            "answers_with_removed_parts": float(removed) if removed is not None else None,
            "slowest": slowest,
            "cost_per_user": [(e, n, float(c)) for e, n, c in per_user],
            "failed_sign_ins_per_day": failed}
