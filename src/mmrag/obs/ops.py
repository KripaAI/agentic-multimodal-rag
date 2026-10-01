"""Operations: retention cleanup and alert checks (plan Phase 8 task 6, LLD §3.9 `obs cleanup`)."""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from mmrag import db, memory
from mmrag.config import Settings

SESSION_KEEP_DAYS = 30  # expired or revoked sessions are deleted this long after they ended
_CHECKPOINT_TABLES = ("checkpoint_writes", "checkpoint_blobs", "checkpoints")  # LangGraph's, if present


def cleanup(settings: Settings) -> dict[str, int]:
    """Delete rows past their retention period. Returns the number deleted per table."""
    r = settings.retention
    counts: dict[str, int] = {}
    with db.connect(settings) as conn:
        counts["query_log"] = conn.execute(
            "DELETE FROM query_log WHERE created_at < now() - make_interval(days => %s)", (r.query_log_days,)).rowcount
        counts["auth_events"] = conn.execute(
            "DELETE FROM auth_events WHERE created_at < now() - make_interval(days => %s)",
            (r.auth_events_days,)).rowcount
        counts["sessions"] = conn.execute(
            "DELETE FROM sessions WHERE least(expires_at, coalesce(revoked_at, expires_at)) "
            "< now() - make_interval(days => %s)", (SESSION_KEEP_DAYS,)).rowcount
        # conversation state of threads whose history has gone (they can no longer be reopened)
        for table in _CHECKPOINT_TABLES:
            if conn.execute("SELECT to_regclass(%s)", (table,)).fetchone()[0]:
                counts[table] = conn.execute(
                    f"DELETE FROM {table} WHERE thread_id NOT IN (SELECT DISTINCT thread_id FROM query_log)").rowcount
        conn.commit()
    counts["memories"] = memory.cleanup_expired(settings)  # Phase 9: memory.retention_days (NFR-13)
    return counts


def check_alerts(settings: Settings) -> list[str]:
    """Alert messages; an empty list means all is well."""
    a, alerts = settings.alerts, []
    tz = ZoneInfo(settings.auth.limits_timezone)
    start = datetime.now(tz).replace(hour=0, minute=0, second=0, microsecond=0)
    with db.connect(settings) as conn:
        cost = float(conn.execute("SELECT coalesce(sum(cost_usd), 0) FROM query_log WHERE created_at >= %s",
                                  (start,)).fetchone()[0])
        n, dropped = conn.execute("SELECT count(*), count(*) FILTER (WHERE validator_result = 'dropped_blocks') "
                                  "FROM query_log WHERE created_at > now() - interval '1 hour'").fetchone()
        failed = conn.execute("SELECT count(*) FROM auth_events WHERE event_type = 'login_failure' "
                              "AND created_at > now() - interval '1 hour'").fetchone()[0]
    if cost > a.daily_cost_usd:
        alerts.append(f"cost today ${cost:.2f} is over the ${a.daily_cost_usd:.2f} budget")
    if n >= a.min_questions and dropped / n > a.dropped_rate:
        alerts.append(f"{dropped} of {n} answers in the last hour had removed parts (unverified content)")
    if failed > a.failed_sign_ins_per_hour:
        alerts.append(f"{failed} failed sign-ins in the last hour (possible password guessing)")
    return alerts
