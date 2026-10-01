"""Episodic memory: one summary per finished conversation (Phase 9, LLD §5.11).

A conversation has no "end" event — the user simply stops. So a thread counts as finished
when no question has been added to it for `memory.idle_minutes`, and a sweep
(`mmrag memory summarize`, run on a schedule) summarises the ones that are finished and not
yet summarised. The thread id is the memory key, so re-running the sweep is harmless, and a
thread taken up again after its summary was written is summarised again, in place.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from mmrag import db
from mmrag.config import Settings
from mmrag.memory import EPISODIC, Memory, list_memories, remember_episode
from mmrag.obs import get_logger

_log = get_logger("mmrag.memory")


@dataclass
class IdleThread:
    user_id: str
    thread_id: str
    last_at: datetime
    turns: list[tuple[str, str]]  # (question, answer text)


def _answer_text(answer_json: dict | None) -> str:
    """The prose of a stored answer; images, charts and tables are not part of an episode."""
    blocks = (answer_json or {}).get("blocks") or []
    return "\n".join(b.get("content", {}).get("markdown", "") for b in blocks if b.get("type") == "text")


def _stale(episode_updated_at: str | None, last_at: datetime) -> bool:
    """The thread has new turns since its summary was written (or there is no summary yet)."""
    if not episode_updated_at:
        return True
    try:
        return datetime.fromisoformat(episode_updated_at) < last_at
    except ValueError:
        return True


def idle_threads(settings: Settings, idle_minutes: int, user_id: str | None = None) -> list[IdleThread]:
    """Threads of signed-in users with no new question for `idle_minutes`, newest first."""
    where = "user_id IS NOT NULL" + (" AND user_id = %s" if user_id else "")
    params = ((user_id,) if user_id else ()) + (idle_minutes,)
    with db.connect(settings) as conn:
        rows = conn.execute(
            f"SELECT user_id::text, thread_id, max(created_at) FROM query_log WHERE {where} GROUP BY 1, 2 "
            "HAVING max(created_at) < now() - make_interval(mins => %s) ORDER BY max(created_at) DESC",
            params).fetchall()
        out = []
        for uid, tid, last_at in rows:
            turns = conn.execute("SELECT question, answer_json FROM query_log WHERE thread_id = %s AND user_id = %s "
                                 "ORDER BY created_at", (tid, uid)).fetchall()
            out.append(IdleThread(uid, tid, last_at, [(q, _answer_text(a)) for q, a in turns]))
    return out


def summarize_idle(settings: Settings, store, llm, idle_minutes: int | None = None,
                   user_id: str | None = None) -> list[Memory]:
    """Summarise every finished thread whose summary is missing or out of date. Users with
    memory switched off are skipped. Returns the episodes written."""
    from mmrag.memory import is_enabled
    from mmrag.memory.extract import summarize_episode

    if store is None:
        return []
    minutes = settings.memory.idle_minutes if idle_minutes is None else idle_minutes
    written: list[Memory] = []
    allowed: dict[str, bool] = {}
    stored: dict[str, dict[str, Memory]] = {}
    for thread in idle_threads(settings, minutes, user_id):
        if allowed.setdefault(thread.user_id, is_enabled(settings, thread.user_id)) is False:
            continue
        if thread.user_id not in stored:
            stored[thread.user_id] = {m.key: m for m in list_memories(store, thread.user_id, EPISODIC)}
        known = stored[thread.user_id].get(thread.thread_id)
        if not _stale(known.updated_at if known else None, thread.last_at):
            continue
        episode = remember_episode(store, thread.user_id, thread.thread_id,
                                   summarize_episode(llm, thread.turns))
        if episode:
            written.append(episode)
            _log.info("episode stored for thread %s", thread.thread_id)
    return written
