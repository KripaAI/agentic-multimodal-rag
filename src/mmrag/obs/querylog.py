"""query_log: one row per question (spec §5.5, LLD §5.9, §8).

Feeds cost tracking, the Phase 6 evaluation and, from Phase 7, per-user limits and history.
`user_id` stays empty until sign-in exists (Phase 7).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from psycopg.types.json import Jsonb

from mmrag import db
from mmrag.config import Settings

if TYPE_CHECKING:
    from mmrag.agent.graph import QueryRun


def cost_usd(model: str, input_tokens: int, output_tokens: int, settings: Settings) -> float | None:
    """Price from `pricing` in config.yaml (US$ per 1M tokens); None if the model is not priced."""
    price = settings.pricing.get(model)
    if price is None:
        return None
    return round((input_tokens * price.input_per_mtok + output_tokens * price.output_per_mtok) / 1_000_000, 6)


def log_chart_turn(settings: Settings, user_id: str | None, thread_id: str, question: str, answer: dict,
                   model: str, latency_ms: int, birth: dict | None, followup: bool) -> str:
    """A birth-chart reading or follow-up from the chat, stored like any other turn so the conversation can be
    reopened from the sidebar. `birth` (the details the chart was computed from) goes into answer_json, so a
    reopened conversation still answers follow-ups from the same chart."""
    import uuid

    stored = dict(answer, birth=birth) if birth else answer
    with db.connect(settings) as conn:
        row = conn.execute(
            "INSERT INTO query_log (user_id, thread_id, trace_id, question, qtype, model, rounds, tool_calls, "
            "tokens_in, tokens_out, cost_usd, latency_ms, validator_result, answer_json) "
            "VALUES (%s,%s,%s,%s,%s,%s,1,%s,0,0,NULL,%s,'ok',%s) RETURNING query_id",
            (user_id, thread_id, uuid.uuid4().hex, question, "chart_followup" if followup else "chart_reading",
             model, Jsonb([]), latency_ms, Jsonb(stored))).fetchone()
        conn.commit()
    return str(row[0])


def log_query(question: str, run: "QueryRun", settings: Settings, user_id: str | None = None) -> str:
    """Insert the row and return its query_id."""
    with db.connect(settings) as conn:
        row = conn.execute(
            "INSERT INTO query_log (user_id, thread_id, trace_id, question, qtype, model, rounds, tool_calls, "
            "tokens_in, tokens_out, cost_usd, latency_ms, validator_result, answer_json) "
            "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) RETURNING query_id",
            (user_id, run.thread_id, run.trace_id, question, run.qtype, run.model, run.rounds, Jsonb(run.tool_calls),
             run.input_tokens, run.output_tokens, run.cost_usd, run.latency_ms, run.validator_result,
             Jsonb(run.answer.model_dump(mode="json")))).fetchone()
        conn.commit()
    return str(row[0])
