"""query_log: one row per question (spec §5.5, LLD §5.9, §8).

Feeds cost tracking, the Phase 6 evaluation and, from Phase 7, per-user limits and history.
`user_id` stays empty until sign-in exists (Phase 7).
"""

from __future__ import annotations

from mmrag.agent.graph import QueryRun
from mmrag.config import Settings


def cost_usd(model: str, input_tokens: int, output_tokens: int, settings: Settings) -> float | None:
    """Price from `pricing` in config.yaml (US$ per 1M tokens); None if the model is not priced."""
    raise NotImplementedError


def log_query(question: str, run: QueryRun, settings: Settings, user_id: str | None = None) -> str:
    """Insert the row and return its query_id."""
    raise NotImplementedError
