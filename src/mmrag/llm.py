"""Thin OpenAI client wrapper (P6). Retries, usage accounting and cost come in Phase 4."""

from __future__ import annotations

from openai import OpenAI

from mmrag.config import Settings


class MissingApiKey(RuntimeError):
    pass


def get_client(settings: Settings) -> OpenAI:
    key = settings.secrets.openai_api_key
    if key is None:
        raise MissingApiKey("OPENAI_API_KEY is not set in .env")
    return OpenAI(api_key=key.get_secret_value(), max_retries=2, timeout=60)
