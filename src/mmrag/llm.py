"""Thin OpenAI client wrapper (P6). Retries, usage accounting and cost come in Phase 4."""

from __future__ import annotations

import time
from dataclasses import dataclass

from openai import OpenAI

from mmrag.config import Settings


class MissingApiKey(RuntimeError):
    pass


def get_client(settings: Settings) -> OpenAI:
    key = settings.secrets.openai_api_key
    if key is None:
        raise MissingApiKey("OPENAI_API_KEY is not set in .env")
    return OpenAI(api_key=key.get_secret_value(), max_retries=2, timeout=60)


REASONING_PREFIXES = ("gpt-5", "gpt-6", "o1", "o3", "o4")


@dataclass
class Completion:
    text: str
    model: str
    input_tokens: int
    output_tokens: int
    seconds: float


def complete(client: OpenAI, model: str, prompt: str, max_tokens: int = 400) -> Completion:
    """One user prompt, one reply. Reasoning models get low effort and room for their hidden
    reasoning tokens, which count against the completion limit."""
    kwargs = {}
    if model.startswith(REASONING_PREFIXES):
        kwargs["reasoning_effort"] = "low"
        max_tokens += 4000
    started = time.monotonic()
    r = client.chat.completions.create(model=model, messages=[{"role": "user", "content": prompt}],
                                       max_completion_tokens=max_tokens, **kwargs)
    return Completion(text=(r.choices[0].message.content or "").strip(), model=r.model,
                      input_tokens=r.usage.prompt_tokens, output_tokens=r.usage.completion_tokens,
                      seconds=round(time.monotonic() - started, 2))
