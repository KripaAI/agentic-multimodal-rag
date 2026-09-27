"""Real OpenAI calls. Run on request only: pytest -m live (costs a fraction of a cent)."""

from __future__ import annotations

import pytest

from mmrag.config import get_settings
from mmrag.llm import get_client

pytestmark = pytest.mark.live


def test_embeddings_return_configured_dims():
    s = get_settings()
    emb = get_client(s).embeddings.create(model=s.embed.model, input="live test", dimensions=s.embed.dims)
    assert len(emb.data[0].embedding) == s.embed.dims


def test_agent_model_answers():
    s = get_settings()
    assert s.agent.model, "agent.model must be set in config.yaml"
    r = get_client(s).chat.completions.create(
        model=s.agent.model,
        messages=[{"role": "user", "content": "Reply with the single word: ready"}],
        max_completion_tokens=20,
    )
    assert "ready" in r.choices[0].message.content.lower()
