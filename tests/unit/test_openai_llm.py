"""The OpenAI wrapper's request shape, with a fake client (no API calls). Test-first."""

from __future__ import annotations

from types import SimpleNamespace as NS

import pytest

from mmrag.agent.answer import Answer
from mmrag.agent.llm import OpenAILLM

pytestmark = pytest.mark.unit
TOOLS = [{"type": "function", "function": {"name": "search_text", "parameters": {"type": "object"}}}]


class FakeClient:
    def __init__(self, content):
        self.sent = []
        reply = NS(choices=[NS(message=NS(content=content, tool_calls=None, refusal=None))],
                   usage=NS(prompt_tokens=7, completion_tokens=3))
        self.chat = NS(completions=NS(create=lambda **kw: self.sent.append(kw) or reply))


def test_reasoning_models_use_effort_none_when_tools_are_sent():
    # Chat Completions rejects function tools with any other reasoning_effort on gpt-5.x.
    c = FakeClient("hi")
    llm = OpenAILLM(c, "gpt-5.4-mini", effort="medium")
    llm.chat([{"role": "user", "content": "q"}], TOOLS)
    llm.chat([{"role": "user", "content": "q"}], None)
    assert c.sent[0]["reasoning_effort"] == "none" and c.sent[1]["reasoning_effort"] == "medium"


def test_other_models_get_a_low_temperature():
    c = FakeClient("hi")
    OpenAILLM(c, "gpt-4o-mini").chat([{"role": "user", "content": "q"}], TOOLS)
    assert c.sent[0]["temperature"] == 0.1 and "reasoning_effort" not in c.sent[0]
    assert c.sent[0]["parallel_tool_calls"] is True


def test_structured_sends_a_strict_schema_and_parses_the_reply():
    c = FakeClient('{"blocks": [{"type": "text", "markdown": "x", "citations": [{"id": "a"}]}], '
                   '"not_found": false, "missing": null}')
    answer, reply = OpenAILLM(c, "gpt-4o-mini").structured([{"role": "user", "content": "q"}], Answer, TOOLS)
    fmt = c.sent[0]["response_format"]
    assert fmt["type"] == "json_schema" and fmt["json_schema"]["strict"] is True
    assert c.sent[0]["tool_choice"] == "none"
    assert answer.blocks[0].markdown == "x" and reply.input_tokens == 7
