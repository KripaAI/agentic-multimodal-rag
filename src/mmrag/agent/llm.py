"""The only place the agent talks to OpenAI (P6: models are swappable by config).

Two calls: `chat` (with tools; may return tool calls) and `structured` (a pydantic schema
enforced by OpenAI's structured outputs). Messages keep images as local paths
(`{"type": "image_path", "path": ...}`) so checkpoints stay small; they are turned into
data URLs only here, when sent.
"""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, TypeVar

from pydantic import BaseModel

from mmrag.llm import REASONING_PREFIXES

T = TypeVar("T", bound=BaseModel)


@dataclass
class LLMReply:
    content: str | None
    tool_calls: list[dict] = field(default_factory=list)  # [{"id", "name", "arguments": dict}]
    input_tokens: int = 0
    output_tokens: int = 0


class LLM(Protocol):
    model: str

    def chat(self, messages: list[dict], tools: list[dict] | None) -> LLMReply: ...
    def structured(self, messages: list[dict], schema: type[T], tools: list[dict] | None = None) -> tuple[T, LLMReply]: ...


def _wire(messages: list[dict]) -> list[dict]:
    """Replace image_path parts with base64 data URLs for the API."""
    out = []
    for m in messages:
        if isinstance(m.get("content"), list):
            parts = []
            for p in m["content"]:
                if p.get("type") == "image_path":
                    data = base64.b64encode(Path(p["path"]).read_bytes()).decode()
                    parts.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{data}"}})
                else:
                    parts.append(p)
            m = {**m, "content": parts}
        out.append(m)
    return out


class OpenAILLM:
    def __init__(self, client, model: str, effort: str | None = None, max_tokens: int = 4000):
        self.client, self.model, self.max_tokens = client, model, max_tokens
        self._extra = {"reasoning_effort": effort or "low"} if model.startswith(REASONING_PREFIXES) else {"temperature": 0.1}

    def chat(self, messages: list[dict], tools: list[dict] | None) -> LLMReply:
        kwargs = {"tools": tools, "parallel_tool_calls": True} if tools else {}
        r = self.client.chat.completions.create(model=self.model, messages=_wire(messages),
                                                max_completion_tokens=self.max_tokens, **kwargs, **self._extra)
        msg = r.choices[0].message
        calls = [{"id": c.id, "name": c.function.name, "arguments": json.loads(c.function.arguments or "{}")}
                 for c in (msg.tool_calls or [])]
        return LLMReply(content=msg.content, tool_calls=calls, input_tokens=r.usage.prompt_tokens,
                        output_tokens=r.usage.completion_tokens)

    def structured(self, messages: list[dict], schema: type[T], tools: list[dict] | None = None) -> tuple[T, LLMReply]:
        kwargs = {"tools": tools, "tool_choice": "none"} if tools else {}
        r = self.client.chat.completions.parse(model=self.model, messages=_wire(messages), response_format=schema,
                                               max_completion_tokens=self.max_tokens, **kwargs, **self._extra)
        msg = r.choices[0].message
        if msg.parsed is None:
            raise ValueError(f"the model returned no {schema.__name__}: {msg.refusal or msg.content!r}")
        return msg.parsed, LLMReply(content=msg.content, input_tokens=r.usage.prompt_tokens,
                                    output_tokens=r.usage.completion_tokens)
