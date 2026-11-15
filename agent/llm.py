"""Thin LLM adapter (Anthropic Messages API with tool use). Imported lazily so tests/CI never need the SDK."""
from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass
class ToolUse:
    id: str
    name: str
    input: dict


@dataclass
class LLMStep:
    text: str = ""
    tool_calls: list[ToolUse] = field(default_factory=list)
    content: list[dict] = field(default_factory=list)  # assistant content blocks to replay in the history


class AnthropicLLM:
    def __init__(self, model: str | None = None, max_tokens: int = 1500):
        import anthropic  # lazy import

        self._client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"], timeout=60.0)
        self._model = model or os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-5-5")
        self._max_tokens = max_tokens

    def step(self, system: str, messages: list[dict], tools: list[dict]) -> LLMStep:
        resp = self._client.messages.create(model=self._model, max_tokens=self._max_tokens, system=system,
                                            messages=messages, tools=tools)
        step = LLMStep()
        for block in resp.content:
            if block.type == "text":
                step.text += block.text
                step.content.append({"type": "text", "text": block.text})
            elif block.type == "tool_use":
                step.tool_calls.append(ToolUse(block.id, block.name, dict(block.input)))
                step.content.append({"type": "tool_use", "id": block.id, "name": block.name, "input": dict(block.input)})
        return step
