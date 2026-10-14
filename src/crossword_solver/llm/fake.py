"""Deterministic offline LLM backends for tests, CI and key-less demos."""

from __future__ import annotations

import json
from collections import deque
from collections.abc import Callable, Iterable

from .base import LLMResponse, Message, TokenUsage, ToolSpec

Handler = Callable[[list[Message], str | None, list[ToolSpec] | None], LLMResponse | str]


class FakeLLM:
    """Offline backend.

    Without a handler it behaves like a cautious model that knows nothing:
    JSON requests get an empty candidate list and agent turns end immediately.
    That keeps the full pipeline runnable without an API key (knowledge base +
    CSP still work), and makes the value of a real model measurable in evals.
    """

    def __init__(self, handler: Handler | None = None, model: str = "fake-llm") -> None:
        self.model = model
        self._handler = handler
        self.calls: list[dict] = []

    def chat(
        self,
        messages: list[Message],
        *,
        system: str | None = None,
        tools: list[ToolSpec] | None = None,
        json_mode: bool = False,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        self.calls.append(
            {"messages": list(messages), "system": system, "tools": tools, "json_mode": json_mode}
        )
        if self._handler is not None:
            out = self._handler(messages, system, tools)
            resp = out if isinstance(out, LLMResponse) else LLMResponse(text=out)
        elif json_mode:
            resp = LLMResponse(text=json.dumps({"candidates": []}))
        else:
            resp = LLMResponse(text="")
        if not resp.model:
            resp.model = self.model
        if resp.usage.input_tokens == 0:
            prompt_chars = sum(len(m.text) for m in messages) + len(system or "")
            resp.usage = TokenUsage(input_tokens=prompt_chars // 4, output_tokens=len(resp.text) // 4)
        return resp


class ScriptedLLM(FakeLLM):
    """Returns pre-recorded responses in order — handy for agent-loop tests."""

    def __init__(self, responses: Iterable[LLMResponse | str], model: str = "scripted-llm") -> None:
        queue: deque[LLMResponse | str] = deque(responses)

        def handler(_m: list[Message], _s: str | None, _t: list[ToolSpec] | None) -> LLMResponse | str:
            return queue.popleft() if queue else LLMResponse(text="")

        super().__init__(handler=handler, model=model)
