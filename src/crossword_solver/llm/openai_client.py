"""OpenAI Chat Completions adapter (text, vision, JSON mode, tool calling)."""

from __future__ import annotations

import json
from typing import Any

from .base import LLMError, LLMResponse, Message, TokenUsage, ToolCall, ToolSpec


def _supports_temperature(model: str) -> bool:
    # Reasoning models reject sampling parameters.
    return not model.startswith(("o1", "o3", "o4", "gpt-5"))


class OpenAIClient:
    def __init__(
        self,
        model: str,
        *,
        temperature: float = 0.2,
        max_tokens: int = 800,
        timeout: float = 60.0,
        max_retries: int = 3,
        client: Any = None,
    ) -> None:
        if client is None:
            try:
                from openai import OpenAI
            except ImportError as e:  # pragma: no cover - depends on extras
                raise LLMError("Install the OpenAI extra: pip install 'crossword-solver[openai]'") from e
            client = OpenAI(timeout=timeout, max_retries=max_retries)
        self._client = client
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    # ---- format conversion ---------------------------------------------
    @staticmethod
    def _convert(messages: list[Message], system: str | None) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        if system:
            out.append({"role": "system", "content": system})
        for m in messages:
            if m.role == "tool":
                out.append({"role": "tool", "tool_call_id": m.tool_call_id, "content": m.text})
            elif m.role == "assistant":
                msg: dict[str, Any] = {"role": "assistant", "content": m.text or None}
                if m.tool_calls:
                    msg["tool_calls"] = [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
                        }
                        for tc in m.tool_calls
                    ]
                out.append(msg)
            else:
                if m.images:
                    parts: list[dict[str, Any]] = [{"type": "text", "text": m.text}]
                    parts += [
                        {"type": "image_url", "image_url": {"url": img.data_url(), "detail": "high"}}
                        for img in m.images
                    ]
                    out.append({"role": "user", "content": parts})
                else:
                    out.append({"role": "user", "content": m.text})
        return out

    @staticmethod
    def _tools(tools: list[ToolSpec]) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {"name": t.name, "description": t.description, "parameters": t.parameters},
            }
            for t in tools
        ]

    # ---- API ---------------------------------------------------------------
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
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": self._convert(messages, system),
            "max_completion_tokens": max_tokens or self.max_tokens,
        }
        if _supports_temperature(self.model):
            kwargs["temperature"] = self.temperature if temperature is None else temperature
        if tools:
            kwargs["tools"] = self._tools(tools)
        if json_mode and not tools:
            kwargs["response_format"] = {"type": "json_object"}

        resp = self._client.chat.completions.create(**kwargs)
        choice = resp.choices[0]
        calls: list[ToolCall] = []
        for tc in choice.message.tool_calls or []:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {"_raw": tc.function.arguments}
            calls.append(ToolCall(id=tc.id, name=tc.function.name, arguments=args))
        usage = getattr(resp, "usage", None)
        return LLMResponse(
            text=(choice.message.content or "").strip(),
            tool_calls=calls,
            usage=TokenUsage(
                input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
                output_tokens=getattr(usage, "completion_tokens", 0) or 0,
            ),
            model=getattr(resp, "model", self.model),
            stop_reason=choice.finish_reason,
        )
