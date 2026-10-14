"""Anthropic Messages API adapter (text, vision, tool use)."""

from __future__ import annotations

from typing import Any

from .base import LLMError, LLMResponse, Message, TokenUsage, ToolCall, ToolSpec


class AnthropicClient:
    def __init__(
        self,
        model: str,
        *,
        max_tokens: int = 800,
        timeout: float = 60.0,
        max_retries: int = 3,
        client: Any = None,
    ) -> None:
        if client is None:
            try:
                import anthropic
            except ImportError as e:  # pragma: no cover - depends on extras
                raise LLMError(
                    "Install the Anthropic extra: pip install 'crossword-solver[anthropic]'"
                ) from e
            client = anthropic.Anthropic(timeout=timeout, max_retries=max_retries)
        self._client = client
        self.model = model
        self.max_tokens = max_tokens

    @staticmethod
    def _convert(messages: list[Message]) -> list[dict[str, Any]]:
        """Anthropic wants tool results as user-turn content blocks, grouped per turn."""
        out: list[dict[str, Any]] = []
        for m in messages:
            if m.role == "tool":
                block = {"type": "tool_result", "tool_use_id": m.tool_call_id, "content": m.text}
                if (
                    out
                    and out[-1]["role"] == "user"
                    and isinstance(out[-1]["content"], list)
                    and all(b.get("type") == "tool_result" for b in out[-1]["content"])
                ):
                    out[-1]["content"].append(block)
                else:
                    out.append({"role": "user", "content": [block]})
            elif m.role == "assistant":
                blocks: list[dict[str, Any]] = []
                if m.text:
                    blocks.append({"type": "text", "text": m.text})
                blocks += [
                    {"type": "tool_use", "id": tc.id, "name": tc.name, "input": tc.arguments}
                    for tc in m.tool_calls
                ]
                out.append({"role": "assistant", "content": blocks or [{"type": "text", "text": ""}]})
            else:
                content: list[dict[str, Any]] = [
                    {
                        "type": "image",
                        "source": {"type": "base64", "media_type": img.media_type, "data": img.b64()},
                    }
                    for img in m.images
                ]
                content.append({"type": "text", "text": m.text})
                out.append({"role": "user", "content": content})
        return out

    def chat(
        self,
        messages: list[Message],
        *,
        system: str | None = None,
        tools: list[ToolSpec] | None = None,
        json_mode: bool = False,
        temperature: float | None = None,  # accepted for interface parity; not sent
        max_tokens: int | None = None,
    ) -> LLMResponse:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens or self.max_tokens,
            "messages": self._convert(messages),
        }
        sys_prompt = system or ""
        if json_mode:
            sys_prompt = (
                sys_prompt + "\n\nRespond with a single valid JSON object and nothing else."
            ).strip()
        if sys_prompt:
            kwargs["system"] = sys_prompt
        if tools:
            kwargs["tools"] = [
                {"name": t.name, "description": t.description, "input_schema": t.parameters} for t in tools
            ]

        resp = self._client.messages.create(**kwargs)
        texts: list[str] = []
        calls: list[ToolCall] = []
        for block in resp.content:
            btype = getattr(block, "type", None)
            if btype == "text":
                texts.append(block.text)
            elif btype == "tool_use":
                calls.append(ToolCall(id=block.id, name=block.name, arguments=dict(block.input or {})))
        usage = getattr(resp, "usage", None)
        return LLMResponse(
            text="".join(texts).strip(),
            tool_calls=calls,
            usage=TokenUsage(
                input_tokens=getattr(usage, "input_tokens", 0) or 0,
                output_tokens=getattr(usage, "output_tokens", 0) or 0,
            ),
            model=getattr(resp, "model", self.model),
            stop_reason=getattr(resp, "stop_reason", None),
        )
