"""Provider-agnostic LLM interface.

The rest of the code base only talks to :class:`LLMClient`. Concrete adapters
(OpenAI, Anthropic, offline fake) translate this neutral format into each
provider's wire format, so switching vendors is a config change and tests never
need network access.
"""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

Role = Literal["user", "assistant", "tool"]


@dataclass(frozen=True)
class ImagePart:
    """A PNG/JPEG image passed to a vision model."""

    data: bytes
    media_type: str = "image/png"

    def b64(self) -> str:
        return base64.b64encode(self.data).decode("ascii")

    def data_url(self) -> str:
        return f"data:{self.media_type};base64,{self.b64()}"


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class Message:
    role: Role
    text: str = ""
    images: list[ImagePart] = field(default_factory=list)
    tool_calls: list[ToolCall] = field(default_factory=list)  # assistant -> tools
    tool_call_id: str | None = None  # role == "tool"

    @staticmethod
    def user(text: str, images: list[ImagePart] | None = None) -> Message:
        return Message(role="user", text=text, images=list(images or []))

    @staticmethod
    def tool_result(call_id: str, payload: Any) -> Message:
        text = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)
        return Message(role="tool", text=text, tool_call_id=call_id)


@dataclass(frozen=True)
class ToolSpec:
    """JSON-schema tool definition, shared by all providers."""

    name: str
    description: str
    parameters: dict[str, Any]


@dataclass
class TokenUsage:
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class LLMResponse:
    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: TokenUsage = field(default_factory=TokenUsage)
    model: str = ""
    stop_reason: str | None = None
    cached: bool = False

    def as_message(self) -> Message:
        return Message(role="assistant", text=self.text, tool_calls=list(self.tool_calls))


@runtime_checkable
class LLMClient(Protocol):
    """Minimal surface every backend implements."""

    model: str

    def chat(
        self,
        messages: list[Message],
        *,
        system: str | None = None,
        tools: list[ToolSpec] | None = None,
        json_mode: bool = False,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse: ...


class LLMError(RuntimeError):
    """Raised for non-recoverable LLM failures (bad config, budget exceeded, invalid output)."""


class BudgetExceededError(LLMError):
    pass
