"""LLM access layer: provider adapters, middleware and a settings-driven factory."""

from __future__ import annotations

from ..config import Settings, get_settings
from .base import (
    BudgetExceededError,
    ImagePart,
    LLMClient,
    LLMError,
    LLMResponse,
    Message,
    TokenUsage,
    ToolCall,
    ToolSpec,
)
from .fake import FakeLLM, ScriptedLLM
from .middleware import CachedLLM, TrackedLLM, UsageTracker


def build_llm(
    settings: Settings | None = None,
    *,
    tracker: UsageTracker | None = None,
    vision: bool = False,
    component: str = "llm",
) -> LLMClient:
    """Create the configured backend wrapped in cache + usage tracking."""
    s = settings or get_settings()
    client: LLMClient
    if s.llm_provider == "openai":
        from .openai_client import OpenAIClient

        client = OpenAIClient(
            s.openai_vision_model if vision else s.openai_model,
            temperature=s.temperature,
            max_tokens=s.max_output_tokens,
            timeout=s.request_timeout_s,
            max_retries=s.max_retries,
        )
    elif s.llm_provider == "anthropic":
        from .anthropic_client import AnthropicClient

        client = AnthropicClient(
            s.anthropic_vision_model if vision else s.anthropic_model,
            max_tokens=s.max_output_tokens,
            timeout=s.request_timeout_s,
            max_retries=s.max_retries,
        )
    else:
        client = FakeLLM()

    if s.enable_llm_cache and s.llm_provider != "fake":
        client = CachedLLM(client, s.cache_dir)
    if tracker is not None:
        client = TrackedLLM(client, tracker, component=component)
    return client


__all__ = [
    "BudgetExceededError",
    "CachedLLM",
    "FakeLLM",
    "ImagePart",
    "LLMClient",
    "LLMError",
    "LLMResponse",
    "Message",
    "ScriptedLLM",
    "TokenUsage",
    "ToolCall",
    "ToolSpec",
    "TrackedLLM",
    "UsageTracker",
    "build_llm",
]
