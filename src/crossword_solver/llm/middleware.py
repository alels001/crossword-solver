"""Cross-cutting LLM concerns implemented as decorators around any :class:`LLMClient`.

* :class:`CachedLLM`  – SQLite response cache (identical prompt → no second bill).
* :class:`TrackedLLM` – token/cost accounting, per-solve budget guard, latency logging.

They compose: ``TrackedLLM(CachedLLM(OpenAIClient(...)), tracker)``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import threading
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .base import (
    BudgetExceededError,
    LLMClient,
    LLMResponse,
    Message,
    TokenUsage,
    ToolCall,
)

log = logging.getLogger(__name__)

# USD per 1M tokens (input, output). List prices at time of writing — override with
# UsageTracker(pricing=...) for your contract. Unknown models are tracked at 0 cost.
DEFAULT_PRICING: dict[str, tuple[float, float]] = {
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1-nano": (0.10, 0.40),
    "gpt-4.1": (2.00, 8.00),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "text-embedding-3-small": (0.02, 0.0),
    "claude-haiku": (1.00, 5.00),
    "claude-sonnet": (3.00, 15.00),
    "claude-opus": (5.00, 25.00),
}


def _price_for(model: str, pricing: dict[str, tuple[float, float]]) -> tuple[float, float]:
    # Longest matching prefix wins ("gpt-4.1-mini" before "gpt-4.1").
    best = ""
    for key in pricing:
        if model.startswith(key) and len(key) > len(best):
            best = key
    return pricing.get(best, (0.0, 0.0))


class UsageTracker:
    """Thread-safe accumulator for one unit of work (e.g. one puzzle solve)."""

    def __init__(
        self, budget_usd: float | None = None, pricing: dict[str, tuple[float, float]] | None = None
    ):
        self.budget_usd = budget_usd
        self.pricing = pricing or DEFAULT_PRICING
        self.llm_calls = 0
        self.cache_hits = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.cost_usd = 0.0
        self._lock = threading.Lock()

    def check_budget(self) -> None:
        if self.budget_usd is not None and self.cost_usd >= self.budget_usd:
            raise BudgetExceededError(
                f"LLM budget of ${self.budget_usd:.2f} exhausted (spent ${self.cost_usd:.4f})"
            )

    def record(self, model: str, usage: TokenUsage, cached: bool) -> None:
        with self._lock:
            self.llm_calls += 1
            if cached:
                self.cache_hits += 1
                return
            p_in, p_out = _price_for(model, self.pricing)
            self.input_tokens += usage.input_tokens
            self.output_tokens += usage.output_tokens
            self.cost_usd += usage.input_tokens / 1e6 * p_in + usage.output_tokens / 1e6 * p_out

    def snapshot(self) -> dict[str, float | int]:
        return {
            "llm_calls": self.llm_calls,
            "cache_hits": self.cache_hits,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cost_usd": round(self.cost_usd, 6),
        }


class TrackedLLM:
    def __init__(self, inner: LLMClient, tracker: UsageTracker, component: str = "llm") -> None:
        self.inner = inner
        self.tracker = tracker
        self.component = component
        self.model = inner.model

    def chat(self, messages: list[Message], **kwargs: Any) -> LLMResponse:
        self.tracker.check_budget()
        t0 = time.perf_counter()
        resp = self.inner.chat(messages, **kwargs)
        latency_ms = (time.perf_counter() - t0) * 1000
        self.tracker.record(resp.model or self.model, resp.usage, resp.cached)
        log.info(
            "llm_call",
            extra={
                "component": self.component,
                "model": resp.model or self.model,
                "latency_ms": round(latency_ms, 1),
                "input_tokens": resp.usage.input_tokens,
                "output_tokens": resp.usage.output_tokens,
                "cached": resp.cached,
                "tool_calls": len(resp.tool_calls),
            },
        )
        return resp


def _key(model: str, messages: list[Message], kwargs: dict) -> str:
    def enc(m: Message) -> dict:
        return {
            "role": m.role,
            "text": m.text,
            "images": [hashlib.sha256(i.data).hexdigest() for i in m.images],
            "tool_calls": [asdict(t) for t in m.tool_calls],
            "tool_call_id": m.tool_call_id,
        }

    tools = kwargs.get("tools") or []
    payload = {
        "model": model,
        "messages": [enc(m) for m in messages],
        "system": kwargs.get("system"),
        "tools": [asdict(t) for t in tools],
        "json_mode": kwargs.get("json_mode", False),
        "temperature": kwargs.get("temperature"),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


class CachedLLM:
    """Persistent response cache. Safe across threads (one connection per call)."""

    def __init__(self, inner: LLMClient, cache_dir: Path) -> None:
        self.inner = inner
        self.model = inner.model
        cache_dir.mkdir(parents=True, exist_ok=True)
        self._path = cache_dir / "llm_cache.sqlite3"
        with sqlite3.connect(self._path) as con:
            con.execute("CREATE TABLE IF NOT EXISTS cache (k TEXT PRIMARY KEY, v TEXT, created REAL)")

    def chat(self, messages: list[Message], **kwargs: Any) -> LLMResponse:
        k = _key(self.model, messages, kwargs)
        with sqlite3.connect(self._path) as con:
            row = con.execute("SELECT v FROM cache WHERE k = ?", (k,)).fetchone()
        if row:
            d = json.loads(row[0])
            return LLMResponse(
                text=d["text"],
                tool_calls=[ToolCall(**tc) for tc in d["tool_calls"]],
                usage=TokenUsage(**d["usage"]),
                model=d["model"],
                stop_reason=d.get("stop_reason"),
                cached=True,
            )
        resp = self.inner.chat(messages, **kwargs)
        d = {
            "text": resp.text,
            "tool_calls": [asdict(tc) for tc in resp.tool_calls],
            "usage": asdict(resp.usage),
            "model": resp.model,
            "stop_reason": resp.stop_reason,
        }
        with sqlite3.connect(self._path) as con:
            con.execute("INSERT OR REPLACE INTO cache VALUES (?, ?, ?)", (k, json.dumps(d), time.time()))
        return resp


__all__ = ["DEFAULT_PRICING", "CachedLLM", "TrackedLLM", "UsageTracker"]
