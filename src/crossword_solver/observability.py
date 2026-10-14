"""Logging and lightweight metrics.

* JSON logs (``CWS_LOG_JSON=true``) carry every ``extra={...}`` field, so LLM calls,
  latencies, token counts and agent steps are queryable in any log stack.
* A request-id context variable ties all log lines of one API request together.
* :class:`Metrics` keeps Prometheus-style counters/histogram sums in memory and
  renders them in the text exposition format for ``GET /metrics``.
"""

from __future__ import annotations

import contextvars
import json
import logging
import threading
import time
from collections import defaultdict

request_id: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="-")

_STD_ATTRS = set(vars(logging.LogRecord("", 0, "", 0, "", None, None))) | {"message", "asctime"}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": round(record.created, 3),
            "level": record.levelname,
            "logger": record.name,
            "event": record.getMessage(),
            "request_id": request_id.get(),
        }
        for k, v in record.__dict__.items():
            if k not in _STD_ATTRS and not k.startswith("_"):
                payload[k] = v
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


class _ExtraFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        base = super().format(record)
        extras = {k: v for k, v in record.__dict__.items() if k not in _STD_ATTRS and not k.startswith("_")}
        return f"{base} {extras}" if extras else base


def setup_logging(level: str = "INFO", json_logs: bool = False) -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(
        JsonFormatter() if json_logs else _ExtraFormatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level.upper())
    for noisy in ("httpx", "httpcore", "openai", "anthropic", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


class Metrics:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.counters: dict[tuple[str, tuple[tuple[str, str], ...]], float] = defaultdict(float)
        self.started = time.time()

    def inc(self, name: str, value: float = 1.0, **labels: str) -> None:
        with self._lock:
            self.counters[(name, tuple(sorted(labels.items())))] += value

    def render(self) -> str:
        lines = [f"cws_uptime_seconds {time.time() - self.started:.1f}"]
        with self._lock:
            for (name, labels), v in sorted(self.counters.items()):
                lab = ",".join(f'{k}="{val}"' for k, val in labels)
                lines.append(f"{name}{{{lab}}} {v:g}" if lab else f"{name} {v:g}")
        return "\n".join(lines) + "\n"


METRICS = Metrics()
