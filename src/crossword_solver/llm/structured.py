"""Structured (JSON) output with schema validation and an automatic repair turn."""

from __future__ import annotations

import json
import logging
import re
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from .base import LLMClient, LLMError, Message

log = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)

_FENCE = re.compile(r"```(?:json)?|```", re.IGNORECASE)


def extract_json(text: str) -> object:
    """Parse the first JSON object/array in ``text`` (tolerates code fences and chatter)."""
    t = _FENCE.sub("", text).strip()
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        pass
    for open_ch, close_ch in (("{", "}"), ("[", "]")):
        start, end = t.find(open_ch), t.rfind(close_ch)
        if 0 <= start < end:
            try:
                return json.loads(t[start : end + 1])
            except json.JSONDecodeError:
                continue
    raise ValueError("no JSON found in model output")


def chat_structured(
    llm: LLMClient,
    schema: type[T],
    messages: list[Message],
    *,
    system: str | None = None,
    max_repairs: int = 1,
    coerce_list_key: str | None = None,
) -> T:
    """Ask for JSON, validate against ``schema``; on failure send one repair turn with the error.

    ``coerce_list_key`` lets legacy prompts that return a bare JSON array
    (``["A", "B"]``) be wrapped as ``{coerce_list_key: [...]}``.
    """
    history = list(messages)
    last_err: Exception | None = None
    for attempt in range(max_repairs + 1):
        resp = llm.chat(history, system=system, json_mode=True)
        try:
            data = extract_json(resp.text)
            if coerce_list_key and isinstance(data, list):
                data = {coerce_list_key: data}
            return schema.model_validate(data)
        except (ValueError, ValidationError) as e:
            last_err = e
            log.warning("structured_output_invalid", extra={"attempt": attempt, "error": str(e)[:300]})
            history = [
                *history,
                resp.as_message(),
                Message.user(
                    "Your previous reply was not valid for the required JSON schema.\n"
                    f"Error: {str(e)[:500]}\n"
                    f"Schema: {json.dumps(schema.model_json_schema())}\n"
                    "Reply again with ONLY the corrected JSON."
                ),
            ]
    raise LLMError(f"model did not return valid {schema.__name__}: {last_err}")
