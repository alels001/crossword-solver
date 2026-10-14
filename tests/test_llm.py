"""LLM layer: adapters' wire format, structured output repair, cache, budget."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from crossword_solver.llm import (
    BudgetExceededError,
    CachedLLM,
    FakeLLM,
    ImagePart,
    LLMError,
    LLMResponse,
    Message,
    ScriptedLLM,
    TokenUsage,
    ToolCall,
    ToolSpec,
    TrackedLLM,
    UsageTracker,
)
from crossword_solver.llm.anthropic_client import AnthropicClient
from crossword_solver.llm.openai_client import OpenAIClient
from crossword_solver.llm.structured import chat_structured, extract_json

TOOL = ToolSpec("get_board", "Show board", {"type": "object", "properties": {}})


class _Recorder:
    def __init__(self, response: object) -> None:
        self.kwargs: dict = {}
        self.response = response

    def create(self, **kwargs: object) -> object:
        self.kwargs = kwargs
        return self.response


def _conversation() -> list[Message]:
    return [
        Message.user("look", images=[ImagePart(b"\x89PNG")]),
        Message(
            role="assistant",
            text="",
            tool_calls=[ToolCall("c1", "get_board", {}), ToolCall("c2", "get_board", {})],
        ),
        Message.tool_result("c1", {"grid": "A"}),
        Message.tool_result("c2", "ok"),
    ]


def test_openai_adapter_wire_format() -> None:
    resp = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=None,
                    tool_calls=[
                        SimpleNamespace(
                            id="t9", function=SimpleNamespace(name="get_board", arguments='{"x": 1}')
                        )
                    ],
                ),
                finish_reason="tool_calls",
            )
        ],
        usage=SimpleNamespace(prompt_tokens=11, completion_tokens=3),
        model="gpt-4.1-mini",
    )
    rec = _Recorder(resp)
    client = OpenAIClient("gpt-4.1-mini", client=SimpleNamespace(chat=SimpleNamespace(completions=rec)))
    out = client.chat(_conversation(), system="sys", tools=[TOOL])

    msgs = rec.kwargs["messages"]
    assert msgs[0] == {"role": "system", "content": "sys"}
    assert msgs[1]["content"][1]["image_url"]["url"].startswith("data:image/png;base64,")
    assert msgs[2]["tool_calls"][0]["function"]["name"] == "get_board"
    assert msgs[3] == {"role": "tool", "tool_call_id": "c1", "content": '{"grid": "A"}'}
    assert rec.kwargs["tools"][0]["function"]["name"] == "get_board"
    assert out.tool_calls == [ToolCall("t9", "get_board", {"x": 1})]
    assert out.usage == TokenUsage(11, 3)


def test_openai_reasoning_models_get_no_temperature() -> None:
    resp = SimpleNamespace(
        choices=[
            SimpleNamespace(message=SimpleNamespace(content="{}", tool_calls=None), finish_reason="stop")
        ],
        usage=None,
    )
    rec = _Recorder(resp)
    OpenAIClient("o4-mini", client=SimpleNamespace(chat=SimpleNamespace(completions=rec))).chat(
        [Message.user("hi")], json_mode=True
    )
    assert "temperature" not in rec.kwargs
    assert rec.kwargs["response_format"] == {"type": "json_object"}


def test_anthropic_adapter_groups_tool_results() -> None:
    resp = SimpleNamespace(
        content=[
            SimpleNamespace(type="text", text="thinking"),
            SimpleNamespace(type="tool_use", id="u1", name="get_board", input={}),
        ],
        usage=SimpleNamespace(input_tokens=20, output_tokens=5),
        model="claude-sonnet-5-5",
        stop_reason="tool_use",
    )
    rec = _Recorder(resp)
    client = AnthropicClient("claude-sonnet-5-5", client=SimpleNamespace(messages=rec))
    out = client.chat(_conversation(), system="sys", tools=[TOOL])

    msgs = rec.kwargs["messages"]
    assert msgs[0]["content"][0]["type"] == "image"
    assert [b["type"] for b in msgs[1]["content"]] == ["tool_use", "tool_use"]
    assert msgs[2]["role"] == "user" and len(msgs[2]["content"]) == 2  # both results in ONE user turn
    assert rec.kwargs["tools"][0]["input_schema"] == TOOL.parameters
    assert "temperature" not in rec.kwargs
    assert out.text == "thinking" and out.tool_calls[0].name == "get_board"


class _Schema(BaseModel):
    candidates: list[str]


def test_extract_json_tolerates_fences_and_chatter() -> None:
    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json('Sure! Here: ["X", "Y"] hope that helps') == ["X", "Y"]
    with pytest.raises(ValueError):
        extract_json("no json here")


def test_structured_output_repairs_once() -> None:
    llm = ScriptedLLM(["not json at all", '{"candidates": ["AETNA"]}'])
    out = chat_structured(llm, _Schema, [Message.user("q")])
    assert out.candidates == ["AETNA"]
    assert len(llm.calls) == 2
    assert "not valid" in llm.calls[1]["messages"][-1].text


def test_structured_output_gives_up() -> None:
    with pytest.raises(LLMError):
        chat_structured(ScriptedLLM(["x", "y"]), _Schema, [Message.user("q")], max_repairs=1)


def test_structured_output_coerces_bare_list() -> None:
    out = chat_structured(
        ScriptedLLM(['["AETNA", "ETNA"]']), _Schema, [Message.user("q")], coerce_list_key="candidates"
    )
    assert out.candidates == ["AETNA", "ETNA"]


def test_cache_returns_identical_response_without_second_call(tmp_path: Path) -> None:
    inner = FakeLLM(handler=lambda m, s, t: LLMResponse(text="hello", usage=TokenUsage(10, 2)))
    cached = CachedLLM(inner, tmp_path)
    a = cached.chat([Message.user("hi")], system="s")
    b = cached.chat([Message.user("hi")], system="s")
    c = cached.chat([Message.user("different")], system="s")
    assert (a.text, b.text, c.text) == ("hello", "hello", "hello")
    assert not a.cached and b.cached and not c.cached
    assert len(inner.calls) == 2


def test_tracker_counts_cost_and_enforces_budget() -> None:
    tracker = UsageTracker(budget_usd=0.001, pricing={"m": (1.0, 2.0)})
    llm = TrackedLLM(
        FakeLLM(handler=lambda m, s, t: LLMResponse(text="x", usage=TokenUsage(600, 100), model="m")), tracker
    )
    llm.chat([Message.user("a")])
    assert tracker.cost_usd == pytest.approx(600 / 1e6 * 1.0 + 100 / 1e6 * 2.0)
    llm.chat([Message.user("b")])  # crosses the budget
    with pytest.raises(BudgetExceededError):
        llm.chat([Message.user("c")])
    assert tracker.snapshot()["llm_calls"] == 2


def test_fake_llm_defaults_are_safe() -> None:
    llm = FakeLLM()
    assert json.loads(llm.chat([Message.user("x")], json_mode=True).text) == {"candidates": []}
    assert llm.chat([Message.user("x")], tools=[TOOL]).tool_calls == []
