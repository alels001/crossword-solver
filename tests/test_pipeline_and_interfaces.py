"""End-to-end orchestration, HTTP API, MCP server and the AI review tool."""

from __future__ import annotations

import asyncio
import importlib
import json
import sys

import pytest

from conftest import simulated_candidate_model
from crossword_solver.config import REPO_ROOT, Settings
from crossword_solver.llm import FakeLLM, LLMResponse, TokenUsage
from crossword_solver.models import PuzzleLayout
from crossword_solver.pipeline import CrosswordSolver


def _accuracy(report, gold: dict[str, str]) -> float:
    answers = {e.entry_id: e.answer for e in report.solution.entries}
    return sum(answers[k] == v for k, v in gold.items()) / len(gold)


def test_pipeline_with_simulated_model(
    settings: Settings, layout: PuzzleLayout, gold: dict[str, str]
) -> None:
    by_clue = {e.clue: gold[e.id] for e in layout.entries}
    llm = FakeLLM(handler=simulated_candidate_model(by_clue, hit_rate=0.85), model="sim-model")
    solver = CrosswordSolver(settings, llm=llm)
    report = solver.solve_layout(layout, use_agent=False)
    assert _accuracy(report, gold) >= 0.85  # crossings recover entries the "model" missed
    assert report.solution.usage.llm_calls >= len(layout.entries)
    assert {"candidates", "csp"} <= set(report.timings_ms)
    assert json.dumps(report.to_dict())  # fully serialisable for the API


def test_refinement_uses_crossing_letters(
    settings: Settings, layout: PuzzleLayout, gold: dict[str, str]
) -> None:
    """The 'model' only knows E01 once some letters are given — exercising the refine loop."""
    by_clue = {e.clue: gold[e.id] for e in layout.entries if e.id != "E01"}
    base = simulated_candidate_model(by_clue)

    def handler(messages, system, tools):
        user = messages[-1].text
        if "<clue>Laube</clue>" in user and "Pattern: __________" not in user:
            return LLMResponse(text=json.dumps({"candidates": [{"word": "GARTENHAUS", "confidence": 0.7}]}))
        return base(messages, system, tools)

    report = CrosswordSolver(settings, llm=FakeLLM(handler=handler, model="sim")).solve_layout(
        layout, use_agent=False
    )
    assert report.solution.entries[0].answer == "GARTENHAUS"


def test_budget_exhaustion_degrades_gracefully(tmp_path, layout: PuzzleLayout) -> None:
    s = Settings(llm_provider="fake", cache_dir=tmp_path, max_cost_usd_per_puzzle=0.0001, _env_file=None)
    pricey = FakeLLM(
        handler=lambda m, sy, t: LLMResponse(
            text='{"candidates": []}', usage=TokenUsage(100_000, 0), model="gpt-4.1"
        ),
        model="gpt-4.1",
    )
    report = CrosswordSolver(s, llm=pricey).solve_layout(layout)
    assert report.solution.stats.get("budget_exhausted") == 1
    assert report.solution.usage.cost_usd > 0


def test_offline_chat_falls_back_to_retrieval(settings: Settings) -> None:
    reply = CrosswordSolver(settings).chat("Vulkan bei Neapel?")
    assert "VESUV" in reply and "Offline" in reply


def test_suggest_validates_pattern(settings: Settings) -> None:
    solver = CrosswordSolver(settings)
    assert solver.suggest("Vulkan bei Neapel", 5)[0].word == "VESUV"
    with pytest.raises(ValueError):
        solver.suggest("x", 5, "AB")


# ---- HTTP API ----------------------------------------------------------------------
@pytest.fixture
def client(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient

    monkeypatch.setenv("CWS_LLM_PROVIDER", "fake")
    monkeypatch.setenv("CWS_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("CWS_API_KEY", "secret")
    from crossword_solver import config

    config.get_settings.cache_clear()
    sys.modules.pop("crossword_solver.serving.api", None)
    api = importlib.import_module("crossword_solver.serving.api")
    yield TestClient(api.app)
    config.get_settings.cache_clear()


def test_api_health_auth_and_candidates(client) -> None:
    assert client.get("/healthz").json()["status"] == "ok"
    assert client.post("/v1/candidates", json={"clue": "Papageienart", "length": 3}).status_code == 401
    h = {"X-API-Key": "secret"}
    r = client.post("/v1/candidates", json={"clue": "Papageienart", "length": 3}, headers=h)
    assert r.status_code == 200 and r.json()["candidates"][0]["word"] == "ARA"
    assert r.headers["x-request-id"]
    assert (
        client.post("/v1/candidates", json={"clue": "x", "length": 3, "pattern": "A"}, headers=h).status_code
        == 422
    )
    assert "cws_http_requests_total" in client.get("/metrics").text


def test_api_solve_layout_and_upload_validation(client, puzzle: dict) -> None:
    h = {"X-API-Key": "secret"}
    r = client.post(
        "/v1/solve/layout",
        json={"layout": puzzle["layout"], "known_answers": {"E01": "Gartenhaus"}},
        headers=h,
    )
    assert r.status_code == 200
    body = r.json()
    assert body["solution"]["entries"][0]["answer"] == "GARTENHAUS"
    assert len(body["solution"]["grid"]) == 13
    bad = client.post("/v1/jobs", files={"file": ("x.txt", b"hello", "text/plain")}, headers=h)
    assert bad.status_code == 415
    assert client.get("/v1/jobs/unknown", headers=h).status_code == 404


def test_api_chat(client) -> None:
    r = client.post("/v1/chat", json={"message": "Hauptstadt Norwegens"}, headers={"X-API-Key": "secret"})
    assert "OSLO" in r.json()["reply"]


# ---- MCP server -----------------------------------------------------------------------
def test_mcp_tools(monkeypatch, tmp_path, puzzle: dict) -> None:
    monkeypatch.setenv("CWS_LLM_PROVIDER", "fake")
    monkeypatch.setenv("CWS_CACHE_DIR", str(tmp_path))
    from crossword_solver import config
    from crossword_solver.serving import mcp_server

    config.get_settings.cache_clear()
    mcp_server._solver.cache_clear()

    async def go():
        names = {t.name for t in await mcp_server.mcp.list_tools()}
        assert {"suggest_answers", "search_clue_database", "solve_crossword", "list_prompts"} <= names
        r = await mcp_server.mcp.call_tool("search_clue_database", {"query": "Vulkan bei Neapel", "k": 2})
        assert "VESUV" in r.content[0].text
        layout = json.dumps(puzzle["layout"], ensure_ascii=False)
        r = await mcp_server.mcp.call_tool("solve_crossword", {"layout_json": layout, "use_agent": False})
        assert '"grid"' in r.content[0].text

    asyncio.run(go())
    config.get_settings.cache_clear()


# ---- AI review tool ----------------------------------------------------------------------
def test_ai_review_filters_and_renders() -> None:
    sys.path.insert(0, str(REPO_ROOT / "tools" / "ai_review"))
    review = importlib.import_module("review")
    diff = (
        "diff --git a/src/x.py b/src/x.py\n+print('hi')\n"
        "diff --git a/data/kb.csv b/data/kb.csv\n+a,b\n"
        "diff --git a/uv.lock b/uv.lock\n+x\n"
    )
    body, skipped = review.filter_diff(diff)
    assert "src/x.py" in body and set(skipped) == {"data/kb.csv", "uv.lock"}
    r = review.Review(
        summary="Looks fine.",
        verdict="comment",
        findings=[
            review.Finding(
                file="src/x.py", line=1, severity="medium", title="Debug print", detail="Remove it."
            )
        ],
    )
    md = review.render(r, skipped, {"model": "m", "input_tokens": 1, "output_tokens": 2, "cost_usd": 0.0})
    assert "Debug print" in md and "`src/x.py:1`" in md and "human approves" in md
