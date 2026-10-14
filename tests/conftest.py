from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from crossword_solver.config import REPO_ROOT, Settings
from crossword_solver.knowledge import ClueStore, TfidfRetriever
from crossword_solver.llm import LLMResponse, Message, ToolSpec
from crossword_solver.models import PuzzleLayout

BENCH = REPO_ROOT / "data" / "benchmark" / "puzzle_01.json"


@pytest.fixture(scope="session")
def puzzle() -> dict:
    return json.loads(BENCH.read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def layout(puzzle: dict) -> PuzzleLayout:
    return PuzzleLayout.model_validate(puzzle["layout"])


@pytest.fixture(scope="session")
def gold(puzzle: dict) -> dict[str, str]:
    return puzzle["gold"]


@pytest.fixture(scope="session")
def store() -> ClueStore:
    return ClueStore.load(REPO_ROOT / "data" / "knowledge" / "clues_de.csv")


@pytest.fixture(scope="session")
def retriever(store: ClueStore) -> TfidfRetriever:
    return TfidfRetriever(store)


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(llm_provider="fake", cache_dir=tmp_path / "cache", enable_llm_cache=False, _env_file=None)


def simulated_candidate_model(gold_by_clue: dict[str, str], seed: int = 0, hit_rate: float = 1.0):
    """A stand-in for a real LLM in tests: answers the candidate prompt with the gold
    word (with probability ``hit_rate``) plus plausible-looking distractors."""
    rng = random.Random(seed)

    def handler(messages: list[Message], system: str | None, tools: list[ToolSpec] | None) -> LLMResponse:
        if tools:  # agent turn: stop immediately
            return LLMResponse(text="done")
        user = messages[-1].text
        clue = user.split("<clue>")[1].split("</clue>")[0] if "<clue>" in user else ""
        length = int(user.split("Length:")[1].split()[0]) if "Length:" in user else 0
        cands = []
        g = gold_by_clue.get(clue)
        if g and rng.random() < hit_rate:
            cands.append({"word": g, "confidence": round(rng.uniform(0.5, 0.9), 2)})
        for _ in range(3):
            w = "".join(rng.choice("AEIRNSTLO") for _ in range(length))
            cands.append({"word": w, "confidence": round(rng.uniform(0.1, 0.5), 2)})
        return LLMResponse(text=json.dumps({"candidates": cands}))

    return handler
