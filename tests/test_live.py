"""Smoke tests against the real provider. Skipped unless RUN_LIVE_TESTS=1 and a key is configured."""

from __future__ import annotations

import os

import pytest

from crossword_solver.config import Settings
from crossword_solver.pipeline import CrosswordSolver

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(os.getenv("RUN_LIVE_TESTS") != "1", reason="set RUN_LIVE_TESTS=1 to call real APIs"),
]


def test_live_suggest_finds_standard_answer() -> None:
    s = Settings()
    assert s.llm_provider != "fake", "configure CWS_LLM_PROVIDER and an API key"
    words = [c.word for c in CrosswordSolver(s).suggest("sizilianischer Vulkan", 5)]
    assert "AETNA" in words[:3]
