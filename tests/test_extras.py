"""Web dictionary source, OCR, photo pipeline and CLI."""

from __future__ import annotations

from types import SimpleNamespace

import cv2
import numpy as np
from typer.testing import CliRunner

from crossword_solver.candidates import WebDictionarySource
from crossword_solver.config import REPO_ROOT, Settings
from crossword_solver.llm import FakeLLM, LLMResponse
from crossword_solver.models import Direction, Entry, PuzzleLayout
from crossword_solver.ocr import ClueOCR
from crossword_solver.pipeline import CrosswordSolver
from test_vision import render

HTML = """<html><body>
<a href="#">Wörterbuch</a><a href="#">Ätna</a><a href="#">Vesuv</a><a href="#">Etna</a><a href="#">AETNA</a>
</body></html>"""


class _Session:
    def __init__(self) -> None:
        self.calls = 0

    def get(self, url: str, headers: dict, timeout: float) -> SimpleNamespace:
        self.calls += 1
        assert "sizilianischer+vulkan/_____/" in url
        return SimpleNamespace(text=HTML, raise_for_status=lambda: None)


def test_web_dictionary_parses_filters_and_caches(tmp_path) -> None:
    session = _Session()
    src = WebDictionarySource(tmp_path, delay_s=0, session=session)  # type: ignore[arg-type]
    entry = Entry(id="E", clue="sizilianischer Vulkan", start=(0, 0), direction=Direction.DOWN, length=5)
    words = [c.word for c in src.generate(entry, "_____")]
    assert words == ["AETNA", "VESUV"]  # umlaut-normalised, deduplicated, wrong lengths dropped
    assert [c.word for c in src.generate(entry, "A____")] == ["AETNA"]
    assert session.calls == 1  # second lookup served from the SQLite cache


def test_ocr_sends_image_and_cleans_text() -> None:
    llm = FakeLLM(handler=lambda m, s, t: LLMResponse(text='"Gebäu-\ndeteil, Zimmer"'))
    text = ClueOCR(llm).read(np.full((40, 60, 3), 255, np.uint8))
    assert text == "Gebäudeteil, Zimmer"
    msg = llm.calls[0]["messages"][0]
    assert msg.images and msg.images[0].data.startswith(b"\x89PNG")
    assert ClueOCR(llm).read(np.zeros((0, 0, 3), np.uint8)) == ""


def test_solve_image_end_to_end(settings: Settings, layout: PuzzleLayout) -> None:
    _ok, buf = cv2.imencode(".png", render(layout))
    vision_llm = FakeLLM(handler=lambda m, s, t: LLMResponse(text="Vulkan bei Neapel"))
    report = CrosswordSolver(settings, vision_llm=vision_llm).solve_image(buf.tobytes(), use_agent=False)
    assert len(report.layout.entries) >= 15
    assert all(e.clue == "Vulkan bei Neapel" for e in report.layout.entries)
    assert {"vision", "ocr", "candidates", "csp"} <= set(report.timings_ms)


def test_cli_suggest_and_solve(monkeypatch, tmp_path) -> None:
    from crossword_solver import config
    from crossword_solver.cli import app

    monkeypatch.setenv("CWS_LLM_PROVIDER", "fake")
    monkeypatch.setenv("CWS_CACHE_DIR", str(tmp_path))
    config.get_settings.cache_clear()
    runner = CliRunner()
    r = runner.invoke(app, ["suggest", "Hauptstadt Norwegens", "4"])
    assert r.exit_code == 0 and "OSLO" in r.output
    out = tmp_path / "report.json"
    bench = REPO_ROOT / "data" / "benchmark" / "puzzle_01.json"
    r = runner.invoke(app, ["solve", str(bench), "--no-agent", "--out", str(out)])
    assert r.exit_code == 0, r.output
    assert out.exists()
    r = runner.invoke(app, ["kb-stats"])
    assert "records" in r.output
    config.get_settings.cache_clear()
