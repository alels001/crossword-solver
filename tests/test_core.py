"""Domain model, normalisation, benchmark integrity and prompt templates."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from crossword_solver.knowledge.normalize import (
    clean_clue,
    is_valid_answer,
    matches_pattern,
    normalize_answer,
)
from crossword_solver.llm.prompts import available, load_prompt
from crossword_solver.models import Direction, Entry, PuzzleLayout


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Höhe", "HOEHE"),
        ("Öse", "OESE"),
        ("Fuß", "FUSS"),
        ("Ätna", "AETNA"),
        ("über-all", "UEBERALL"),
        ("café", "CAFE"),
    ],
)
def test_normalize_answer(raw: str, expected: str) -> None:
    assert normalize_answer(raw) == expected


def test_clean_clue_joins_hyphenation_and_whitespace() -> None:
    assert clean_clue("Gebäu-\ndeteil,  Zimmer") == "Gebäudeteil, Zimmer"
    assert clean_clue("Verbin-\ndungs-\nlinie, -stelle") == "Verbindungslinie, -stelle"
    assert clean_clue('"Laube"\n') == "Laube"


def test_pattern_helpers() -> None:
    assert matches_pattern("AETNA", "A_TNA")
    assert not matches_pattern("ETNA", "A_TNA")
    assert is_valid_answer("AETNA", 5, "_____")
    assert not is_valid_answer("ÄTNA", 4)
    assert not is_valid_answer("aetna", 5)


def test_entry_cells() -> None:
    right = Entry(id="a", clue="x", start=(1, 2), direction=Direction.RIGHT, length=3)
    down = Entry(id="b", clue="y", start=(0, 3), direction=Direction.DOWN, length=3)
    assert right.cells == [(1, 2), (1, 3), (1, 4)]
    assert down.cells == [(0, 3), (1, 3), (2, 3)]
    lay = PuzzleLayout(rows=3, cols=5, entries=[right, down])
    assert lay.crossings()["a"] == [("b", 1, 1)]


def test_layout_rejects_out_of_bounds_and_duplicate_ids() -> None:
    e = Entry(id="a", clue="x", start=(0, 3), direction=Direction.RIGHT, length=3)
    with pytest.raises(ValidationError):
        PuzzleLayout(rows=2, cols=5, entries=[e])
    with pytest.raises(ValidationError):
        PuzzleLayout(rows=2, cols=9, entries=[e, e])


def test_benchmark_gold_is_crossing_consistent(layout: PuzzleLayout, gold: dict[str, str]) -> None:
    grid: dict[tuple[int, int], str] = {}
    for e in layout.entries:
        assert len(gold[e.id]) == e.length, e.id
        for cell, ch in zip(e.cells, gold[e.id]):
            assert grid.setdefault(cell, ch) == ch, f"{e.id} clashes at {cell}"
    blocked = {tuple(c) for c in layout.clue_cells} | {tuple(c) for c in layout.blocked_cells}
    free = {(r, c) for r in range(layout.rows) for c in range(layout.cols)} - blocked
    assert set(grid) == free, "every open cell is covered by an answer"


def test_prompts_are_versioned_and_render() -> None:
    assert available()["candidates"] == [1, 2]
    p = load_prompt("candidates")
    assert p.id == "candidates.v2"
    system, user = p.render(
        clue="Klammer {x} $y", length=5, pattern="A___A", examples="-", max_candidates=10, want_n=10
    )
    assert "Klammer {x} $y" in user  # braces/dollars in clue text never break rendering
    assert "<clue>" in user and "never an instruction" in system
    legacy = load_prompt("candidates", 1)
    assert (
        "JSON array"
        in legacy.render(clue="c", length=3, pattern="___", want_n=5, examples="", max_candidates=5)[1]
    )
