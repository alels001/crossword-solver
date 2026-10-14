"""Constraint solver and tool-using agent."""

from __future__ import annotations

import random

from crossword_solver.llm import LLMResponse, ScriptedLLM, ToolCall
from crossword_solver.models import Candidate, CandidateSource, PuzzleLayout
from crossword_solver.solver import TOOLS, Board, SolverAgent, objective, solve, to_solution


def _c(word: str, score: float = 0.5) -> Candidate:
    return Candidate(word=word, score=score, source=CandidateSource.LLM)


def test_gold_only_candidates_solve_completely(layout: PuzzleLayout, gold: dict[str, str]) -> None:
    res = solve(layout, {k: [_c(v)] for k, v in gold.items()})
    sol = to_solution(layout, res.assignment, res.grid)
    assert sol.complete
    assert {e.entry_id: e.answer for e in sol.entries} == gold
    assert sol.grid[1] == list("GARTENHAUS")
    assert sol.grid[0][0] == "#"


def test_crossings_pick_gold_over_higher_scored_distractors(
    layout: PuzzleLayout, gold: dict[str, str]
) -> None:
    rng = random.Random(7)
    cands = {}
    for e in layout.entries:
        noise = ["".join(rng.choice("AEIOU") for _ in range(e.length)) for _ in range(4)]
        cands[e.id] = [_c(gold[e.id], 0.5)] + [_c(w, 0.7) for w in noise]
    res = solve(layout, cands)
    answers = {e.entry_id: e.answer for e in to_solution(layout, res.assignment, res.grid).entries}
    assert sum(answers[k] == v for k, v in gold.items()) >= 24


def test_missing_candidates_leave_entry_open_instead_of_failing(
    layout: PuzzleLayout, gold: dict[str, str]
) -> None:
    cands = {k: [_c(v)] for k, v in gold.items() if k != "E01"}  # GARTENHAUS unknown
    res = solve(layout, cands)
    assert res.assignment["E01"] is None
    sol = to_solution(layout, res.assignment, res.grid)
    # crossing entries still fix the letters they share with the open entry
    assert sol.entries[0].answer == "_ARTE_HAU_"
    assert not sol.complete
    assert sum(e.answer == gold[e.entry_id] for e in sol.entries) == 25


def test_locked_answers_are_respected(layout: PuzzleLayout, gold: dict[str, str]) -> None:
    cands = {k: [_c(v)] for k, v in gold.items()}
    cands["E05"] = [_c("VESUV", 0.99), _c("AETNA", 0.1)]
    res = solve(layout, cands, locked={"E01": _c("GARTENHAUS", 1.0)})
    assert res.assignment["E01"].word == "GARTENHAUS"
    assert res.assignment["E05"].word == "AETNA"  # VESUV clashes with the locked E in row 1


def test_objective_counts_crossings(layout: PuzzleLayout, gold: dict[str, str]) -> None:
    a = {k: _c(v, 0.5) for k, v in gold.items()}
    links = sum(len(x) for x in layout.crossings().values()) / 2
    assert objective(layout, a, crossing_bonus=0.25) == 0.5 * len(gold) + 0.25 * links


def test_board_detects_conflicts(layout: PuzzleLayout) -> None:
    b = Board(layout)
    assert b.place("E01", _c("GARTENHAUS")) == []
    clashes = b.place("E02", _c("XBREGEN"))
    assert clashes and clashes[0][0] == "E01"
    assert b.pattern("E02") == "A______"


def test_agent_uses_tools_and_cannot_corrupt_grid(layout: PuzzleLayout, gold: dict[str, str]) -> None:
    board = Board(layout)
    for k, v in gold.items():
        if k not in ("E05", "E08"):
            board.answers[k] = _c(v, 0.8)
    script = [
        LLMResponse(text="", tool_calls=[ToolCall("1", "get_board", {})]),
        LLMResponse(text="", tool_calls=[ToolCall("2", "check_word", {"entry_id": "E05", "word": "Ätna"})]),
        LLMResponse(
            text="",
            tool_calls=[
                ToolCall("3", "place_answer", {"entry_id": "E05", "word": "VESUV", "confidence": 0.9})
            ],
        ),
        LLMResponse(
            text="",
            tool_calls=[
                ToolCall("4", "place_answer", {"entry_id": "E05", "word": "AETNA", "confidence": 0.9}),
                ToolCall(
                    "5", "search_knowledge_base", {"query": "veraltet: eben", "length": 4, "pattern": "_UST"}
                ),
            ],
        ),
        LLMResponse(
            text="",
            tool_calls=[
                ToolCall("6", "place_answer", {"entry_id": "E08", "word": "JUST", "confidence": 0.7})
            ],
        ),
        LLMResponse(text="", tool_calls=[ToolCall("7", "finish", {"summary": "done"})]),
    ]
    llm = ScriptedLLM(script)
    agent = SolverAgent(llm, layout, board, {"E05": [_c("AETNA")]}, max_steps=10)
    report = agent.run()

    assert report.finished and report.placed == ["E05", "E08"]
    assert board.answers["E05"].word == "AETNA" and board.answers["E08"].word == "JUST"
    results = {t["tool"] + t["args"].get("word", ""): t["result"] for t in report.transcript}
    assert '"ok": false' in results["place_answerVESUV"]  # conflicting write rejected
    assert '"fits_pattern": true' in results["check_wordÄtna"]  # normalised to AETNA
    assert agent.open_entries() == []
    # dictionary tool is hidden when no dictionary is configured
    first_call_tools = {t.name for t in llm.calls[0]["tools"]}
    assert "lookup_dictionary" not in first_call_tools and len(first_call_tools) == len(TOOLS) - 1
    # tool results are fed back with matching ids
    assert llm.calls[1]["messages"][-1].tool_call_id == "1"


def test_agent_stops_on_step_budget(layout: PuzzleLayout) -> None:
    looping = ScriptedLLM(
        [LLMResponse(text="", tool_calls=[ToolCall(str(i), "get_board", {})]) for i in range(50)]
    )
    report = SolverAgent(looping, layout, Board(layout), {}, max_steps=3).run()
    assert report.steps == 3 and not report.finished


def test_agent_handles_bad_arguments(layout: PuzzleLayout) -> None:
    llm = ScriptedLLM(
        [
            LLMResponse(
                text="",
                tool_calls=[
                    ToolCall("1", "place_answer", {"entry_id": "NOPE", "word": "X", "confidence": 1})
                ],
            ),
            LLMResponse(text="", tool_calls=[ToolCall("2", "teleport", {})]),
            LLMResponse(text="giving up"),
        ]
    )
    report = SolverAgent(llm, layout, Board(layout), {}, max_steps=5).run()
    assert "invalid arguments" in report.transcript[0]["result"]
    assert "unknown tool" in report.transcript[1]["result"]
    assert report.summary == "giving up"
