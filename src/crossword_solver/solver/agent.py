"""Tool-using LLM agent that completes what the constraint solver could not.

The agent sees the board through tools instead of one giant prompt. Every write
goes through ``place_answer``, which validates format and crossings in code, so
the model can reason freely but can never corrupt the grid. All tool calls are
recorded in a transcript for debugging and for the UI.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from ..candidates.web_dictionary import WebDictionarySource
from ..knowledge.normalize import clean_clue, is_valid_answer, matches_pattern, normalize_answer
from ..knowledge.retrievers import Retriever
from ..llm.base import LLMClient, Message, ToolSpec
from ..llm.prompts import load_prompt
from ..models import Candidate, CandidateSource, PuzzleLayout
from .csp import Board

log = logging.getLogger(__name__)


TOOLS: list[ToolSpec] = [
    ToolSpec(
        "get_board",
        "Show the grid, the open entries with their letter patterns and their current candidate words.",
        {"type": "object", "properties": {}, "additionalProperties": False},
    ),
    ToolSpec(
        "search_knowledge_base",
        "Search a database of German crossword clues and their answers for clues similar to `query`.",
        {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Clue text or a paraphrase of it."},
                "length": {"type": "integer", "description": "Optional: only answers of this length."},
                "pattern": {"type": "string", "description": "Optional: letter pattern like 'A_T_A'."},
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    ),
    ToolSpec(
        "lookup_dictionary",
        "Look up an entry's clue in an online crossword dictionary (may be disabled).",
        {
            "type": "object",
            "properties": {"entry_id": {"type": "string"}},
            "required": ["entry_id"],
            "additionalProperties": False,
        },
    ),
    ToolSpec(
        "check_word",
        "Check whether a word fits an entry: length, known letters and the effect on crossing entries.",
        {
            "type": "object",
            "properties": {"entry_id": {"type": "string"}, "word": {"type": "string"}},
            "required": ["entry_id", "word"],
            "additionalProperties": False,
        },
    ),
    ToolSpec(
        "place_answer",
        "Write a word into an entry. Rejected if it conflicts with letters already on the board.",
        {
            "type": "object",
            "properties": {
                "entry_id": {"type": "string"},
                "word": {"type": "string"},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                "reason": {"type": "string", "description": "One short sentence."},
            },
            "required": ["entry_id", "word", "confidence"],
            "additionalProperties": False,
        },
    ),
    ToolSpec(
        "remove_answer",
        "Remove the word currently placed in an entry.",
        {
            "type": "object",
            "properties": {"entry_id": {"type": "string"}},
            "required": ["entry_id"],
            "additionalProperties": False,
        },
    ),
    ToolSpec(
        "finish",
        "Stop working. Call when the grid is complete or no further progress is possible.",
        {
            "type": "object",
            "properties": {"summary": {"type": "string"}},
            "required": ["summary"],
            "additionalProperties": False,
        },
    ),
]


@dataclass
class AgentReport:
    steps: int = 0
    tool_calls: int = 0
    placed: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    finished: bool = False
    summary: str = ""
    transcript: list[dict[str, Any]] = field(default_factory=list)


class SolverAgent:
    def __init__(
        self,
        llm: LLMClient,
        layout: PuzzleLayout,
        board: Board,
        candidates: dict[str, list[Candidate]],
        *,
        retriever: Retriever | None = None,
        dictionary: WebDictionarySource | None = None,
        max_steps: int = 25,
    ) -> None:
        self.llm = llm
        self.layout = layout
        self.board = board
        self.candidates = candidates
        self.retriever = retriever
        self.dictionary = dictionary
        self.max_steps = max_steps
        self.prompt = load_prompt("agent")

    # ---- helpers ----------------------------------------------------------------
    def open_entries(self) -> list[str]:
        grid = self.board.letters()
        return [e.id for e in self.layout.entries if "_" in self.board.pattern(e.id, grid)]

    def _render_grid(self) -> str:
        grid = self.board.letters()
        blocked = {tuple(c) for c in self.layout.clue_cells} | {tuple(c) for c in self.layout.blocked_cells}
        lines = []
        for r in range(self.layout.rows):
            lines.append(
                " ".join("#" if (r, c) in blocked else grid.get((r, c), "_") for c in range(self.layout.cols))
            )
        return "\n".join(lines)

    # ---- tools --------------------------------------------------------------------
    def _get_board(self) -> dict[str, Any]:
        grid = self.board.letters()
        open_ids = set(self.open_entries())
        entries = []
        for e in self.layout.entries:
            patt = self.board.pattern(e.id, grid)
            if e.id not in open_ids:
                continue
            fitting = [c for c in self.candidates.get(e.id, []) if matches_pattern(c.word, patt)]
            entries.append(
                {
                    "entry_id": e.id,
                    "clue": e.clue,
                    "length": e.length,
                    "direction": e.direction.value,
                    "pattern": patt,
                    "candidates": [{"word": c.word, "score": c.score} for c in fitting[:8]],
                }
            )
        entries.sort(key=lambda d: str(d["pattern"]).count("_"))
        return {
            "grid": self._render_grid(),
            "legend": "# = clue/blocked cell, _ = empty",
            "open_entries": entries,
            "filled_entries": len(self.layout.entries) - len(entries),
        }

    def _search_kb(self, query: str, length: int | None = None, pattern: str | None = None) -> dict[str, Any]:
        if self.retriever is None:
            return {"error": "knowledge base not available"}
        patt = "".join("_" if ch in "_?." else ch for ch in pattern.upper()) if pattern else None
        hits = self.retriever.search(clean_clue(query), 10, length=length, pattern=patt)
        return {
            "results": [
                {"clue": h.record.clue, "answer": h.record.answer, "similarity": round(h.score, 3)}
                for h in hits
            ]
        }

    def _lookup_dictionary(self, entry_id: str) -> dict[str, Any]:
        if self.dictionary is None:
            return {"error": "online dictionary disabled in this deployment"}
        e = self.layout.entry(entry_id)
        cands = self.dictionary.generate(e, self.board.pattern(entry_id))
        return {"words": [c.word for c in cands]}

    def _check_word(self, entry_id: str, word: str) -> dict[str, Any]:
        e = self.layout.entry(entry_id)
        w = normalize_answer(word)
        patt = self.board.pattern(entry_id, self.board.letters(exclude=entry_id))
        res: dict[str, Any] = {
            "word": w,
            "length_ok": len(w) == e.length,
            "pattern": patt,
            "fits_pattern": is_valid_answer(w, e.length, patt),
        }
        if res["length_ok"]:
            effects = []
            for other, i, j in self.layout.crossings()[entry_id]:
                if other in self.board.answers:
                    continue
                op = list(self.board.pattern(other))
                op[j] = w[i]
                new_patt = "".join(op)
                remaining = [
                    c.word for c in self.candidates.get(other, []) if matches_pattern(c.word, new_patt)
                ]
                effects.append(
                    {"entry_id": other, "new_pattern": new_patt, "candidates_still_fitting": remaining[:5]}
                )
            res["crossing_effects"] = effects
        return res

    def _place(self, entry_id: str, word: str, confidence: float, reason: str = "") -> dict[str, Any]:
        e = self.layout.entry(entry_id)
        w = normalize_answer(word)
        if not is_valid_answer(w, e.length):
            return {"ok": False, "error": f"'{w}' is not {e.length} letters A-Z"}
        clashes = self.board.place(
            entry_id,
            Candidate(
                word=w,
                score=max(0.0, min(1.0, float(confidence))),
                source=CandidateSource.AGENT,
                evidence=reason or None,
            ),
        )
        if clashes:
            return {
                "ok": False,
                "conflicts": [
                    {"with_entry": o, "cell": list(cell), "board_letter": a, "your_letter": b}
                    for o, cell, a, b in clashes
                ],
            }
        return {
            "ok": True,
            "pattern_now": self.board.pattern(entry_id),
            "open_entries_left": len(self.open_entries()),
        }

    def _remove(self, entry_id: str) -> dict[str, Any]:
        removed = self.board.remove(entry_id)
        return {"ok": removed is not None}

    def _dispatch(self, name: str, args: dict[str, Any], report: AgentReport) -> Any:
        try:
            if name == "get_board":
                return self._get_board()
            if name == "search_knowledge_base":
                return self._search_kb(**args)
            if name == "lookup_dictionary":
                return self._lookup_dictionary(**args)
            if name == "check_word":
                return self._check_word(**args)
            if name == "place_answer":
                out = self._place(**args)
                if out.get("ok"):
                    report.placed.append(args["entry_id"])
                return out
            if name == "remove_answer":
                out = self._remove(**args)
                if out.get("ok"):
                    report.removed.append(args["entry_id"])
                return out
            if name == "finish":
                report.finished = True
                report.summary = str(args.get("summary", ""))
                return {"ok": True}
            return {"error": f"unknown tool {name}"}
        except (KeyError, TypeError, ValueError) as e:  # bad arguments from the model
            return {"error": f"invalid arguments for {name}: {e}"}

    # ---- loop -------------------------------------------------------------------
    def run(self) -> AgentReport:
        report = AgentReport()
        open_now = self.open_entries()
        if not open_now:
            report.finished, report.summary = True, "grid already complete"
            return report
        system, user = self.prompt.render(max_steps=self.max_steps, open_count=len(open_now))
        tools = [t for t in TOOLS if t.name != "lookup_dictionary" or self.dictionary is not None]
        messages = [Message.user(user)]

        while report.steps < self.max_steps and not report.finished:
            report.steps += 1
            resp = self.llm.chat(messages, system=system, tools=tools)
            messages.append(resp.as_message())
            if not resp.tool_calls:
                report.summary = resp.text
                break
            for call in resp.tool_calls:
                report.tool_calls += 1
                result = self._dispatch(call.name, call.arguments, report)
                report.transcript.append(
                    {
                        "step": report.steps,
                        "tool": call.name,
                        "args": call.arguments,
                        "result": _short(result),
                    }
                )
                messages.append(Message.tool_result(call.id, result))
        log.info(
            "agent_done",
            extra={"steps": report.steps, "placed": len(report.placed), "finished": report.finished},
        )
        return report


def _short(result: Any, limit: int = 400) -> str:
    s = json.dumps(result, ensure_ascii=False)
    return s if len(s) <= limit else s[:limit] + "…"
