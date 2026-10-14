"""Domain models shared by every layer (vision, retrieval, solver, API).

The vision pipeline produces a :class:`PuzzleLayout`; candidate generation turns
each :class:`Entry` into ranked :class:`Candidate` objects; the solver returns a
:class:`Solution`. All models are pydantic so they serialise straight to JSON
for the HTTP API, the MCP server and the evaluation reports.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field, field_validator, model_validator

Cell = tuple[int, int]


class Direction(str, Enum):
    RIGHT = "right"
    DOWN = "down"


class CandidateSource(str, Enum):
    KNOWLEDGE_BASE = "knowledge_base"  # exact/near match retrieved from the clue KB
    WEB_DICTIONARY = "web_dictionary"  # wort-suchen.de
    LLM = "llm"  # LLM generation grounded with RAG few-shots
    AGENT = "agent"  # proposed by the tool-using agent
    USER = "user"


class Candidate(BaseModel):
    word: str
    score: float = Field(ge=0.0, le=1.0, description="Calibrated-ish confidence in [0, 1].")
    source: CandidateSource
    evidence: str | None = Field(default=None, description="Why this candidate was proposed.")

    @field_validator("word")
    @classmethod
    def _upper(cls, v: str) -> str:
        return v.strip().upper()


class Entry(BaseModel):
    """One clue together with the cells its answer occupies."""

    id: str
    clue: str
    start: Cell
    direction: Direction
    length: int = Field(gt=0)
    clue_cell: Cell | None = None

    @property
    def cells(self) -> list[Cell]:
        r, c = self.start
        if self.direction == Direction.RIGHT:
            return [(r, c + i) for i in range(self.length)]
        return [(r + i, c) for i in range(self.length)]


class PuzzleLayout(BaseModel):
    """Structure of a puzzle, independent of how it was obtained (photo or JSON)."""

    rows: int
    cols: int
    entries: list[Entry]
    clue_cells: list[Cell] = Field(default_factory=list)
    blocked_cells: list[Cell] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_bounds(self) -> PuzzleLayout:
        ids = [e.id for e in self.entries]
        if len(ids) != len(set(ids)):
            raise ValueError("entry ids must be unique")
        for e in self.entries:
            for r, c in e.cells:
                if not (0 <= r < self.rows and 0 <= c < self.cols):
                    raise ValueError(f"entry {e.id} leaves the {self.rows}x{self.cols} grid at {(r, c)}")
        return self

    def entry(self, entry_id: str) -> Entry:
        for e in self.entries:
            if e.id == entry_id:
                return e
        raise KeyError(entry_id)

    def crossings(self) -> dict[str, list[tuple[str, int, int]]]:
        """For each entry: list of (other_entry_id, my_index, other_index) sharing a cell."""
        where: dict[Cell, list[tuple[str, int]]] = {}
        for e in self.entries:
            for i, cell in enumerate(e.cells):
                where.setdefault(cell, []).append((e.id, i))
        out: dict[str, list[tuple[str, int, int]]] = {e.id: [] for e in self.entries}
        for occupants in where.values():
            for a_id, a_i in occupants:
                for b_id, b_i in occupants:
                    if a_id != b_id:
                        out[a_id].append((b_id, a_i, b_i))
        return out


class EntryResult(BaseModel):
    entry_id: str
    clue: str
    answer: str = Field(description="Final word; unknown letters are '_'.")
    confidence: float = 0.0
    source: CandidateSource | None = None
    candidates_considered: int = 0


class Usage(BaseModel):
    llm_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    cache_hits: int = 0


class Solution(BaseModel):
    complete: bool
    grid: list[list[str]] = Field(description="Letters; '#' = clue/blocked cell, '_' = unknown.")
    entries: list[EntryResult]
    usage: Usage = Field(default_factory=Usage)
    stats: dict[str, float | int | str] = Field(default_factory=dict)

    def as_text(self) -> str:
        return "\n".join(" ".join(row) for row in self.grid)
