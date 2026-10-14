"""MCP server: exposes the solver as tools for coding agents and AI assistants.

Any MCP client (IDE coding agents, desktop assistants …) can call these tools.
Register it in the client's MCP configuration::

    {"mcpServers": {"crossword": {"command": "crossword", "args": ["mcp"]}}}

Tools
-----
* ``suggest_answers``      – ranked answers for one clue (RAG + LLM)
* ``search_clue_database`` – raw retrieval over the clue knowledge base
* ``solve_crossword``      – full solve of a JSON layout (CSP + agent)
* ``list_prompts``         – the versioned prompt templates in use
"""

from __future__ import annotations

import json
from functools import lru_cache
from typing import Any

from mcp.server.mcpserver import MCPServer

from ..llm.prompts import available, load_prompt
from ..models import PuzzleLayout
from ..pipeline import CrosswordSolver

mcp = MCPServer(
    "crossword-solver",
    instructions=(
        "Tools for solving German crossword puzzles. Answers use crossword spelling "
        "(upper case, Ä→AE, Ö→OE, Ü→UE, ß→SS). Patterns use '_' for unknown letters."
    ),
)


@lru_cache(maxsize=1)
def _solver() -> CrosswordSolver:
    return CrosswordSolver()


@mcp.tool()
def suggest_answers(clue: str, length: int, pattern: str | None = None) -> dict[str, Any]:
    """Suggest ranked answers for a German crossword clue.

    Args:
        clue: The clue text, e.g. "sizilianischer Vulkan".
        length: Number of letters of the answer.
        pattern: Optional known letters, '_' for unknown, e.g. "A__NA".
    """
    cands = _solver().suggest(clue, length, pattern)
    return {"candidates": [c.model_dump(mode="json") for c in cands[:15]]}


@mcp.tool()
def search_clue_database(
    query: str, length: int | None = None, pattern: str | None = None, k: int = 10
) -> dict[str, Any]:
    """Search the crossword clue knowledge base for clues similar to `query`."""
    hits = _solver().retriever.search(query, min(k, 50), length=length, pattern=pattern)
    return {
        "results": [
            {"clue": h.record.clue, "answer": h.record.answer, "similarity": round(h.score, 3)} for h in hits
        ]
    }


@mcp.tool()
def solve_crossword(layout_json: str, use_agent: bool = True) -> dict[str, Any]:
    """Solve a crossword given as JSON (the PuzzleLayout schema: rows, cols, entries[id, clue, start, direction, length]).

    Returns the filled grid ('#' = clue cell, '_' = unknown) and one answer per entry.
    """
    layout = PuzzleLayout.model_validate(json.loads(layout_json))
    report = _solver().solve_layout(layout, use_agent=use_agent)
    sol = report.solution
    return {
        "complete": sol.complete,
        "grid": sol.as_text(),
        "answers": {e.entry_id: e.answer for e in sol.entries},
        "usage": sol.usage.model_dump(),
    }


@mcp.tool()
def list_prompts() -> dict[str, Any]:
    """List the versioned prompt templates and show the latest version of each."""
    out = {}
    for name, versions in available().items():
        p = load_prompt(name)
        out[name] = {"versions": versions, "latest": p.id, "system_preview": p.system[:400]}
    return out


def main() -> None:
    mcp.run("stdio")


if __name__ == "__main__":
    main()
