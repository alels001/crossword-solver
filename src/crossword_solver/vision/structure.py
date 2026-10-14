"""Stages 8–10: link clue cells to arrows and compute answer lengths."""

from __future__ import annotations

from dataclasses import dataclass

from .arrows import Arrow

Cell = tuple[int, int]


@dataclass(frozen=True)
class LinkedEntry:
    clue_cell: Cell
    start: Cell
    direction: str
    length: int
    arrow_kind: str


def _neighbors(r: int, c: int, rows: int, cols: int) -> list[Cell]:
    cand = [(r, c + 1), (r, c - 1), (r + 1, c), (r - 1, c)]
    return [(rr, cc) for rr, cc in cand if 0 <= rr < rows and 0 <= cc < cols]


def link_clues_to_arrows(
    clue_cells: list[Cell], arrows: list[Arrow], rows: int, cols: int
) -> list[tuple[Cell, Arrow]]:
    """Pass 1: small arrows adjacent in arrow direction. Pass 2: large arrows by grid walk."""
    small = [a for a in arrows if a.kind == "small"]
    large = [a for a in arrows if a.kind == "large"]
    assigned: dict[Cell, Arrow] = {}
    used_small: set[int] = set()
    used_large: set[int] = set()

    for qr, qc in clue_cells:
        best: tuple[float, int] | None = None
        for i, a in enumerate(small):
            if i in used_small or abs(a.row - qr) + abs(a.col - qc) != 1:
                continue
            if (a.direction == "right" and a.col > qc) or (a.direction == "down" and a.row > qr):
                d = ((a.row - qr) ** 2 + (a.col - qc) ** 2) ** 0.5
                if best is None or d < best[0]:
                    best = (d, i)
        if best is not None:
            assigned[(qr, qc)] = small[best[1]]
            used_small.add(best[1])

    clue_set = set(clue_cells)
    for r in range(rows):
        for c in range(cols):
            # Case A: an unused large arrow claims an adjacent unassigned clue.
            for i, a in enumerate(large):
                if i in used_large or (a.row, a.col) != (r, c):
                    continue
                for nb in _neighbors(r, c, rows, cols):
                    if nb in clue_set and nb not in assigned:
                        assigned[nb] = a
                        used_large.add(i)
                        break
            # Case B: an unassigned clue looks for a large arrow nearby (incl. down-diagonals).
            if (r, c) in clue_set and (r, c) not in assigned:
                around = _neighbors(r, c, rows, cols) + [(r + 1, c + 1), (r + 1, c - 1)]
                for nb in around:
                    if not (0 <= nb[0] < rows and 0 <= nb[1] < cols):
                        continue
                    hit = next(
                        (i for i, a in enumerate(large) if i not in used_large and (a.row, a.col) == nb), None
                    )
                    if hit is not None:
                        assigned[(r, c)] = large[hit]
                        used_large.add(hit)
                        break
    return sorted(assigned.items())


def answer_length(start: Cell, direction: str, rows: int, cols: int, stop_cells: set[Cell]) -> int:
    """Walk from the start cell (inclusive) until the border or a clue/blocked cell."""
    r, c = start
    n = 0
    while 0 <= r < rows and 0 <= c < cols and (r, c) not in stop_cells:
        n += 1
        r, c = (r, c + 1) if direction == "right" else (r + 1, c)
    return n


def build_entries(
    clue_cells: list[Cell], arrows: list[Arrow], rows: int, cols: int, blocked: set[Cell]
) -> list[LinkedEntry]:
    stop = set(clue_cells) | blocked
    out = []
    for clue_cell, arrow in link_clues_to_arrows(clue_cells, arrows, rows, cols):
        start = (arrow.row, arrow.col)
        n = answer_length(start, arrow.direction, rows, cols, stop)
        if n >= 2:  # one-letter "answers" are detection errors
            out.append(LinkedEntry(clue_cell, start, arrow.direction, n, arrow.kind))
    return out
