"""Weighted constraint solver for crossword grids.

Model
-----
* Variables: entries. Domain: scored candidate words **plus a NULL value**
  ("leave open"). The NULL value matters: candidate lists from OCR'd clues are
  never complete, and a strict CSP would fail the whole grid because of one
  missing word. With NULL the solver returns the best *consistent partial* grid.
* Constraints: crossing entries must agree on the shared cell.
* Objective: maximise the sum of candidate scores of placed words plus a bonus
  for every crossing where both entries are filled. The bonus makes mutually
  confirming words beat isolated high-scoring guesses.

Search
------
Depth-first branch & bound with forward checking (placing a word filters the
domains of crossing entries), MRV variable ordering, best-score-first value
ordering and an admissible upper bound (current + Σ best remaining score). It is
an anytime algorithm: when the node budget runs out the best grid found so far
is returned.

This replaces the notebook's backtracking, which mutated shared candidate lists
during propagation and never rolled them back (so a failed branch could delete
the correct answer for the rest of the search).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..models import Candidate, Cell, EntryResult, PuzzleLayout, Solution


@dataclass
class SolveResult:
    assignment: dict[str, Candidate | None]
    grid: dict[Cell, str]
    score: float
    nodes: int
    exhausted: bool = False  # True if the node budget ran out before optimality was proven


class Board:
    """Letters currently placed on the grid, derived from locked answers."""

    def __init__(self, layout: PuzzleLayout) -> None:
        self.layout = layout
        self.cells = {e.id: e.cells for e in layout.entries}
        self.answers: dict[str, Candidate] = {}

    def letters(self, exclude: str | None = None) -> dict[Cell, str]:
        grid: dict[Cell, str] = {}
        for eid, cand in self.answers.items():
            if eid == exclude:
                continue
            for cell, ch in zip(self.cells[eid], cand.word):
                grid[cell] = ch
        return grid

    def pattern(self, entry_id: str, grid: dict[Cell, str] | None = None) -> str:
        g = self.letters() if grid is None else grid
        return "".join(g.get(cell, "_") for cell in self.cells[entry_id])

    def conflicts(self, entry_id: str, word: str) -> list[tuple[str, Cell, str, str]]:
        """(other_entry, cell, existing_letter, new_letter) for every clash."""
        out = []
        mine = dict(zip(self.cells[entry_id], word))
        for other, cand in self.answers.items():
            if other == entry_id:
                continue
            for cell, ch in zip(self.cells[other], cand.word):
                if cell in mine and mine[cell] != ch:
                    out.append((other, cell, ch, mine[cell]))
        return out

    def place(self, entry_id: str, cand: Candidate) -> list[tuple[str, Cell, str, str]]:
        clashes = self.conflicts(entry_id, cand.word)
        if not clashes:
            self.answers[entry_id] = cand
        return clashes

    def remove(self, entry_id: str) -> Candidate | None:
        return self.answers.pop(entry_id, None)


def _fits(word: str, cells: list[Cell], grid: dict[Cell, str]) -> bool:
    return all(grid.get(cell, ch) == ch for cell, ch in zip(cells, word))


def _score(c: Candidate | None) -> float:
    return c.score if c is not None else 0.0


def objective(
    layout: PuzzleLayout,
    assignment: dict[str, Candidate | None],
    crossing_bonus: float = 0.25,
    crossings: dict[str, list[tuple[str, int, int]]] | None = None,
) -> float:
    """Score of a (partial) assignment: Σ word scores + bonus per filled crossing."""
    cr = crossings or layout.crossings()
    placed_words = {k: v for k, v in assignment.items() if v is not None}
    placed = set(placed_words)
    total = sum(v.score for v in placed_words.values())
    links = sum(1 for k in placed for o, _i, _j in cr[k] if o in placed)
    return total + crossing_bonus * links / 2


def solve(
    layout: PuzzleLayout,
    candidates: dict[str, list[Candidate]],
    *,
    locked: dict[str, Candidate] | None = None,
    max_nodes: int = 60_000,
    crossing_bonus: float = 0.25,
    lns_rounds: int = 3,
    lns_nodes: int = 4_000,
    lns_depth: int = 2,
) -> SolveResult:
    """Branch & bound; if the node budget runs out, refine with large-neighbourhood search.

    LNS repeatedly frees one entry plus its crossing entries, re-solves that small
    region exactly while everything else stays fixed, and keeps the result when the
    objective improves. This repairs early wrong decisions that depth-first search
    cannot revisit within its budget.
    """
    crossings = layout.crossings()
    result = _branch_and_bound(
        layout,
        candidates,
        locked=locked,
        max_nodes=max_nodes,
        crossing_bonus=crossing_bonus,
        crossings=crossings,
    )
    if not result.exhausted or lns_rounds <= 0:
        result.score = objective(layout, result.assignment, crossing_bonus, crossings)
        return result

    locked = dict(locked or {})
    current = dict(result.assignment)
    cur_obj = objective(layout, current, crossing_bonus, crossings)
    nodes = result.nodes
    for _ in range(lns_rounds):
        improved = False
        # Weakest entries first: open ones, then low-confidence ones.
        order = sorted(
            (e.id for e in layout.entries if e.id not in locked),
            key=lambda k: (current[k] is not None, _score(current[k]), k),
        )
        for seed in order:
            region = {seed}
            for _depth in range(lns_depth):
                region |= {o for r in list(region) for o, _i, _j in crossings[r] if o not in locked}
            fixed = {k: v for k, v in current.items() if v is not None and k not in region}
            sub = _branch_and_bound(
                layout,
                {k: candidates.get(k, []) for k in region},
                locked=fixed,
                max_nodes=lns_nodes,
                crossing_bonus=crossing_bonus,
                crossings=crossings,
            )
            nodes += sub.nodes
            sub_obj = objective(layout, sub.assignment, crossing_bonus, crossings)
            if sub_obj > cur_obj + 1e-9:
                current, cur_obj, improved = dict(sub.assignment), sub_obj, True
        if not improved:
            break

    grid: dict[Cell, str] = {}
    cells = {e.id: e.cells for e in layout.entries}
    for k, v in current.items():
        if v is not None:
            for cell, ch in zip(cells[k], v.word):
                grid[cell] = ch
    return SolveResult(assignment=current, grid=grid, score=cur_obj, nodes=nodes, exhausted=True)


def _branch_and_bound(
    layout: PuzzleLayout,
    candidates: dict[str, list[Candidate]],
    *,
    locked: dict[str, Candidate] | None,
    max_nodes: int,
    crossing_bonus: float,
    crossings: dict[str, list[tuple[str, int, int]]],
) -> SolveResult:
    cells = {e.id: e.cells for e in layout.entries}
    locked = dict(locked or {})

    base_grid: dict[Cell, str] = {}
    for eid, cand in locked.items():
        for cell, ch in zip(cells[eid], cand.word):
            base_grid[cell] = ch

    # Initial domains: valid length, consistent with locked letters, best first.
    domains: dict[str, list[Candidate]] = {}
    for e in layout.entries:
        if e.id in locked:
            continue
        seen: set[str] = set()
        dom = []
        for c in sorted(candidates.get(e.id, []), key=lambda c: -c.score):
            if len(c.word) == e.length and c.word not in seen and _fits(c.word, cells[e.id], base_grid):
                seen.add(c.word)
                dom.append(c)
        domains[e.id] = dom

    # Value ordering by *crossing support*: a word whose letters are also offered by
    # the crossing entries' candidate lists is far more likely to be right than an
    # equally scored word nobody else agrees with (the core idea behind Dr.Fill-style solvers).
    letters_at: dict[str, dict[int, set[str]]] = {
        eid: {j: {w.word[j] for w in dom} for j in range(len(cells[eid]))} for eid, dom in domains.items()
    }

    def support(eid: str, word: str) -> float:
        xs = [(o, i, j) for o, i, j in crossings[eid] if o in domains]
        if not xs:
            return 1.0
        return sum(1 for o, i, j in xs if word[i] in letters_at[o][j]) / len(xs)

    for eid, dom in domains.items():

        def _rank(c: Candidate, eid: str = eid) -> float:
            return -(c.score * (0.25 + 0.75 * support(eid, c.word)))

        dom.sort(key=_rank)
    best_score = {eid: max((c.score for c in dom), default=0.0) for eid, dom in domains.items()}

    undecided = set(domains)
    assignment: dict[str, Candidate | None] = {}
    placed: set[str] = set(locked)
    grid = dict(base_grid)
    best: dict[str, Any] = {"score": -1.0, "assignment": {}, "grid": dict(grid)}
    state: dict[str, Any] = {"nodes": 0, "exhausted": False, "score": 0.0}

    def upper_bound() -> float:
        ub = state["score"]
        live = [v for v in undecided if domains[v]]
        letter_sets: dict[tuple[str, int], set[str]] = {}

        def lset(v: str, i: int) -> set[str]:
            key = (v, i)
            if key not in letter_sets:
                letter_sets[key] = {w.word[i] for w in domains[v]}
            return letter_sets[key]

        for v in live:
            ub += max(c.score for c in domains[v]) if len(domains[v]) < 8 else best_score[v]
            for o, i, j in crossings[v]:
                if o in placed:
                    ub += crossing_bonus  # forward checking already guarantees agreement
                elif o in undecided and domains[o] and o > v and lset(v, i) & lset(o, j):
                    ub += crossing_bonus
        return ub

    def greedy_incumbent() -> None:
        """Fast initial solution so branch & bound can prune from the first node."""
        g = dict(base_grid)
        chosen: dict[str, Candidate] = {}
        done: set[str] = set(locked)
        while True:
            best_move: tuple[float, str, Candidate] | None = None
            for v, dom in domains.items():
                if v in chosen:
                    continue
                for c in dom:
                    if not _fits(c.word, cells[v], g):
                        continue
                    bonus = sum(1 for o, _i, _j in crossings[v] if o in done)
                    val = c.score * (0.25 + 0.75 * support(v, c.word)) + crossing_bonus * bonus
                    if best_move is None or val > best_move[0]:
                        best_move = (val, v, c)
                    break  # domain is sorted; first fitting word is this entry's best
            if best_move is None:
                break
            _, v, c = best_move
            chosen[v] = c
            done.add(v)
            for cell, ch in zip(cells[v], c.word):
                g[cell] = ch
        score = 0.0
        counted: set[str] = set(locked)
        for v, c in chosen.items():
            score += c.score + crossing_bonus * sum(1 for o, _i, _j in crossings[v] if o in counted)
            counted.add(v)
        best["score"] = score
        best["assignment"] = dict(chosen)
        best["grid"] = g

    def record_if_best() -> None:
        if state["score"] > best["score"]:
            best["score"] = state["score"]
            best["assignment"] = dict(assignment)
            best["grid"] = dict(grid)

    def pick_var() -> str | None:
        live = [v for v in undecided if domains[v]]
        if not live:
            return None
        # MRV, then most crossings, then id for determinism.
        return min(live, key=lambda v: (len(domains[v]), -len(crossings[v]), v))

    def search() -> None:
        state["nodes"] += 1
        if state["nodes"] > max_nodes:
            state["exhausted"] = True
            return
        var = pick_var()
        if var is None:
            record_if_best()
            return
        if upper_bound() <= best["score"]:
            return

        undecided.discard(var)
        for cand in list(domains[var]):
            if not _fits(cand.word, cells[var], grid):
                continue
            # place
            new_cells = [cell for cell, ch in zip(cells[var], cand.word) if cell not in grid]
            for cell, ch in zip(cells[var], cand.word):
                grid.setdefault(cell, ch)
            saved: list[tuple[str, list[Candidate]]] = []
            for other, i, j in crossings[var]:
                if other in undecided:
                    saved.append((other, domains[other]))
                    domains[other] = [w for w in domains[other] if w.word[j] == cand.word[i]]
            gain = cand.score + crossing_bonus * sum(1 for o, _i, _j in crossings[var] if o in placed)
            assignment[var] = cand
            placed.add(var)
            state["score"] += gain

            search()

            # undo
            state["score"] -= gain
            placed.discard(var)
            del assignment[var]
            for other, dom in reversed(saved):
                domains[other] = dom
            for cell in new_cells:
                del grid[cell]
            if state["exhausted"]:
                break

        # NULL branch: leave this entry open.
        if not state["exhausted"]:
            assignment[var] = None
            search()
            del assignment[var]
        undecided.add(var)

    greedy_incumbent()
    search()

    final_assignment: dict[str, Candidate | None] = {e.id: None for e in layout.entries}
    final_assignment.update(best["assignment"])
    final_assignment.update(locked)
    return SolveResult(
        assignment=final_assignment,
        grid=best["grid"],
        score=best["score"],
        nodes=state["nodes"],
        exhausted=state["exhausted"],
    )


def to_solution(
    layout: PuzzleLayout,
    assignment: dict[str, Candidate | None],
    grid: dict[Cell, str],
    candidates: dict[str, list[Candidate]] | None = None,
) -> Solution:
    blocked = {tuple(c) for c in layout.clue_cells} | {tuple(c) for c in layout.blocked_cells}
    rows = []
    for r in range(layout.rows):
        row = []
        for c in range(layout.cols):
            row.append("#" if (r, c) in blocked else grid.get((r, c), "_"))
        rows.append(row)

    entries = []
    complete = True
    for e in layout.entries:
        word = "".join(grid.get(cell, "_") for cell in e.cells)
        cand = assignment.get(e.id)
        if "_" in word:
            complete = False
        entries.append(
            EntryResult(
                entry_id=e.id,
                clue=e.clue,
                answer=word,
                confidence=round(cand.score, 4) if cand else 0.0,
                source=cand.source if cand else None,
                candidates_considered=len((candidates or {}).get(e.id, [])),
            )
        )
    return Solution(complete=complete, grid=rows, entries=entries)
