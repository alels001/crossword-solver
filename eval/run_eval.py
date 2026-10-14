"""Evaluation harness.

Two complementary evaluations:

1. **End-to-end** — every ``data/benchmark/puzzle_*.json`` is solved with the
   configured LLM provider. Benchmark clues are removed from the knowledge base
   first (``holdout``) so retrieval cannot leak gold answers. Reports entry
   accuracy, letter accuracy, precision of filled entries, LLM calls, tokens,
   cost and latency, plus the prompt versions used.

2. **Solver robustness** — independent of any LLM: the solver receives the gold
   answer (with a given probability) mixed with realistic distractors
   (other real words of the same length) and random noise words, all with
   random confidences. This isolates the constraint-solving stage.

Usage::

    python -m eval.run_eval                     # uses CWS_LLM_PROVIDER from .env
    CWS_LLM_PROVIDER=openai python -m eval.run_eval
"""

from __future__ import annotations

import json
import random
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from crossword_solver.config import Settings, get_settings  # noqa: E402
from crossword_solver.knowledge import ClueStore  # noqa: E402
from crossword_solver.llm.prompts import available  # noqa: E402
from crossword_solver.models import Candidate, CandidateSource, PuzzleLayout  # noqa: E402
from crossword_solver.pipeline import CrosswordSolver  # noqa: E402
from crossword_solver.solver import solve, to_solution  # noqa: E402


def load_benchmark(folder: Path) -> list[dict]:
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(folder.glob("puzzle_*.json"))]


def score_solution(layout: PuzzleLayout, answers: dict[str, str], gold: dict[str, str]) -> dict[str, float]:
    correct = sum(1 for k, g in gold.items() if answers.get(k) == g)
    filled = [k for k, a in answers.items() if a and "_" not in a]
    filled_correct = sum(1 for k in filled if answers[k] == gold.get(k))
    gold_cells: dict[tuple[int, int], str] = {}
    pred_cells: dict[tuple[int, int], str] = {}
    for e in layout.entries:
        for cell, g, p in zip(e.cells, gold[e.id], answers.get(e.id, "_" * e.length)):
            gold_cells[cell] = g
            if p != "_":
                pred_cells[cell] = p
    letters_ok = sum(1 for c, g in gold_cells.items() if pred_cells.get(c) == g)
    return {
        "entry_accuracy": correct / len(gold),
        "letter_accuracy": letters_ok / len(gold_cells),
        "fill_rate": len(filled) / len(gold),
        "precision_filled": filled_correct / len(filled) if filled else 0.0,
    }


def end_to_end(benchmark: list[dict], settings: Settings, use_agent: bool, holdout: bool) -> list[dict]:
    base_store = ClueStore.load(settings.knowledge_base_path)
    rows = []
    for puzzle in benchmark:
        layout = PuzzleLayout.model_validate(puzzle["layout"])
        store = base_store.without_clues(e.clue for e in layout.entries) if holdout else base_store
        solver = CrosswordSolver(settings, store=store)
        t0 = time.perf_counter()
        report = solver.solve_layout(layout, use_agent=use_agent)
        latency = time.perf_counter() - t0
        answers = {e.entry_id: e.answer for e in report.solution.entries}
        metrics = score_solution(layout, answers, puzzle["gold"])
        rows.append(
            {
                "puzzle": puzzle["name"],
                "provider": settings.llm_provider,
                "model": solver._llm.model,
                "use_agent": use_agent,
                "kb_records": len(store),
                "kb_removed_for_holdout": len(base_store) - len(store),
                **{k: round(v, 4) for k, v in metrics.items()},
                "latency_s": round(latency, 2),
                **report.solution.usage.model_dump(),
                "timings_ms": report.timings_ms,
                "errors": {
                    k: {"pred": answers[k], "gold": g}
                    for k, g in puzzle["gold"].items()
                    if answers.get(k) != g
                },
            }
        )
    return rows


def solver_robustness(benchmark: list[dict], kb_path: Path, seeds: int = 10, p_gold: float = 0.85) -> dict:
    pool: dict[int, list[str]] = {}
    for r in ClueStore.load(kb_path):
        pool.setdefault(r.length, []).append(r.answer)
    results = []
    for puzzle in benchmark:
        layout = PuzzleLayout.model_validate(puzzle["layout"])
        gold = puzzle["gold"]
        for seed in range(seeds):
            rng = random.Random(seed)
            cands: dict[str, list[Candidate]] = {}
            for e in layout.entries:
                g = gold[e.id]
                lst = []
                if rng.random() < p_gold:
                    lst.append(Candidate(word=g, score=rng.uniform(0.3, 0.8), source=CandidateSource.LLM))
                distract = [w for w in pool.get(e.length, []) if w != g]
                for w in rng.sample(distract, min(8, len(distract))):
                    lst.append(Candidate(word=w, score=rng.uniform(0.2, 0.9), source=CandidateSource.LLM))
                for _ in range(6):
                    w = "".join(rng.choice("AEIRNSTLOHMU") for _ in range(e.length))
                    lst.append(Candidate(word=w, score=rng.uniform(0.2, 0.9), source=CandidateSource.LLM))
                cands[e.id] = lst
            present = sum(1 for e in layout.entries if any(c.word == gold[e.id] for c in cands[e.id]))
            t0 = time.perf_counter()
            res = solve(layout, cands)
            sol = to_solution(layout, res.assignment, res.grid, cands)
            m = score_solution(layout, {e.entry_id: e.answer for e in sol.entries}, gold)
            results.append(
                {"gold_in_candidates": present / len(gold), "seconds": time.perf_counter() - t0, **m}
            )
    agg = {k: round(statistics.mean(r[k] for r in results), 4) for k in results[0]}
    return {"runs": len(results), "p_gold": p_gold, "distractors_per_entry": 14, "mean": agg}


def render_markdown(e2e: list[dict], robust: dict) -> str:
    lines = [
        "# Evaluation report",
        "",
        f"Prompt versions: `{json.dumps(available())}`",
        "",
        "## End-to-end",
        "",
    ]
    lines.append(
        "| puzzle | provider | model | agent | entry acc | letter acc | fill rate | precision | LLM calls | tokens in/out | cost $ | latency s |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in e2e:
        lines.append(
            f"| {r['puzzle']} | {r['provider']} | {r['model']} | {r['use_agent']} | {r['entry_accuracy']:.1%} | "
            f"{r['letter_accuracy']:.1%} | {r['fill_rate']:.1%} | {r['precision_filled']:.1%} | {r['llm_calls']} | "
            f"{r['input_tokens']}/{r['output_tokens']} | {r['cost_usd']:.4f} | {r['latency_s']} |"
        )
    m = robust["mean"]
    lines += [
        "",
        "## Solver robustness (no LLM)",
        "",
        f"{robust['runs']} runs; gold answer offered with p={robust['p_gold']}, mixed with "
        f"{robust['distractors_per_entry']} distractors per entry (real words + noise) with random confidences.",
        "",
        "| gold in candidates | entry acc | letter acc | precision of filled | solve time s |",
        "|---|---|---|---|---|",
        f"| {m['gold_in_candidates']:.1%} | {m['entry_accuracy']:.1%} | {m['letter_accuracy']:.1%} | "
        f"{m['precision_filled']:.1%} | {m['seconds']:.2f} |",
        "",
    ]
    return "\n".join(lines)


def run(benchmark_dir: Path, report_dir: Path, use_agent: bool = True, holdout: bool = True) -> dict:
    settings = get_settings()
    bench = load_benchmark(benchmark_dir)
    e2e = end_to_end(bench, settings, use_agent, holdout)
    robust = solver_robustness(bench, settings.knowledge_base_path)
    report = {"end_to_end": e2e, "solver_robustness": robust, "prompts": available()}
    report_dir.mkdir(parents=True, exist_ok=True)
    tag = settings.llm_provider
    (report_dir / f"report_{tag}.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    md = render_markdown(e2e, robust)
    (report_dir / f"report_{tag}.md").write_text(md, encoding="utf-8")
    print(md)
    return report


if __name__ == "__main__":
    run(ROOT / "data" / "benchmark", ROOT / "eval" / "reports")
