"""Command-line interface: ``crossword --help``."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from .config import REPO_ROOT, get_settings
from .observability import setup_logging

app = typer.Typer(add_completion=False, help="GenAI crossword solver: photo/JSON → solved grid.")
console = Console()


@app.callback()
def _main(verbose: bool = typer.Option(False, "--verbose", "-v", help="Debug logging.")) -> None:
    s = get_settings()
    setup_logging("DEBUG" if verbose else "WARNING", s.log_json)


@app.command()
def suggest(
    clue: str,
    length: int,
    pattern: str = typer.Option(None, "--pattern", "-p", help="Known letters, '_' for unknown."),
) -> None:
    """Suggest answers for a single clue."""
    from .pipeline import CrosswordSolver

    cands = CrosswordSolver().suggest(clue, length, pattern)
    t = Table("answer", "score", "source", "evidence")
    for c in cands[:15]:
        t.add_row(c.word, f"{c.score:.2f}", c.source.value, (c.evidence or "")[:70])
    console.print(t if cands else "[yellow]no candidates[/]")


@app.command()
def solve(
    path: Path = typer.Argument(..., exists=True, help="Puzzle photo (.png/.jpg) or layout JSON."),
    out: Path = typer.Option(None, "--out", "-o", help="Write the full report as JSON."),
    no_agent: bool = typer.Option(False, "--no-agent", help="Skip the LLM agent stage."),
) -> None:
    """Solve a crossword from a photo or a JSON layout."""
    from .models import PuzzleLayout
    from .pipeline import CrosswordSolver

    solver = CrosswordSolver()
    with console.status("Solving…") as status:
        progress = status.update
        if path.suffix.lower() == ".json":
            data = json.loads(path.read_text(encoding="utf-8"))
            layout = PuzzleLayout.model_validate(data.get("layout", data))
            report = solver.solve_layout(layout, use_agent=not no_agent, on_progress=progress)
        else:
            report = solver.solve_image(path.read_bytes(), use_agent=not no_agent, on_progress=progress)
    sol = report.solution
    console.print(sol.as_text())
    t = Table("id", "clue", "answer", "conf", "source")
    for e in sol.entries:
        t.add_row(
            e.entry_id, e.clue[:40], e.answer, f"{e.confidence:.2f}", e.source.value if e.source else ""
        )
    console.print(t)
    console.print(f"usage: {sol.usage.model_dump()}  timings_ms: {report.timings_ms}")
    if out:
        out.write_text(json.dumps(report.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        console.print(f"report → {out}")


@app.command()
def evaluate(
    benchmark_dir: Path = typer.Option(REPO_ROOT / "data" / "benchmark", help="Folder with puzzle_*.json."),
    report_dir: Path = typer.Option(REPO_ROOT / "eval" / "reports", help="Where to write the report."),
    no_agent: bool = typer.Option(False, "--no-agent"),
    holdout: bool = typer.Option(True, help="Remove benchmark clues from the knowledge base (no leakage)."),
) -> None:
    """Run the evaluation harness (accuracy, cost, latency) on the benchmark puzzles."""
    sys.path.insert(0, str(REPO_ROOT))
    from eval.run_eval import run

    run(benchmark_dir, report_dir, use_agent=not no_agent, holdout=holdout)


@app.command()
def serve(host: str = "0.0.0.0", port: int = 8000, reload: bool = False) -> None:
    """Start the HTTP API (FastAPI + Uvicorn)."""
    import uvicorn

    uvicorn.run("crossword_solver.serving.api:app", host=host, port=port, reload=reload)


@app.command()
def ui(port: int = 8501) -> None:
    """Start the Streamlit chat/solver UI."""
    subprocess.run(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(REPO_ROOT / "apps" / "streamlit_app.py"),
            "--server.port",
            str(port),
        ],
        check=False,
    )


@app.command()
def mcp() -> None:
    """Run the MCP server on stdio (for IDE coding agents and AI assistants)."""
    from .serving.mcp_server import main

    main()


@app.command("kb-stats")
def kb_stats() -> None:
    """Show knowledge-base statistics."""
    from .knowledge import ClueStore

    s = get_settings()
    store = ClueStore.load(s.knowledge_base_path)
    lengths: dict[int, int] = {}
    for r in store:
        lengths[r.length] = lengths.get(r.length, 0) + 1
    console.print(f"{len(store)} records from {s.knowledge_base_path} (fingerprint {store.fingerprint()})")
    console.print({k: lengths[k] for k in sorted(lengths)})


if __name__ == "__main__":
    app()
