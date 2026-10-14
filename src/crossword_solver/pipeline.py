"""End-to-end orchestration: photo/layout → candidates (RAG + LLM) → CSP → agent → solution.

``CrosswordSolver`` is the single entry point used by the CLI, the HTTP API, the
MCP server and the evaluation harness, so every interface behaves identically.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from .candidates import CandidatePipeline, KnowledgeBaseSource, LLMCandidateSource, WebDictionarySource, merge
from .candidates.base import CandidateGenerator
from .config import Settings, get_settings
from .knowledge import ClueStore, Retriever, build_retriever, clean_clue, normalize_answer
from .llm import BudgetExceededError, LLMClient, Message, TrackedLLM, UsageTracker, build_llm
from .llm.prompts import load_prompt
from .models import Candidate, CandidateSource, Direction, Entry, PuzzleLayout, Solution, Usage
from .solver import Board, SolverAgent, solve, to_solution

log = logging.getLogger(__name__)


@dataclass
class SolveReport:
    solution: Solution
    layout: PuzzleLayout
    candidates: dict[str, list[Candidate]]
    agent_transcript: list[dict[str, Any]] = field(default_factory=list)
    timings_ms: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "solution": self.solution.model_dump(mode="json"),
            "layout": self.layout.model_dump(mode="json"),
            "candidates": {
                k: [c.model_dump(mode="json") for c in v[:10]] for k, v in self.candidates.items()
            },
            "agent_transcript": self.agent_transcript,
            "timings_ms": self.timings_ms,
        }


class _Timer:
    def __init__(self, sink: dict[str, float], name: str) -> None:
        self.sink, self.name = sink, name

    def __enter__(self) -> None:
        self.t0 = time.perf_counter()

    def __exit__(self, *exc: object) -> None:
        self.sink[self.name] = round((time.perf_counter() - self.t0) * 1000, 1)


class CrosswordSolver:
    def __init__(
        self,
        settings: Settings | None = None,
        *,
        store: ClueStore | None = None,
        retriever: Retriever | None = None,
        llm: LLMClient | None = None,
        vision_llm: LLMClient | None = None,
    ) -> None:
        self.settings = s = settings or get_settings()
        self.store = store if store is not None else ClueStore.load(s.knowledge_base_path)
        self.retriever = retriever or build_retriever(
            self.store, s.retriever, cache_dir=s.cache_dir, embedding_model=s.openai_embedding_model
        )
        self._llm = llm or build_llm(s)
        self._vision_llm = vision_llm
        self.dictionary = (
            WebDictionarySource(s.cache_dir, delay_s=s.web_dictionary_delay_s)
            if s.enable_web_dictionary
            else None
        )

    @property
    def llm_enabled(self) -> bool:
        return self.settings.llm_provider != "fake" or not self._llm.model.startswith("fake")

    # ---- building blocks ---------------------------------------------------------
    def _sources(self, llm: LLMClient) -> list[CandidateGenerator]:
        sources: list[CandidateGenerator] = [KnowledgeBaseSource(self.retriever)]
        if self.dictionary is not None:
            sources.append(self.dictionary)
        sources.append(
            LLMCandidateSource(llm, self.retriever, max_candidates=15, n_examples=self.settings.rag_top_k)
        )
        return sources

    def suggest(
        self, clue: str, length: int, pattern: str | None = None, tracker: UsageTracker | None = None
    ) -> list[Candidate]:
        """Ranked candidates for a single clue — used by the API, MCP tools and the chat UI."""
        tracker = tracker or UsageTracker(self.settings.max_cost_usd_per_puzzle)
        llm = TrackedLLM(self._llm, tracker, component="suggest")
        patt = (pattern or "_" * length).upper().replace("?", "_").replace(".", "_")
        if len(patt) != length:
            raise ValueError("pattern length must equal answer length")
        entry = Entry(id="Q", clue=clean_clue(clue), start=(0, 0), direction=Direction.RIGHT, length=length)
        return CandidatePipeline(
            self._sources(llm), max_per_entry=self.settings.max_candidates_per_entry
        ).for_entry(entry, patt)

    # ---- full solve --------------------------------------------------------------
    def solve_layout(
        self,
        layout: PuzzleLayout,
        *,
        known_answers: dict[str, str] | None = None,
        use_agent: bool = True,
        refine_rounds: int = 2,
        on_progress: Callable[[str], object] | None = None,
    ) -> SolveReport:
        s = self.settings
        tracker = UsageTracker(s.max_cost_usd_per_puzzle)
        llm = TrackedLLM(self._llm, tracker, component="solve")
        timings: dict[str, float] = {}
        notify = on_progress or (lambda _m: None)
        stats: dict[str, float | int | str] = {}

        pipe = CandidatePipeline(self._sources(llm), max_per_entry=s.max_candidates_per_entry)
        locked = {
            eid: Candidate(word=normalize_answer(w), score=1.0, source=CandidateSource.USER, evidence="given")
            for eid, w in (known_answers or {}).items()
        }
        open_entries = [e for e in layout.entries if e.id not in locked]

        notify("Generating candidates (knowledge base + LLM)…")
        with _Timer(timings, "candidates"):
            try:
                candidates = pipe.for_entries(open_entries)
            except BudgetExceededError:
                candidates, stats["budget_exhausted"] = {}, 1

        notify("Solving constraints…")
        with _Timer(timings, "csp"):
            result = solve(layout, candidates, locked=locked, max_nodes=s.csp_max_nodes)
        stats["csp_nodes"] = result.nodes

        # Refinement: entries left open but now partially constrained get a second,
        # pattern-aware round of candidate generation, then we re-solve.
        with _Timer(timings, "refine"):
            for round_ in range(refine_rounds):
                board = Board(layout)
                for k, v in result.assignment.items():
                    if v is not None:
                        board.answers[k] = v
                grid = board.letters()
                todo = [
                    (e, board.pattern(e.id, grid))
                    for e in open_entries
                    if result.assignment.get(e.id) is None
                    and 0 < board.pattern(e.id, grid).count("_") < e.length
                ]
                if not todo:
                    break
                notify(f"Refining {len(todo)} open entries with crossing letters (round {round_ + 1})…")
                added = 0
                try:
                    for e, patt in todo:
                        new = pipe.for_entry(e, patt)
                        before = {c.word for c in candidates.get(e.id, [])}
                        candidates[e.id] = merge(candidates.get(e.id, []) + new)[: s.max_candidates_per_entry]
                        added += len({c.word for c in new} - before)
                except BudgetExceededError:
                    stats["budget_exhausted"] = 1
                if added == 0:
                    break
                result = solve(layout, candidates, locked=locked, max_nodes=s.csp_max_nodes)

        assignment = dict(result.assignment)
        transcript: list[dict[str, Any]] = []
        board = Board(layout)
        for k, v in assignment.items():
            if v is not None:
                board.answers[k] = v

        if use_agent and self.llm_enabled:
            agent = SolverAgent(
                llm,
                layout,
                board,
                candidates,
                retriever=self.retriever,
                dictionary=self.dictionary,
                max_steps=s.agent_max_steps,
            )
            if agent.open_entries():
                notify("Agent completing remaining entries…")
                with _Timer(timings, "agent"):
                    try:
                        report = agent.run()
                        transcript = report.transcript
                        stats["agent_steps"] = report.steps
                        stats["agent_placed"] = len(report.placed)
                    except BudgetExceededError:
                        stats["budget_exhausted"] = 1
                assignment = {e.id: board.answers.get(e.id) for e in layout.entries}

        solution = to_solution(layout, assignment, board.letters(), candidates)
        solution.usage = Usage(**{k: v for k, v in tracker.snapshot().items()})  # type: ignore[arg-type]
        solved = sum(1 for er in solution.entries if "_" not in er.answer)
        stats.update({"entries": len(layout.entries), "entries_filled": solved})
        solution.stats = stats
        return SolveReport(solution, layout, candidates, transcript, timings)

    def solve_image(self, image_bytes: bytes, **kwargs: Any) -> SolveReport:
        from .ocr import ClueOCR
        from .vision import analyse_image, decode_image

        s = self.settings
        timings: dict[str, float] = {}
        with _Timer(timings, "vision"):
            vis = analyse_image(
                decode_image(image_bytes), rows=s.grid_rows, cols=s.grid_cols, mask_region=s.mask_region
            )
        vision_llm = self._vision_llm or build_llm(s, vision=True)
        ocr = ClueOCR(vision_llm)
        with _Timer(timings, "ocr"):
            texts = ocr.read_many([vis.clue_crop(e.clue_cell) for e in vis.layout.entries if e.clue_cell])
        for e, t in zip(vis.layout.entries, texts):
            e.clue = t
        report = self.solve_layout(vis.layout, **kwargs)
        report.timings_ms = {**timings, **report.timings_ms}
        return report

    # ---- chat assistant (RAG) -------------------------------------------------------
    def chat(self, message: str, history: list[dict[str, str]] | None = None) -> str:
        hits = self.retriever.search(message, self.settings.rag_top_k)
        context = (
            "\n".join(f"- {h.record.clue} → {h.record.answer} ({h.record.length})" for h in hits)
            or "(keine Treffer)"
        )
        system, user = load_prompt("assistant_chat").render(context=context, message=message)
        msgs = [
            Message(role="user" if m["role"] == "user" else "assistant", text=m["content"])
            for m in (history or [])
            if m["role"] in ("user", "assistant")
        ]
        msgs.append(Message.user(user))
        tracker = UsageTracker(self.settings.max_cost_usd_per_puzzle)
        resp = TrackedLLM(self._llm, tracker, component="chat").chat(msgs, system=system)
        if resp.text:
            return resp.text
        # Offline fallback: answer from retrieval only.
        if not hits:
            return "Ich habe dazu nichts in der Wissensdatenbank gefunden. (Offline-Modus: kein LLM konfiguriert.)"
        lines = [f"- **{h.record.answer}** ({h.record.length}) — „{h.record.clue}“" for h in hits[:5]]
        return "Passende Einträge aus der Wissensdatenbank (Offline-Modus):\n" + "\n".join(lines)
