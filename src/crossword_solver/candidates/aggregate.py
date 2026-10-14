"""Merge candidates from several sources into one ranked list per entry."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor

from ..llm.base import BudgetExceededError
from ..models import Candidate, CandidateSource, Entry
from .base import CandidateGenerator

log = logging.getLogger(__name__)

# How much we trust each source relative to its own confidence.
DEFAULT_SOURCE_WEIGHTS: dict[CandidateSource, float] = {
    CandidateSource.KNOWLEDGE_BASE: 1.0,
    CandidateSource.WEB_DICTIONARY: 0.9,
    CandidateSource.LLM: 0.85,
    CandidateSource.AGENT: 0.85,
    CandidateSource.USER: 1.0,
}


def merge(
    candidates: list[Candidate], weights: dict[CandidateSource, float] | None = None
) -> list[Candidate]:
    """Noisy-OR fusion: independent sources agreeing on a word raise its score."""
    w = weights or DEFAULT_SOURCE_WEIGHTS
    miss: dict[str, float] = {}
    best: dict[str, Candidate] = {}
    for c in candidates:
        p = min(1.0, c.score * w.get(c.source, 0.8))
        miss[c.word] = miss.get(c.word, 1.0) * (1.0 - p)
        if c.word not in best or c.score * w.get(c.source, 0.8) > best[c.word].score * w.get(
            best[c.word].source, 0.8
        ):
            best[c.word] = c
    merged = []
    for word, m in miss.items():
        b = best[word]
        sources = sorted({c.source.value for c in candidates if c.word == word})
        evidence = b.evidence if len(sources) == 1 else f"{b.evidence} [+agreement: {', '.join(sources)}]"
        merged.append(Candidate(word=word, score=round(1.0 - m, 4), source=b.source, evidence=evidence))
    return sorted(merged, key=lambda c: (-c.score, c.word))


class CandidatePipeline:
    def __init__(
        self,
        sources: list[CandidateGenerator],
        *,
        max_per_entry: int = 40,
        max_workers: int = 4,
        weights: dict[CandidateSource, float] | None = None,
    ) -> None:
        self.sources = sources
        self.max_per_entry = max_per_entry
        self.max_workers = max_workers
        self.weights = weights

    def for_entry(self, entry: Entry, pattern: str | None = None) -> list[Candidate]:
        patt = pattern or "_" * entry.length
        found: list[Candidate] = []
        for src in self.sources:
            try:
                found.extend(src.generate(entry, patt))
            except BudgetExceededError:
                raise
            except Exception as e:  # one flaky source must not kill the solve
                log.warning(
                    "candidate_source_failed", extra={"source": src.name, "entry": entry.id, "error": str(e)}
                )
        return merge(found, self.weights)[: self.max_per_entry]

    def for_entries(self, entries: list[Entry]) -> dict[str, list[Candidate]]:
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            results = list(pool.map(self.for_entry, entries))
        return {e.id: r for e, r in zip(entries, results)}
