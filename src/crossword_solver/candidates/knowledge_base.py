"""Candidates taken directly from answers of near-duplicate clues in the knowledge base.

Lexical similarity is a poor proxy for meaning ("Absicht" ~ "Gesicht"), so only
near-duplicates become direct candidates. Everything else the retriever finds is
still used — as grounding evidence in the LLM prompt (see ``LLMCandidateSource``).
"""

from __future__ import annotations

from ..knowledge.normalize import normalize_clue_key
from ..knowledge.retrievers import Retriever
from ..models import Candidate, CandidateSource, Entry


class KnowledgeBaseSource:
    name = "knowledge_base"

    def __init__(self, retriever: Retriever, k: int = 10, min_similarity: float = 0.75) -> None:
        self.retriever = retriever
        self.k = k
        self.min_similarity = min_similarity

    def generate(self, entry: Entry, pattern: str) -> list[Candidate]:
        key = normalize_clue_key(entry.clue)
        out: dict[str, Candidate] = {}
        for hit in self.retriever.search(entry.clue, self.k, length=entry.length, pattern=pattern):
            exact = normalize_clue_key(hit.record.clue) == key
            if not exact and hit.score < self.min_similarity:
                continue
            score = 0.95 if exact else min(0.9, 0.45 + 0.45 * hit.score)
            word = hit.record.answer
            if word not in out or out[word].score < score:
                out[word] = Candidate(
                    word=word,
                    score=round(score, 4),
                    source=CandidateSource.KNOWLEDGE_BASE,
                    evidence=f'KB: "{hit.record.clue}" → {word} (sim={hit.score:.2f})',
                )
        return sorted(out.values(), key=lambda c: -c.score)
