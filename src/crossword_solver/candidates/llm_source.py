"""RAG-grounded LLM candidate generation with schema-validated output."""

from __future__ import annotations

import logging

from pydantic import BaseModel, Field

from ..knowledge.normalize import clean_clue, is_valid_answer, normalize_answer
from ..knowledge.retrievers import Retriever
from ..llm.base import LLMClient, LLMError, Message
from ..llm.prompts import load_prompt
from ..llm.structured import chat_structured
from ..models import Candidate, CandidateSource, Entry

log = logging.getLogger(__name__)


class _LLMCandidate(BaseModel):
    word: str
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class _LLMCandidates(BaseModel):
    candidates: list[_LLMCandidate | str] = Field(default_factory=list)


class LLMCandidateSource:
    name = "llm"

    def __init__(
        self,
        llm: LLMClient,
        retriever: Retriever | None = None,
        *,
        prompt_version: int | None = None,
        max_candidates: int = 15,
        n_examples: int = 8,
    ) -> None:
        self.llm = llm
        self.retriever = retriever
        self.prompt = load_prompt("candidates", prompt_version)
        self.max_candidates = max_candidates
        self.n_examples = n_examples

    def _examples(self, clue: str) -> str:
        if self.retriever is None:
            return "(none)"
        hits = self.retriever.search(clue, self.n_examples)
        if not hits:
            return "(none)"
        return "\n".join(f"- {h.record.clue} → {h.record.answer} ({h.record.length})" for h in hits)

    def generate(self, entry: Entry, pattern: str) -> list[Candidate]:
        clue = clean_clue(entry.clue)
        system, user = self.prompt.render(
            clue=clue,
            length=entry.length,
            pattern=pattern,
            examples=self._examples(clue),
            max_candidates=self.max_candidates,
            want_n=self.max_candidates,
        )
        try:
            parsed = chat_structured(
                self.llm,
                _LLMCandidates,
                [Message.user(user)],
                system=system or None,
                coerce_list_key="candidates",
            )
        except LLMError as e:
            log.warning("llm_candidates_failed", extra={"entry": entry.id, "error": str(e)[:200]})
            return []

        out: dict[str, Candidate] = {}
        n = len(parsed.candidates)
        for rank, item in enumerate(parsed.candidates):
            if isinstance(item, str):  # legacy v1 prompt: bare list, rank-based confidence
                word, conf = item, max(0.2, 0.7 - 0.05 * rank)
            else:
                word, conf = item.word, item.confidence
            w = normalize_answer(word)
            # Output guardrail: never trust the model on length/pattern.
            if not is_valid_answer(w, entry.length, pattern) or w in out:
                continue
            out[w] = Candidate(
                word=w,
                score=round(float(conf), 4),
                source=CandidateSource.LLM,
                evidence=f"{self.llm.model} via {self.prompt.id} (rank {rank + 1}/{n})",
            )
        return list(out.values())[: self.max_candidates]
