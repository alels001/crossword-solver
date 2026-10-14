"""Knowledge base, retrievers and candidate generation."""

from __future__ import annotations

import json

import numpy as np

from crossword_solver.candidates import KnowledgeBaseSource, LLMCandidateSource, merge
from crossword_solver.knowledge import (
    ClueRecord,
    ClueStore,
    EmbeddingRetriever,
    HybridRetriever,
    TfidfRetriever,
)
from crossword_solver.llm import FakeLLM, LLMResponse, ScriptedLLM
from crossword_solver.models import Candidate, CandidateSource, Direction, Entry


def _entry(clue: str, length: int) -> Entry:
    return Entry(id="E", clue=clue, start=(0, 0), direction=Direction.RIGHT, length=length)


def test_store_normalises_and_deduplicates(tmp_path) -> None:
    s = ClueStore(
        [
            ClueRecord("Vulkan bei Neapel", "Vesuv"),
            ClueRecord("Vulkan  bei Neapel", "VESUV"),
            ClueRecord("x", "Öse"),
        ]
    )
    assert len(s) == 2
    assert s[1].answer == "OESE"
    s.save(tmp_path / "kb.csv")
    assert len(ClueStore.load(tmp_path / "kb.csv")) == 2
    assert len(s.without_clues(["vulkan bei neapel!"])) == 1


def test_tfidf_is_robust_to_ocr_noise(retriever: TfidfRetriever) -> None:
    hits = retriever.search("Vu1kan bei Neape1", 3)  # OCR confused l/1
    assert hits[0].record.answer == "VESUV"
    filtered = retriever.search("Hauptstadt", 5, length=4, pattern="O__O")
    assert [h.record.answer for h in filtered] == ["OSLO"]


class _BagOfWordsEmbedder:
    """Deterministic toy embedder so the dense/hybrid path is testable offline."""

    model = "toy"

    def embed(self, texts: list[str]) -> np.ndarray:
        vocab = ["vulkan", "stadt", "fluss", "insel", "hauptstadt", "vogel"]
        m = np.array([[t.lower().count(w) for w in vocab] for t in texts], dtype=np.float32) + 1e-3
        return m / np.linalg.norm(m, axis=1, keepdims=True)


def test_embedding_and_hybrid_retrievers(store: ClueStore, tmp_path) -> None:
    dense = EmbeddingRetriever(store, _BagOfWordsEmbedder(), cache_dir=tmp_path)
    assert any(h.record.answer == "VESUV" for h in dense.search("ein Vulkan", 10, length=5))
    assert list(tmp_path.glob("kb_*_toy.npy")), "vectors cached per KB fingerprint"
    hybrid = HybridRetriever([TfidfRetriever(store), dense])
    top = hybrid.search("Vulkan bei Neapel", 3)
    assert top[0].record.answer == "VESUV"
    assert 0 < top[0].score <= 1.0  # similarity scale preserved, not RRF scale


def test_kb_source_only_trusts_near_duplicates(retriever: TfidfRetriever) -> None:
    src = KnowledgeBaseSource(retriever)
    exact = src.generate(_entry("Vulkan bei Neapel", 5), "_____")
    assert exact[0].word == "VESUV" and exact[0].score >= 0.9
    assert src.generate(_entry("Haarwuchs im Gesicht", 4), "____") == []  # "Absicht"→PLAN is lexical noise


def test_llm_source_validates_model_output(retriever: TfidfRetriever) -> None:
    reply = {
        "candidates": [
            {"word": "Ätna", "confidence": 0.4},  # normalised → AETNA
            {"word": "AETNA", "confidence": 0.9},  # duplicate after normalisation
            {"word": "ETNA", "confidence": 0.8},  # wrong length → dropped
            {"word": "VESUV", "confidence": 0.3},  # violates pattern → dropped
        ]
    }
    llm = FakeLLM(handler=lambda m, s, t: LLMResponse(text=json.dumps(reply)))
    src = LLMCandidateSource(llm, retriever)
    out = src.generate(_entry("sizilianischer Vulkan", 5), "A____")
    assert [c.word for c in out] == ["AETNA"]
    prompt = llm.calls[0]["messages"][0].text
    assert "<clue>sizilianischer Vulkan</clue>" in prompt and "VESUV" in prompt  # RAG few-shots injected


def test_llm_source_survives_garbage() -> None:
    src = LLMCandidateSource(ScriptedLLM(["nope", "still nope"]))
    assert src.generate(_entry("x", 3), "___") == []


def test_merge_noisy_or_rewards_agreement() -> None:
    merged = merge(
        [
            Candidate(word="AETNA", score=0.6, source=CandidateSource.LLM),
            Candidate(word="AETNA", score=0.6, source=CandidateSource.WEB_DICTIONARY),
            Candidate(word="ETNAS", score=0.7, source=CandidateSource.LLM),
        ]
    )
    assert merged[0].word == "AETNA"
    assert merged[0].score > 0.7 and "agreement" in (merged[0].evidence or "")
