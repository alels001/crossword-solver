"""Retrievers over the clue knowledge base.

* :class:`TfidfRetriever` — character n-gram TF-IDF. Runs offline, needs no model
  download and is robust to OCR noise ("Gebaudeteil" still matches "Gebäudeteil").
* :class:`EmbeddingRetriever` — dense semantic search with any :class:`Embedder`
  (OpenAI embeddings by default); vectors are cached on disk per KB fingerprint.
* :class:`HybridRetriever` — reciprocal-rank fusion of several retrievers, the
  usual production choice (lexical recall + semantic recall).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from .normalize import clean_clue, matches_pattern
from .store import ClueRecord, ClueStore

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Hit:
    record: ClueRecord
    score: float


class Retriever(Protocol):
    def search(
        self, query: str, k: int = 8, *, length: int | None = None, pattern: str | None = None
    ) -> list[Hit]: ...


def _filter(records: list[ClueRecord], idx: int, length: int | None, pattern: str | None) -> bool:
    r = records[idx]
    if length is not None and r.length != length:
        return False
    return not (pattern is not None and not matches_pattern(r.answer, pattern))


def _rank(
    scores: np.ndarray, records: list[ClueRecord], k: int, length: int | None, pattern: str | None
) -> list[Hit]:
    order = np.argsort(-scores)
    out: list[Hit] = []
    for i in order:
        if scores[i] <= 0:
            break
        if _filter(records, int(i), length, pattern):
            out.append(Hit(records[int(i)], min(1.0, float(scores[i]))))
            if len(out) >= k:
                break
    return out


class TfidfRetriever:
    def __init__(self, store: ClueStore) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.store = store
        self._records = list(store)
        self._vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), lowercase=True, sublinear_tf=True)
        corpus = [r.clue for r in self._records] or [""]
        self._matrix = self._vec.fit_transform(corpus)

    def search(
        self, query: str, k: int = 8, *, length: int | None = None, pattern: str | None = None
    ) -> list[Hit]:
        if not self._records:
            return []
        q = self._vec.transform([clean_clue(query)])
        scores = (self._matrix @ q.T).toarray().ravel()
        return _rank(scores, self._records, k, length, pattern)


class Embedder(Protocol):
    model: str

    def embed(self, texts: list[str]) -> np.ndarray: ...


class OpenAIEmbedder:
    def __init__(self, model: str = "text-embedding-3-small", client: object | None = None) -> None:
        if client is None:
            from openai import OpenAI

            client = OpenAI()
        self._client = client
        self.model = model

    def embed(self, texts: list[str]) -> np.ndarray:
        vecs: list[list[float]] = []
        for i in range(0, len(texts), 256):
            resp = self._client.embeddings.create(model=self.model, input=texts[i : i + 256])  # type: ignore[attr-defined]
            vecs.extend(d.embedding for d in resp.data)
        arr = np.asarray(vecs, dtype=np.float32)
        return arr / np.clip(np.linalg.norm(arr, axis=1, keepdims=True), 1e-9, None)


class EmbeddingRetriever:
    def __init__(self, store: ClueStore, embedder: Embedder, cache_dir: Path | None = None) -> None:
        self.store = store
        self._records = list(store)
        self._embedder = embedder
        cache = None
        if cache_dir is not None:
            cache_dir.mkdir(parents=True, exist_ok=True)
            safe_model = embedder.model.replace("/", "_")
            cache = cache_dir / f"kb_{store.fingerprint()}_{safe_model}.npy"
        if cache is not None and cache.exists():
            self._matrix = np.load(cache)
        else:
            self._matrix = (
                embedder.embed([r.clue for r in self._records]) if self._records else np.zeros((0, 1))
            )
            if cache is not None:
                np.save(cache, self._matrix)
                log.info("kb_index_built", extra={"records": len(self._records), "path": str(cache)})

    def search(
        self, query: str, k: int = 8, *, length: int | None = None, pattern: str | None = None
    ) -> list[Hit]:
        if not self._records:
            return []
        q = self._embedder.embed([clean_clue(query)])[0]
        scores = self._matrix @ q
        return _rank(scores, self._records, k, length, pattern)


class HybridRetriever:
    """Reciprocal Rank Fusion: rank by Σ 1 / (rrf_k + rank_i).

    The returned ``Hit.score`` is the best *similarity* any retriever assigned, so
    downstream thresholds keep the same meaning as with a single retriever.
    """

    def __init__(self, retrievers: list[Retriever], rrf_k: int = 60, depth: int = 50) -> None:
        self.retrievers = retrievers
        self.rrf_k = rrf_k
        self.depth = depth

    def search(
        self, query: str, k: int = 8, *, length: int | None = None, pattern: str | None = None
    ) -> list[Hit]:
        fused: dict[tuple[str, str], float] = {}
        best_sim: dict[tuple[str, str], float] = {}
        recs: dict[tuple[str, str], ClueRecord] = {}
        for r in self.retrievers:
            for rank, hit in enumerate(r.search(query, self.depth, length=length, pattern=pattern)):
                key = (hit.record.clue, hit.record.answer)
                recs[key] = hit.record
                fused[key] = fused.get(key, 0.0) + 1.0 / (self.rrf_k + rank + 1)
                best_sim[key] = max(best_sim.get(key, 0.0), hit.score)
        order = sorted(fused, key=lambda key: -fused[key])[:k]
        return [Hit(recs[key], best_sim[key]) for key in order]


def build_retriever(
    store: ClueStore, kind: str = "tfidf", cache_dir: Path | None = None, embedding_model: str | None = None
) -> Retriever:
    if kind == "tfidf":
        return TfidfRetriever(store)
    emb = EmbeddingRetriever(store, OpenAIEmbedder(embedding_model or "text-embedding-3-small"), cache_dir)
    if kind == "embedding":
        return emb
    return HybridRetriever([TfidfRetriever(store), emb])
