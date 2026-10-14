# ADR 0003 — Retrieval: lexical by default, dense and hybrid optional

**Status:** accepted

## Context
Clues are short (2–6 words) and arrive with OCR noise. The knowledge base is small today and must work offline in CI.

## Decision
Default retriever: character n-gram TF-IDF (2–4-grams, word-boundary aware). It needs no model download, is robust
to character-level OCR errors and is fast. Dense retrieval (OpenAI embeddings, vectors cached per KB fingerprint) and a
hybrid with reciprocal-rank fusion are available via `CWS_RETRIEVER`. Retrieval feeds the LLM as few-shot evidence;
only near-duplicates (similarity ≥ 0.75) become direct candidates.

## Consequences
+ Zero-dependency default, deterministic tests.
+ Semantic recall available when a key is configured.
− Lexical similarity can match unrelated clues; mitigated by the high threshold for direct candidates.
