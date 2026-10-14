# ADR 0002 — Weighted CSP with open values, agent as fallback

**Status:** accepted

## Context
Candidate lists are incomplete (OCR errors, rare words). A strict CSP fails the whole grid if one entry has no
correct candidate. Letting an LLM fill the whole grid alone is expensive, slow and produces crossing conflicts.

## Decision
Solve deterministically first: a weighted CSP whose domains include *NULL* (leave open), maximising candidate scores
plus a crossing bonus. Run pattern-aware candidate generation for entries that became partially constrained, re-solve,
and only then hand the remaining open entries to a tool-using agent whose writes are validated by the same board logic.

## Consequences
+ Most of the grid is solved without agent calls (cheaper, reproducible).
+ The agent works on small, well-constrained sub-problems and cannot corrupt the grid.
− Two search mechanisms to maintain; mitigated by sharing `Board` and the candidate pipeline.
