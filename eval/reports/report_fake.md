# Evaluation report

Prompt versions: `{"ocr_clue": [1], "candidates": [1, 2], "agent": [1], "assistant_chat": [1]}`

## End-to-end

| puzzle | provider | model | agent | entry acc | letter acc | fill rate | precision | LLM calls | tokens in/out | cost $ | latency s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| puzzle_01 | fake | fake-llm | True | 0.0% | 0.0% | 0.0% | 0.0% | 26 | 11312/104 | 0.0000 | 0.06 |

## Solver robustness (no LLM)

10 runs; gold answer offered with p=0.85, mixed with 14 distractors per entry (real words + noise) with random confidences.

| gold in candidates | entry acc | letter acc | precision of filled | solve time s |
|---|---|---|---|---|
| 82.3% | 79.6% | 89.5% | 90.5% | 1.94 |
