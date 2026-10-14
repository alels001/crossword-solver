# Prompt engineering

## Versioned templates

Prompts live next to the code in `src/crossword_solver/llm/prompts/` as `<name>.v<N>.md` with a `## system` and a
`## user` section. Templates use `$placeholders` (`string.Template`), so clue text containing `{}` or `$` can never
break rendering. A new version is a new file — the old one stays for comparison, and the evaluation report records
which versions produced which numbers.

| Prompt | Purpose |
|---|---|
| `candidates.v1` | The original notebook prompt (bare JSON array, no retrieval). Kept as a baseline. |
| `candidates.v2` | Production prompt: rules in the system turn, per-clue data in the user turn, retrieved few-shots, crossing pattern, JSON object with confidences. |
| `ocr_clue.v1` | Vision transcription of one clue cell (hyphenation, umlauts, two-clue cells). |
| `agent.v1` | System prompt for the tool-using agent. |
| `assistant_chat.v1` | German chat assistant with retrieved context. |

## What changed from v1 to v2 and why

1. **Stable system prompt, variable user turn.** The rules no longer change per clue, which makes the prefix
   cacheable by providers and keeps the per-call token count low.
2. **Grounding with retrieval.** Up to eight similar clues from the knowledge base are shown as evidence ("use when
   they fit, ignore otherwise"). This replaces the long list of hand-written examples in v1.
3. **Crossing letters.** The refine loop passes patterns like `A_TNA`, turning an open question into a constrained one.
4. **Calibrated output.** Each candidate carries a confidence; the solver uses it as the value of placing the word.
5. **Shorter.** v1 repeated several rules in capital letters; v2 states each rule once.

## Output handling

Model output is treated as untrusted input:

* `chat_structured` parses the first JSON object/array (tolerating code fences), validates it with a pydantic schema,
  and on failure sends **one** repair turn containing the validation error and the schema;
* every word is normalised (Ä→AE …) and re-checked for length, alphabet and pattern — invalid words are dropped
  silently, never "fixed";
* an LLM failure in one source never fails the solve; the pipeline continues with the other sources.

## Prompt-injection hardening

Clue text comes from OCR of a user-supplied photo, so it is untrusted:

* it is wrapped in `<clue>…</clue>` (chat: `<kontext>…</kontext>`) and the system prompt states that tagged text is
  data, never instructions;
* the agent cannot act outside the board: its only side effects are `place_answer` / `remove_answer`, both validated in
  code; there is no tool that reaches the network or the file system (the dictionary lookup takes an entry id, not a URL);
* the AI code-review tool applies the same rule to diffs.

## Evaluating a prompt change

1. Add `candidates.v3.md`.
2. `CWS_LLM_PROVIDER=openai python -m eval.run_eval` (the newest version is used by default).
3. Compare `eval/reports/report_openai.md` with the previous run: accuracy, precision of filled entries, tokens, cost.
4. Commit the prompt together with the report, so the review shows the effect of the change.
