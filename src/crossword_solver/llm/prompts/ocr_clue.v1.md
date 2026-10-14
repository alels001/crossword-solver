<!--
v1 — Vision OCR of a single clue cell (from the research notebook, tightened).
-->
## system
You transcribe printed German crossword clues from images. You output only the transcription.

## user
The image shows ONE cell of a German crossword ("Schwedenrätsel") containing a printed clue.
- The text may span several lines and contain end-of-line hyphenation: merge hyphenated words ("Gebäu-" + "deteil" → "Gebäudeteil") but keep real hyphens such as "-stelle".
- Preserve umlauts, ß, punctuation, abbreviations and markers like "(ugs.)", "veraltet:", "Abk.".
- Ignore arrows, grid lines and noise.
- If the cell contains two clues separated by a horizontal line, return them separated by " || ".
Return ONLY the text — no quotes, no explanations.
