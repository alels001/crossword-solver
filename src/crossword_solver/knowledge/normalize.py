"""Text normalisation for German crossword answers and OCR'd clues."""

from __future__ import annotations

import re
import unicodedata

_UMLAUTS = {"Ä": "AE", "Ö": "OE", "Ü": "UE", "ß": "SS", "ẞ": "SS"}
_OCR_FIXES = {
    "|": "",
    "Uber": "Über",
    "uber": "über",
    "Stré": "Strö",
}


def normalize_answer(word: str) -> str:
    """Crossword spelling: upper-case, umlauts expanded (Ä→AE …), letters only."""
    w = word.strip().upper()
    for k, v in _UMLAUTS.items():
        w = w.replace(k, v)
    w = unicodedata.normalize("NFKD", w)
    w = "".join(ch for ch in w if not unicodedata.combining(ch))
    return re.sub(r"[^A-Z]", "", w)


def clean_clue(text: str | None) -> str:
    """Clean OCR output: join hyphenated line breaks, collapse whitespace, fix common confusions."""
    if not text:
        return ""
    t = str(text).replace("\r", "\n")
    t = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", t)  # "Gebäu-\ndeteil" -> "Gebäudeteil"
    t = t.replace("\n", " ")
    t = re.sub(r"(\w)-\s+([a-zäöüß])", r"\1\2", t)  # "Gebäu- deteil" (already flattened)
    for k, v in _OCR_FIXES.items():
        t = t.replace(k, v)
    t = re.sub(r"\s{2,}", " ", t)
    return t.strip().strip('"').strip()


def normalize_clue_key(text: str) -> str:
    """Key for exact-duplicate detection (case/punctuation-insensitive)."""
    t = clean_clue(text).lower()
    t = re.sub(r"[^\wäöüß ]+", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def matches_pattern(word: str, pattern: str) -> bool:
    """``pattern`` uses '_' (or '?') for unknown letters, e.g. ``"A_TNA"``."""
    if len(word) != len(pattern):
        return False
    return all(p in "_?" or p == w for w, p in zip(word, pattern))


def is_valid_answer(word: str, length: int, pattern: str | None = None) -> bool:
    if len(word) != length or not word.isalpha() or not word.isascii() or not word.isupper():
        return False
    return pattern is None or matches_pattern(word, pattern)
