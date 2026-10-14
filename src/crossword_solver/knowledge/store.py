"""Clue/answer knowledge base used for retrieval-augmented generation."""

from __future__ import annotations

import csv
import hashlib
import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

from .normalize import clean_clue, normalize_answer, normalize_clue_key


@dataclass(frozen=True)
class ClueRecord:
    clue: str
    answer: str
    source: str = "curated"

    @property
    def length(self) -> int:
        return len(self.answer)


class ClueStore:
    """In-memory store; loaded from CSV (clue,answer[,source]) or JSONL."""

    def __init__(self, records: Iterable[ClueRecord] = ()) -> None:
        self._records: list[ClueRecord] = []
        self._seen: set[tuple[str, str]] = set()
        for r in records:
            self.add(r)

    # ---- mutation -----------------------------------------------------------
    def add(self, record: ClueRecord) -> bool:
        clue = clean_clue(record.clue)
        answer = normalize_answer(record.answer)
        if not clue or not answer:
            return False
        key = (normalize_clue_key(clue), answer)
        if key in self._seen:
            return False
        self._seen.add(key)
        self._records.append(ClueRecord(clue=clue, answer=answer, source=record.source))
        return True

    def without_clues(self, clues: Iterable[str]) -> ClueStore:
        """Copy with exact clue duplicates removed — used to hold out eval puzzles."""
        banned = {normalize_clue_key(c) for c in clues}
        return ClueStore(r for r in self._records if normalize_clue_key(r.clue) not in banned)

    # ---- access -------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._records)

    def __iter__(self) -> Iterator[ClueRecord]:
        return iter(self._records)

    def __getitem__(self, i: int) -> ClueRecord:
        return self._records[i]

    @property
    def records(self) -> list[ClueRecord]:
        return self._records

    def fingerprint(self) -> str:
        h = hashlib.sha256()
        for r in self._records:
            h.update(f"{r.clue}\t{r.answer}\n".encode())
        return h.hexdigest()[:16]

    # ---- io -----------------------------------------------------------------
    @classmethod
    def load(cls, path: str | Path) -> ClueStore:
        p = Path(path)
        if not p.exists():
            return cls()
        if p.suffix == ".jsonl":
            recs = []
            with p.open(encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        d = json.loads(line)
                        recs.append(ClueRecord(d["clue"], d["answer"], d.get("source", p.stem)))
            return cls(recs)
        with p.open(encoding="utf-8", newline="") as f:
            return cls(
                ClueRecord(r["clue"], r["answer"], r.get("source") or p.stem) for r in csv.DictReader(f)
            )

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["clue", "answer", "source"])
            for r in self._records:
                w.writerow([r.clue, r.answer, r.source])
