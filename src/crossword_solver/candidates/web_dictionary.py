"""Candidates from the wort-suchen.de crossword dictionary (opt-in, cached, rate-limited)."""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
from pathlib import Path
from urllib.parse import quote

import requests
from bs4 import BeautifulSoup

from ..knowledge.normalize import clean_clue, matches_pattern, normalize_answer
from ..models import Candidate, CandidateSource, Entry

log = logging.getLogger(__name__)

_BLACKLIST = {"WOERTERBUCH", "WORTLISTEN", "KREUZWORTRAETSEL", "LOESUNG", "LOESUNGEN"}
_HEADERS = {"User-Agent": "crossword-solver/2.0 (research project; contact via GitHub)"}


class WebDictionarySource:
    name = "web_dictionary"

    def __init__(
        self,
        cache_dir: Path,
        delay_s: float = 0.5,
        timeout_s: float = 20.0,
        max_results: int = 30,
        session: requests.Session | None = None,
    ) -> None:
        self.delay_s = delay_s
        self.timeout_s = timeout_s
        self.max_results = max_results
        self._session = session or requests.Session()
        self._lock = threading.Lock()
        self._last = 0.0
        cache_dir.mkdir(parents=True, exist_ok=True)
        self._db = cache_dir / "web_dictionary.sqlite3"
        with sqlite3.connect(self._db) as con:
            con.execute("CREATE TABLE IF NOT EXISTS cache (k TEXT PRIMARY KEY, v TEXT)")

    @staticmethod
    def url(clue: str, length: int) -> str:
        encoded = quote(clue.lower()).replace("%20", "+")
        return f"https://www.wort-suchen.de/kreuzwortraetsel-hilfe/loesungen/{encoded}/{'_' * length}/"

    @staticmethod
    def parse(html: str, length: int) -> list[str]:
        soup = BeautifulSoup(html, "html.parser")
        seen: dict[str, None] = {}
        for a in soup.find_all("a"):
            w = normalize_answer(a.get_text(strip=True))
            if len(w) == length and w not in _BLACKLIST:
                seen.setdefault(w, None)
        return list(seen)

    def _fetch(self, clue: str, length: int) -> list[str]:
        key = f"{clue.lower()}|{length}"
        with sqlite3.connect(self._db) as con:
            row = con.execute("SELECT v FROM cache WHERE k = ?", (key,)).fetchone()
        if row:
            return list(json.loads(row[0]))
        with self._lock:  # polite: one request at a time, fixed delay
            wait = self.delay_s - (time.monotonic() - self._last)
            if wait > 0:
                time.sleep(wait)
            try:
                r = self._session.get(self.url(clue, length), headers=_HEADERS, timeout=self.timeout_s)
                r.raise_for_status()
                words = self.parse(r.text, length)
            except requests.RequestException as e:
                log.warning("web_dictionary_failed", extra={"clue": clue, "error": str(e)})
                return []
            finally:
                self._last = time.monotonic()
        with sqlite3.connect(self._db) as con:
            con.execute("INSERT OR REPLACE INTO cache VALUES (?, ?)", (key, json.dumps(words)))
        return words

    def generate(self, entry: Entry, pattern: str) -> list[Candidate]:
        words = [w for w in self._fetch(clean_clue(entry.clue), entry.length) if matches_pattern(w, pattern)]
        out = []
        for rank, w in enumerate(words[: self.max_results]):
            out.append(
                Candidate(
                    word=w,
                    score=round(max(0.3, 0.75 - 0.03 * rank), 4),
                    source=CandidateSource.WEB_DICTIONARY,
                    evidence=f"wort-suchen.de rank {rank + 1}",
                )
            )
        return out
