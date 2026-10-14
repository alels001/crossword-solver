from __future__ import annotations

from typing import Protocol

from ..models import Candidate, Entry


class CandidateGenerator(Protocol):
    """Produces scored answer candidates for one entry given the currently known letters."""

    name: str

    def generate(self, entry: Entry, pattern: str) -> list[Candidate]: ...
