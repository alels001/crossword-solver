"""Versioned prompt templates.

Each prompt lives in ``<name>.v<N>.md`` with ``## system`` and ``## user``
sections. Templates use ``$variable`` placeholders (``string.Template``) so clue
text containing braces can never break formatting. Bumping the version creates a
new file; the evaluation report records which version produced which score, so
prompt changes are reviewed like code changes.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from string import Template

PROMPT_DIR = Path(__file__).resolve().parent
_SECTION = re.compile(r"^## (system|user)\s*$", re.MULTILINE)


@dataclass(frozen=True)
class PromptTemplate:
    name: str
    version: int
    system: str
    user: str

    @property
    def id(self) -> str:
        return f"{self.name}.v{self.version}"

    def render(self, **variables: object) -> tuple[str, str]:
        vals = {k: str(v) for k, v in variables.items()}
        return Template(self.system).substitute(vals), Template(self.user).substitute(vals)


def available() -> dict[str, list[int]]:
    out: dict[str, list[int]] = {}
    for p in PROMPT_DIR.glob("*.v*.md"):
        name, ver = p.stem.rsplit(".v", 1)
        out.setdefault(name, []).append(int(ver))
    return {k: sorted(v) for k, v in out.items()}


@cache
def load_prompt(name: str, version: int | None = None) -> PromptTemplate:
    versions = available().get(name)
    if not versions:
        raise FileNotFoundError(f"No prompt named {name!r} in {PROMPT_DIR}")
    v = version if version is not None else versions[-1]
    text = (PROMPT_DIR / f"{name}.v{v}.md").read_text(encoding="utf-8")
    parts = _SECTION.split(text)
    sections = {parts[i]: parts[i + 1].strip() for i in range(1, len(parts) - 1, 2)}
    return PromptTemplate(
        name=name, version=v, system=sections.get("system", ""), user=sections.get("user", "")
    )
