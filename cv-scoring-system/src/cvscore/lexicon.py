"""Loading of the YAML resources and fast phrase / skill matching.

Matching works on *stemmed token sequences* (see :mod:`cvscore.textnorm`): an alias
"prowadzenie warsztatów" becomes ``("prowadz", "warsztat")`` and is found in any
line whose stems contain that sequence, whatever the inflection or diacritics.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources
from pathlib import Path

import yaml

from . import textnorm

DATA_DIR = Path(str(resources.files("cvscore") / "data"))


@lru_cache(maxsize=None)
def load_yaml(name: str) -> dict:
    """Load ``cvscore/data/<name>.yaml`` (cached). A file of the same name in
    ``$CVSCORE_DATA`` overrides the packaged one, so users can extend lists."""
    import os  # pylint: disable=import-outside-toplevel

    override = os.environ.get("CVSCORE_DATA")
    if override and (Path(override) / f"{name}.yaml").is_file():
        path = Path(override) / f"{name}.yaml"
    else:
        path = DATA_DIR / f"{name}.yaml"
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def lexicons() -> dict:
    """Word lists (lexicons.yaml)."""
    return load_yaml("lexicons")


def phrase_stems(phrase: str) -> tuple[str, ...]:
    """Stemmed token tuple of a phrase."""
    return tuple(textnorm.stems(phrase))


class PhraseSet:
    """A set of phrases matched as stem sequences inside a text."""

    def __init__(self, phrases):
        self.phrases = [p for p in phrases if p]
        self._index: dict[str, list[tuple[tuple[str, ...], str]]] = {}
        for phrase in self.phrases:
            seq = phrase_stems(phrase)
            if seq:
                self._index.setdefault(seq[0], []).append((seq, phrase))
        for lst in self._index.values():
            lst.sort(key=lambda item: -len(item[0]))

    def find(self, text_or_stems) -> list[str]:
        """Return matched phrases (with repetition) in order of appearance."""
        st = text_or_stems if isinstance(text_or_stems, list) else textnorm.stems(text_or_stems)
        found, i = [], 0
        while i < len(st):
            step = 1
            for seq, phrase in self._index.get(st[i], ()):
                if tuple(st[i:i + len(seq)]) == seq:
                    found.append(phrase)
                    step = len(seq)
                    break
            i += step
        return found

    def count(self, text) -> int:
        """Number of matches."""
        return len(self.find(text))

    def any(self, text) -> bool:
        """True if at least one phrase occurs."""
        return bool(self.find(text))

    def starts(self, text: str, window: int = 3) -> bool:
        """True if a phrase starts within the first ``window`` tokens of the text."""
        st = textnorm.stems(text)[: window + 3]
        for i in range(min(window, len(st))):
            for seq, _ in self._index.get(st[i], ()):
                if tuple(st[i:i + len(seq)]) == seq:
                    return True
        return False


@lru_cache(maxsize=None)
def phrase_set(*path: str) -> PhraseSet:
    """PhraseSet built from ``lexicons.yaml`` at the given key path."""
    node = lexicons()
    for key in path:
        node = node[key]
    return PhraseSet(node)


@dataclass(frozen=True)
class Skill:
    """A taxonomy entry."""

    name: str
    domain: str
    kind: str


@dataclass
class SkillHit:
    """A skill found in a given line of a document."""

    skill: Skill
    alias: str
    line: int


class Taxonomy:
    """Skill dictionary with stem-sequence matching (taxonomy.yaml)."""

    def __init__(self, data: dict | None = None):
        data = data if data is not None else load_yaml("taxonomy")
        self.domains: dict[str, str] = {}
        self.skills: dict[str, Skill] = {}
        self._index: dict[str, list[tuple[tuple[str, ...], Skill, str]]] = {}
        for domain, body in data.items():
            self.domains[domain] = body.get("label", domain)
            for entry in body.get("skills", []):
                skill = Skill(entry["name"], domain, entry.get("kind", "knowledge"))
                self.skills[skill.name] = skill
                for alias in entry.get("aliases", []):
                    seq = phrase_stems(str(alias))
                    if seq:
                        self._index.setdefault(seq[0], []).append((seq, skill, str(alias)))
        for lst in self._index.values():
            lst.sort(key=lambda item: -len(item[0]))

    def find_in_stems(self, st: list[str], line: int = 0) -> list[SkillHit]:
        """Find skills in a stem list (longest alias wins at each position)."""
        hits, i = [], 0
        while i < len(st):
            step = 1
            for seq, skill, alias in self._index.get(st[i], ()):
                if tuple(st[i:i + len(seq)]) == seq:
                    hits.append(SkillHit(skill, alias, line))
                    step = len(seq)
                    break
            i += step
        return hits

    def find(self, text: str) -> list[SkillHit]:
        """Find skills line by line in a text; ``SkillHit.line`` is the line index."""
        hits = []
        for idx, line in enumerate(text.split("\n")):
            if line.strip():
                hits.extend(self.find_in_stems(textnorm.stems(line), idx))
        return hits

    def names(self, text: str) -> set[str]:
        """Canonical names of the skills in a text."""
        return {h.skill.name for h in self.find(text)}

    def domain_profile(self, text: str) -> dict[str, float]:
        """Share of skill mentions per domain (sums to 1, empty if nothing found).

        Distinct skills count once per line so that a long list of one domain
        does not drown everything else; soft and office skills count half.
        """
        weights: dict[str, float] = {}
        for hit in self.find(text):
            w = 0.5 if hit.skill.kind == "soft" or hit.skill.domain == "office_general" else 1.0
            weights[hit.skill.domain] = weights.get(hit.skill.domain, 0.0) + w
        total = sum(weights.values())
        return {d: w / total for d, w in sorted(weights.items(), key=lambda kv: -kv[1])} if total else {}


@lru_cache(maxsize=1)
def taxonomy() -> Taxonomy:
    """Shared taxonomy instance."""
    return Taxonomy()


def regex_any(patterns, text: str) -> list[str]:
    """Return the matched substrings of any regex pattern found in the (folded) text."""
    folded = textnorm.fold(text)
    out = []
    for pat in patterns:
        out.extend(m.group(0) for m in re.finditer(pat, folded))
    return out
