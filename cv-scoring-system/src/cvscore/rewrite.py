"""Deterministic rewrite drafts for weak CV bullets.

No text is invented: the drafts restructure what is already written and leave
``[placeholders]`` for the facts only the candidate knows (how many, for whom,
what changed). Grammatical gender follows the CV (-łam / -łem).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from . import textnorm
from .analysis import CVAnalysis
from .lexicon import phrase_set
from .metrics.content import _is_result, _numbers

# verbal noun (folded, first word) -> past tense, feminine; masculine is derived
NOUN_TO_VERB = {
    "prowadzenie": "Prowadziłam", "organizacja": "Zorganizowałam", "organizowanie": "Organizowałam",
    "projektowanie": "Projektowałam", "tworzenie": "Tworzyłam", "przygotowanie": "Przygotowałam",
    "przygotowywanie": "Przygotowywałam", "koordynacja": "Koordynowałam", "koordynowanie": "Koordynowałam",
    "realizacja": "Zrealizowałam", "opracowanie": "Opracowałam", "opracowywanie": "Opracowywałam",
    "obsluga": "Obsługiwałam", "wspolpraca": "Współpracowałam", "zarzadzanie": "Zarządzałam",
    "planowanie": "Planowałam", "redagowanie": "Redagowałam", "moderowanie": "Moderowałam",
    "nadzor": "Nadzorowałam", "wdrozenie": "Wdrożyłam", "wdrazanie": "Wdrażałam", "pisanie": "Pisałam",
    "animowanie": "Animowałam", "aranzacja": "Aranżowałam", "malowanie": "Malowałam", "ilustrowanie": "Ilustrowałam",
    "szkolenie": "Szkoliłam", "nauczanie": "Uczyłam", "pozyskiwanie": "Pozyskiwałam", "rozliczanie": "Rozliczałam",
    "wspoltworzenie": "Współtworzyłam", "wspolorganizacja": "Współorganizowałam",
}
_DUTY_RE = re.compile(r"(?i)^(odpowiedzialn[ay] za|odpowiedzialność za|do moich obowiązków należał[ao]?|"
                      r"moje obowiązki:?|zajmowałam się|zajmowałem się|responsible for)\s+")
_WEAK_RE = re.compile(r"(?i)^(pomagałam|pomagałem|pomoc|wspierałam|wspierałem|wsparcie|uczestniczyłam|uczestniczyłem|"
                      r"brałam udział|brałem udział|udział|asystowałam|asystowałem)\s+(w|przy|we)?\s*")


@dataclass
class Rewrite:
    """A weak bullet and a restructured draft."""

    original: str
    draft: str
    why: str


def _gender(cv: CVAnalysis) -> str:
    text = cv.doc.text
    fem = len(re.findall(r"\b\w{3,}łam\b", text))
    masc = len(re.findall(r"\b\w{3,}łem\b", text))
    return "m" if masc > fem else "f"


def _verb(word: str, gender: str) -> str:
    return word.replace("łam", "łem") if gender == "m" else word


def _tail(has_number: bool, has_result: bool) -> str:
    parts = []
    if not has_number:
        parts.append("[dla kogo / ile osób, grup, edycji]")
    if not has_result:
        parts.append("efekt: [co powstało / co się zmieniło]")
    return (" – " + "; ".join(parts)) if parts else ""


def draft(text: str, gender: str = "f") -> Rewrite | None:
    """Draft for one bullet, or None when the bullet is already strong."""
    body = text.lstrip("•-– ").strip()
    has_number, has_result = bool(_numbers(body)), _is_result(body)
    starts_action = phrase_set("action_verbs").starts(body)
    m = _DUTY_RE.match(body)
    if m:
        rest = body[m.end():]
        first = textnorm.fold(rest.split()[0]) if rest.split() else ""
        verb = NOUN_TO_VERB.get(first)
        new = (_verb(verb, gender) + " " + " ".join(rest.split()[1:])) if verb else \
            (_verb("Realizowałam", gender) + ": " + rest)
        return Rewrite(text, new.rstrip(".") + _tail(has_number, has_result),
                       "obowiązek zamiast działania i efektu")
    m = _WEAK_RE.match(body)
    if m:
        rest = body[m.end():]
        return Rewrite(text, _verb("Współprowadziłam", gender) + " " + rest.rstrip(".") +
                       " (moja część: [co robiłaś samodzielnie])" + _tail(has_number, has_result),
                       "słaba samodzielność ('pomagałam', 'udział')")
    words = body.split()
    first = textnorm.fold(words[0]) if words else ""
    if first in NOUN_TO_VERB and not starts_action and (not has_number or not has_result):
        return Rewrite(text, _verb(NOUN_TO_VERB[first], gender) + " " + " ".join(words[1:]).rstrip(".") +
                       _tail(has_number, has_result), "rzeczownik zamiast czasownika, brak skali/efektu")
    if not has_number and not has_result and len(words) >= 4:
        return Rewrite(text, body.rstrip(".") + _tail(False, False), "brak skali i efektu")
    return None


def suggest(cv: CVAnalysis, limit: int = 8) -> list[Rewrite]:
    """Drafts for the weakest experience bullets (most recent roles first)."""
    gender = _gender(cv)
    out = []
    for st in cv.statements:
        if st.section not in ("experience", "volunteering"):
            continue
        rw = draft(st.text, gender)
        if rw is not None:
            out.append(rw)
        if len(out) >= limit:
            break
    return out
