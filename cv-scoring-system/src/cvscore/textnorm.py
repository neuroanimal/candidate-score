"""Text normalisation for Polish and English CVs, cover letters and job offers.

Everything that compares text in this package goes through the functions here, so
that a CV and a job offer are always normalised the same way:

* ``clean_text``   - repairs artefacts typical of PDF export (Adobe InDesign in
  particular): ligatures, soft hyphens, icon glyphs from the Private Use Area,
  letter-spaced headings ("D O Ś W I A D C Z E N I E") and words hyphenated at
  line ends.
* ``fold``         - lower-case and strip Polish diacritics (``ł`` included, which
  Unicode does not decompose), so that "Doświadczenie" == "doswiadczenie".
* ``stem``         - a deliberately light suffix-stripping stemmer. Polish is
  heavily inflected ("warsztaty", "warsztatów", "warsztatach"); a real lemmatiser
  (Morfeusz, spaCy pl) is not required - both sides of every comparison are
  stemmed by the same function, so consistency matters more than linguistic
  correctness.
"""

from __future__ import annotations

import re
import unicodedata
from functools import lru_cache

# Characters Unicode NFKD does not decompose into base letter + combining mark.
_EXTRA_FOLD = str.maketrans({
    "ł": "l", "Ł": "l", "đ": "d", "ß": "ss", "æ": "ae", "ø": "o", "œ": "oe",
    "‐": "-", "‑": "-", "‒": "-", "–": "-", "—": "-", "―": "-", "−": "-",
    "’": "'", "‘": "'", "‚": "'", "“": '"', "”": '"', "„": '"', "«": '"', "»": '"',
})

# Bullet-like glyphs used by designers; normalised to a single "•".
BULLET_CHARS = "•●▪■□◦○◆◇►▸▹➢➤✓✔✗☐☑-*·‣⁃∙"
_BULLET_RE = re.compile(r"^\s*[" + re.escape(BULLET_CHARS) + r"]\s+")
# numbered lists in public-sector offers: "1) wykształcenie wyższe;", "2. staż", "a) prawo jazdy"
_NUMBERED_RE = re.compile(r"^\s*(?:\d{1,2}|[a-z])[).]\s+(?=[A-Za-ząćęłńóśźżĄĆĘŁŃÓŚŹŻ])")

# Private Use Area - icon fonts (Font Awesome etc.) end up here in PDF text.
_PUA_RE = re.compile("[\ue000-\uf8ff\U000f0000-\U000ffffd]")
_CID_RE = re.compile(r"\(cid:\d+\)")
_WS_RE = re.compile(r"[ \t\u00a0\u2000-\u200b\u202f\u205f\u3000]+")
# A line made of single letters separated by single spaces, words by 2+ spaces.
_SPACED_RE = re.compile(r"(?<!\S)(?:\w ){3,}\w(?!\S)")
_HYPHEN_EOL_RE = re.compile(r"([a-ząćęłńóśźż]{2,})-\n([a-ząćęłńóśźż]{2,})")
_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[+#][a-z0-9+#]*)?|[a-z]+\+\+|c#")

POLISH_DIACRITICS = set("ąćęłńóśźżĄĆĘŁŃÓŚŹŻ")


def collapse_spaced_letters(line: str) -> str:
    """Join letter-spaced text such as ``"P O R T F O L I O"`` into ``"PORTFOLIO"``.

    InDesign tracking or manual spacing makes PDF extractors emit one space
    between every letter. Words are then separated by two or more spaces, so the
    line is split on 2+ spaces first and each part checked separately.
    """
    parts = re.split(r"( {2,})", line)
    out = []
    for part in parts:
        if part.strip() and _SPACED_RE.fullmatch(part.strip()):
            out.append(part.replace(" ", ""))
        else:
            out.append(part)
    joined = "".join(out)
    # A fully spaced line like "D O Ś W I A D C Z E N I E  Z A W O D O W E"
    return re.sub(r" {2,}", " ", joined) if joined != line else line


def clean_text(text: str) -> str:
    """Repair PDF/OCR artefacts and return text with one logical line per line."""
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)  # ligatures ﬁ ﬂ, full-width forms
    text = text.replace("\u00ad", "")  # soft hyphen
    text = _CID_RE.sub("", text)
    text = _PUA_RE.sub(" ", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\f", "\n")
    # Words broken at the end of a line: "warsz-\ntaty" -> "warsztaty".
    text = _HYPHEN_EOL_RE.sub(r"\1\2", text)
    lines = []
    for raw in text.split("\n"):
        line = _WS_RE.sub(lambda m: "  " if len(m.group(0)) > 1 else " ", raw)
        line = collapse_spaced_letters(line)
        line = re.sub(r" {2,}", " ", line).strip()
        line = _BULLET_RE.sub("• ", line)
        line = _NUMBERED_RE.sub("• ", line)
        lines.append(line)
    # Collapse runs of blank lines to one.
    out, blank = [], False
    for line in lines:
        if not line:
            if not blank:
                out.append("")
            blank = True
        else:
            out.append(line)
            blank = False
    return "\n".join(out).strip()


def fold(text: str) -> str:
    """Lower-case and strip diacritics: ``"Łódź Świętokrzyska"`` -> ``"lodz swietokrzyska"``."""
    text = text.translate(_EXTRA_FOLD)
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return text.lower()


# Suffixes in folded form, longest first. Polish nominal, adjectival and verbal
# (past tense, both genders: prowadzilam / prowadzilem) endings plus a few English ones.
_SUFFIXES = sorted({
    # verbal nouns and their cases
    "owaniami", "owaniach", "owaniem", "owaniu", "owania", "owanie",
    "ywaniem", "ywania", "ywanie", "ywaniu",
    "eniami", "eniach", "eniem", "eniu", "enia", "enie",
    "aniami", "aniach", "aniem", "aniu", "ania", "anie",
    "osciami", "osciach", "oscia", "osci", "osc",
    "acjami", "acjach", "acja", "acje", "acji",
    # past tense, masculine / feminine / plural, 1st and 3rd person
    "owalysmy", "owalismy", "owalam", "owalem", "owala", "owali", "owaly", "owal",
    "ywalam", "ywalem", "ywala", "ywal",
    "ilam", "ilem", "ila", "ili", "ily", "il",
    "ylam", "ylem", "yla", "yli", "yly", "yl",
    "alam", "alem", "ala", "ali", "aly", "al",
    "elam", "elem", "ela", "eli", "ely",
    "lam", "lem",
    # adjectives
    "owymi", "owych", "owego", "owemu", "owej", "owym", "owa", "owe", "owy",
    "nymi", "nych", "nego", "nemu", "nej", "nym", "na", "ne", "ny",
    "ymi", "ych", "ego", "emu", "ej", "ym", "imi", "ich", "im",
    "ska", "skie", "ski", "skich", "skim", "skiej", "skiego",
    # nominal cases
    "ami", "ach", "om", "owi", "owie", "ow", "em", "ie", "ia", "iu", "ii",
    "a", "e", "i", "o", "u", "y",
    # English
    "ations", "ation", "ments", "ment", "ings", "ing", "ies", "ers", "er", "ed", "es", "ly", "s",
}, key=len, reverse=True)

_MIN_STEM = 4

# Past-tense endings (folded) that mark a verb, and perfective prefixes that are
# removed from such verbs so that "zorganizowalam" meets "organizacja" and
# "zaprojektowalam" meets "projektowanie". Restricted to verbs: nouns such as
# "zespol" or "zajecia" must keep their first letters.
_PAST_ENDINGS = ("lam", "lem", "lysmy", "lismy", "ala", "ila", "yla", "ali", "ili", "yli", "aly", "ily", "yly",
                 "al", "il", "yl")
_PERFECTIVE = ("prze", "przy", "roz", "wy", "za", "po", "na", "z", "s")
_INFIX = ("ow", "yw", "iw")

# Feminine job titles -> masculine base, so that an offer for a "grafik / instruktor"
# matches a CV of a "graficzka / instruktorka" (any case ending).
_FEM_END = r"(?:a|i|e|ę|ą|om|ami|ach)"
_FEMININE = [
    (re.compile(rf"^(.{{3,}})iczk{_FEM_END}$"), r"\1ik"),           # graficzka -> grafik
    (re.compile(rf"^(.{{4,}}(?:tor|sor|ist|ant|ent|eut|ciel|log|nik|ik|yk))k{_FEM_END}$"), r"\1"),  # instruktorka, specjalistka
    (re.compile(rf"^(.{{4,}})ozk{_FEM_END}$"), r"\1og"),            # pedagozka -> pedagog
    (re.compile(rf"^(.{{3,}}(?:al|ch|rz)?ar)k{_FEM_END}$"), r"\1z"),  # malarka -> malarz, kucharka -> kucharz
]


@lru_cache(maxsize=200_000)
def stem(token: str) -> str:
    """Strip the longest known suffix while keeping at least 4 characters.

    ``token`` must already be folded. Tokens with digits or of length <= 4 are
    returned unchanged ("sql", "b2", "2020"). Perfective prefixes are removed from
    past-tense verbs, and a trailing "ow"/"yw" left by an aspect infix is dropped
    ("opracowywanie" and "opracowalam" both give "oprac").
    """
    if len(token) <= _MIN_STEM or any(ch.isdigit() for ch in token):
        return token
    for rx, repl in _FEMININE:
        if rx.match(token):
            token = rx.sub(repl, token)
            break
    if token.endswith(_PAST_ENDINGS):
        for pre in _PERFECTIVE:
            if token.startswith(pre) and len(token) - len(pre) >= 8:
                token = token[len(pre):]
                break
    out = token
    for suf in _SUFFIXES:
        if token.endswith(suf) and len(token) - len(suf) >= _MIN_STEM:
            out = token[: -len(suf)]
            break
    if out.endswith(_INFIX) and len(out) - 2 >= _MIN_STEM + 1:
        out = out[:-2]
    if out.endswith("nic") and len(out) >= 6:  # k/c alternation: uczestnicy / uczestnik
        out = out[:-1] + "k"
    return out


def tokens(text: str) -> list[str]:
    """Return folded tokens (letters/digits, keeping ``c++``, ``c#``)."""
    return _TOKEN_RE.findall(fold(text))


def stems(text: str) -> list[str]:
    """Return stemmed folded tokens."""
    return [stem(t) for t in tokens(text)]


# Function words removed before similarity computations (folded forms).
STOPWORDS = frozenset("""
a aby ale albo ani az bardzo bez bo by byc byl byla bylo byly bede beda bedzie
ci cie co czy dla do gdy gdzie go i ich ile im inne inny iz ja jak jako je jego jej
jest jestem jestes jeszcze juz ktora ktore ktory ktorych ktorym ktorzy lub ma mam
mi mnie moj moja moje moze mozna mu my na nad nam nas nasz nasza nasze naszej nie
niz o od oraz po pod poza przez przy sa sie sobie tak takze tam te tego tej ten to
tu ty tylko w we wiec wszystko wy z za ze zeby oraz rowniez itp itd np m.in min tj
the and or of to in on for with a an is are be by as at from that this it its our
your you we will can have has was were not but also into over under per via
""".split())

# Words that occur in almost every requirement and carry no domain meaning.
GENERIC_REQUIREMENT_WORDS = frozenset(stem(w) for w in """
doswiadczenie doswiadczenia znajomosc umiejetnosc umiejetnosci praca pracy
wymagane wymagany wymagana mile widziane atutem atut lat lata rok roku minimum min
dobra dobry dobre bardzo wysoka wysokie wysoki zdolnosc zdolnosci kandydat osoba
osoby experience knowledge skills skill ability years year good strong excellent
required preferred plus nice have must candidate work working
""".split())


def content_stems(text: str, extra_stop: frozenset[str] | None = None) -> list[str]:
    """Stems without stop-words, 1-letter tokens and (optionally) extra stems."""
    out = []
    for tok in tokens(text):
        if tok in STOPWORDS or len(tok) < 2:
            continue
        st = stem(tok)
        if extra_stop and st in extra_stop:
            continue
        out.append(st)
    return out


_SENT_SPLIT_RE = re.compile(r"(?<=[.!?;])\s+(?=[A-ZĄĆĘŁŃÓŚŹŻ0-9•])")


def sentences(text: str) -> list[str]:
    """Split into sentence-like units: lines first, then sentence punctuation.

    CVs are mostly bullet lists, so a line is treated as a unit on its own;
    paragraphs (cover letters) are additionally split on ``. ! ? ;``.
    """
    out = []
    for line in text.split("\n"):
        line = line.strip().lstrip("•").strip()
        if not line:
            continue
        out.extend(s.strip() for s in _SENT_SPLIT_RE.split(line) if s.strip())
    return out


def has_polish_diacritics(text: str) -> bool:
    """True if the text contains at least one Polish diacritic letter."""
    return any(ch in POLISH_DIACRITICS for ch in text)


_VOWELS = set("aeiouyąęó")


def syllables(word: str) -> int:
    """Approximate syllable count: groups of vowels (works for Polish and roughly English)."""
    word = word.lower()
    count, prev = 0, False
    for ch in word:
        is_v = ch in _VOWELS
        if is_v and not prev:
            count += 1
        prev = is_v
    # "ia", "ie", "io", "iu" after a consonant ("nie", "ścią", "historia") already form one
    # vowel group, which is how Polish syllabifies them - no correction needed.
    return max(1, count)


def garbage_ratio(text: str) -> float:
    """Share of characters/words that look like broken font encoding.

    Returns a value in 0..1. PDFs whose fonts lack a ToUnicode map yield text such
    as ``"Ä‡Å‚Ã³"`` or long runs without vowels; values above ~0.25 mean the text
    layer should not be trusted and the page should be OCR-ed.
    """
    stripped = [ch for ch in text if not ch.isspace()]
    if not stripped:
        return 1.0
    allowed = sum(
        1 for ch in stripped
        if ch.isascii() or ch in POLISH_DIACRITICS or ch in "•–—’‘“”„«»€§°±×…"
        or unicodedata.category(ch).startswith(("L", "N", "P", "S"))
        and unicodedata.name(ch, "").startswith(("LATIN", "DIGIT"))
    )
    bad_chars = 1.0 - allowed / len(stripped)
    words = re.findall(r"[^\W\d_]{4,}", text)
    if not words:
        return bad_chars
    no_vowel = sum(1 for w in words if not _VOWELS & set(w.lower()))
    mojibake = len(re.findall(r"[ÃÅÄ][\u0080-\u00bf]", text))
    return min(1.0, max(bad_chars, no_vowel / len(words)) + mojibake / max(1, len(words)))
