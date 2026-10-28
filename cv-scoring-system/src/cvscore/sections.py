"""CV structure: sections, contact data and the timeline of dated entries.

The parser does not need a well-formed CV. It tolerates:

* headings in any case, letter-spaced, with a trailing colon, in Polish or English;
* entries where the date is on the title line, before it or after it;
* every common date notation: ``03.2019 - 06.2021``, ``III 2019 – obecnie``,
  ``marzec 2019 - nadal``, ``2019-2021``, ``od 2018``, ``since 2018``, ``2019``.
"""

from __future__ import annotations

import datetime as dt
import re
import statistics
from dataclasses import dataclass, field
from functools import lru_cache

from . import textnorm
from .extract import Document, Line
from .lexicon import lexicons, phrase_stems

# ------------------------------------------------------------------- sections


@dataclass
class Section:
    """A CV section; ``key`` is a canonical name from lexicons.yaml (or ``header``)."""

    key: str
    title: str
    lines: list[str] = field(default_factory=list)
    start: int = 0  # index of the first body line in Document.lines

    @property
    def text(self) -> str:
        """Body text of the section."""
        return "\n".join(self.lines)


def _heading_variants(kind: str = "cv_sections") -> list[tuple[tuple[str, ...], str]]:
    out = []
    for key, variants in lexicons()[kind].items():
        for variant in variants:
            out.append((phrase_stems(variant), key))
    out.sort(key=lambda item: -len(item[0]))
    return out


def match_heading(text: str, kind: str = "cv_sections", strict: bool = False) -> str | None:
    """Return the section key if ``text`` reads like a heading of that section.

    ``strict``: the heading must be the whole line (used when the line has no
    distinguishing style); otherwise up to two extra words are tolerated, e.g.
    "Doświadczenie zawodowe i artystyczne".
    """
    clean = text.strip().strip(":-–—|•").strip()
    if not clean or len(clean) > 60 or clean.endswith((".", ",", ";")):
        return None
    st = textnorm.stems(clean)
    if not st or len(st) > 7:
        return None
    for seq, key in _heading_variants(kind):
        if tuple(st) == seq:
            return key
        if not strict and tuple(st[: len(seq)]) == seq and len(st) - len(seq) <= 2:
            return key
    # letter-spaced headings often lose their word spaces in PDF extraction:
    # "DOŚWIADCZENIEZAWODOWE", "KURSYISZKOLENIA"
    squashed = re.sub(r"[^a-z0-9]", "", textnorm.fold(clean))
    if len(st) <= 2 and len(squashed) >= 6:
        return _squashed_variants(kind).get(squashed)
    return None


@lru_cache(maxsize=None)
def _squashed_variants(kind: str) -> dict[str, str]:
    """Heading variants with spaces removed (folded, unstemmed)."""
    out = {}
    for key, variants in lexicons()[kind].items():
        for variant in variants:
            out.setdefault(re.sub(r"[^a-z0-9]", "", textnorm.fold(variant)), key)
    return out


def _body_size(lines: list[Line]) -> float:
    sizes = [ln.size for ln in lines if ln.size > 0 and len(ln.text) > 30]
    return statistics.median(sizes) if sizes else 0.0


@lru_cache(maxsize=1)
def _europass_re() -> re.Pattern:
    labels = sorted(lexicons()["europass_labels"], key=len, reverse=True)
    return re.compile(r"^(?:" + "|".join(re.escape(lbl) for lbl in labels) + r")\s*(?::|\s{1,}|$)")


def strip_label(text: str) -> str:
    """Remove a leading Europass field label; returns '' for a label-only line.

    The label is removed when followed by a colon, by nothing, or by a value that starts
    with a capital letter or a digit ("Zajmowane stanowisko Instruktorka", "Daty 09/2018"),
    so an ordinary sentence such as "Stanowisko kierownicze objęłam..." stays intact.
    """
    folded = textnorm.fold(text)
    match = _europass_re().match(folded)
    if not match:
        return text
    rest = text[match.end():].strip(" :\t")
    if not rest or ":" in match.group(0) or rest[0].isupper() or rest[0].isdigit():
        return rest
    return text


def split_sections(doc: Document, kind: str = "cv_sections") -> list[Section]:
    """Split a document into sections using heading words and typography."""
    lines = doc.lines or [Line(t, 1) for t in doc.text.split("\n") if t.strip()]
    body = _body_size(lines)
    sections = [Section("header", "", [], 0)]
    clause_started: int | bool = False  # page number of the clause once found
    for idx, ln in enumerate(lines):
        text = ln.text.strip()
        if kind == "cv_sections":
            text = strip_label(text)
            if not text:
                continue
        folded = textnorm.fold(text)
        if kind == "cv_sections" and not clause_started and (
                "wyrazam zgode na przetwarzanie" in folded or "i hereby" in folded and "consent" in folded
                or "zgode na przetwarzanie moich danych" in folded):
            clause_started = ln.page
            sections.append(Section("clause", "Klauzula RODO", [text], idx + 1))
            continue
        if clause_started:
            if ln.page == clause_started and not match_heading(text, kind, strict=True):
                sections[-1].lines.append(text)
                continue
            # content after the clause on a later page: an appendix, usually the portfolio
            clause_started = False
            if not match_heading(text, kind, strict=True):
                sections.append(Section("portfolio", "Załącznik", [], idx))
        styled = (body and ln.size >= body * 1.12) or ln.bold or (text.isupper() and len(text) > 3)
        key = match_heading(text, kind, strict=not styled)
        if key:
            sections.append(Section(key, text, [], idx + 1))
            continue
        # "Wymagane dokumenty: CV, list..." / "Języki: angielski B2" - heading and content on one line
        head, colon, tail = text.partition(":")
        key = match_heading(head, kind, strict=True) if colon and tail.strip() and len(head) <= 40 else None
        if key and not re.search(r"\d", head):
            sections.append(Section(key, head.strip(), [tail.strip()], idx + 1))
        else:
            sections[-1].lines.append(text)
    # merge consecutive sections with the same key (heading repeated on page 2)
    merged: list[Section] = []
    for sec in sections:
        if merged and merged[-1].key == sec.key and sec.key != "header":
            merged[-1].lines.extend(sec.lines)
        else:
            merged.append(sec)
    return [s for s in merged if s.lines or s.key != "header"]


def section_text(sections: list[Section], *keys: str) -> str:
    """Concatenated text of all sections with the given keys."""
    return "\n".join(s.text for s in sections if s.key in keys)


# --------------------------------------------------------------------- contact

EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
PHONE_RE = re.compile(r"(?<!\d)(?:\+?\d{2}[\s-]?)?(?:\d{3}[\s-]?\d{3}[\s-]?\d{3}|\d{2}[\s-]?\d{3}[\s-]?\d{2}[\s-]?\d{2})(?!\d)")
PORTFOLIO_HOSTS = ("behance", "dribbble", "artstation", "instagram", "pinterest", "issuu", "saatchiart",
                   "etsy", "vimeo", "youtube", "tiktok", "github", "gitlab", "kaggle", "wix", "squarespace",
                   "cargo", "format.com", "carbonmade", "wordpress", "blogspot", "facebook")


@dataclass
class Contact:
    """Contact data found anywhere in the document."""

    emails: list[str]
    phones: list[str]
    linkedin: list[str]
    portfolio: list[str]
    other_links: list[str]
    city: str | None


def extract_contact(doc: Document) -> Contact:
    """Find e-mail, phone, LinkedIn, portfolio links and a probable city."""
    text = doc.text
    emails = list(dict.fromkeys(EMAIL_RE.findall(text) + [
        u[7:] for u in doc.links if u.lower().startswith("mailto:")]))
    phones = list(dict.fromkeys(p.strip() for p in PHONE_RE.findall(text)
                                if not re.fullmatch(r"(19|20)\d{2}\s*-\s*(19|20)\d{2}", p.strip())))
    links = [u for u in doc.all_links() if not u.lower().startswith(("mailto:", "tel:"))]
    linkedin = [u for u in links if "linkedin" in u.lower()]
    portfolio = [u for u in links if any(h in u.lower() for h in PORTFOLIO_HOSTS) and "linkedin" not in u.lower()]
    other = [u for u in links if u not in linkedin and u not in portfolio]
    # a personal website is a portfolio too, unless it is an employer's address in the body
    for url in list(other):
        host = re.sub(r"^https?://(www\.)?|^www\.", "", url.lower()).split("/")[0]
        local = EMAIL_RE.search(text)
        if local and host and host.split(".")[0] in local.group(0).lower():
            portfolio.append(url)
            other.remove(url)
    city = None
    head = "\n".join(text.split("\n")[:25])
    match = re.search(r"\b(\d{2}-\d{3})\s+([A-ZĄĆĘŁŃÓŚŹŻ][\wąćęłńóśźż-]+)", head)
    if match:
        city = match.group(2)
    else:
        for name in KNOWN_CITIES:
            if re.search(rf"\b{name}\b", head):
                city = name
                break
    return Contact(emails, phones, linkedin, portfolio, other, city)


KNOWN_CITIES = ("Warszawa", "Kraków", "Łódź", "Wrocław", "Poznań", "Gdańsk", "Szczecin", "Bydgoszcz",
                "Lublin", "Białystok", "Katowice", "Gdynia", "Częstochowa", "Radom", "Toruń", "Sosnowiec",
                "Kielce", "Rzeszów", "Gliwice", "Zabrze", "Olsztyn", "Bielsko-Biała", "Bytom", "Zielona Góra",
                "Rybnik", "Ruda Śląska", "Opole", "Tychy", "Gorzów Wielkopolski", "Elbląg", "Płock",
                "Wałbrzych", "Włocławek", "Tarnów", "Chorzów", "Koszalin", "Kalisz", "Legnica", "Sopot")

# -------------------------------------------------------------------- timeline

_MONTHS = {
    1: ["styczen", "stycznia", "sty", "january", "jan", "i"],
    2: ["luty", "lutego", "lut", "february", "feb", "ii"],
    3: ["marzec", "marca", "mar", "march", "iii"],
    4: ["kwiecien", "kwietnia", "kwi", "april", "apr", "iv"],
    5: ["maj", "maja", "may", "v"],
    6: ["czerwiec", "czerwca", "cze", "june", "jun", "vi"],
    7: ["lipiec", "lipca", "lip", "july", "jul", "vii"],
    8: ["sierpien", "sierpnia", "sie", "august", "aug", "viii"],
    9: ["wrzesien", "wrzesnia", "wrz", "september", "sep", "sept", "ix"],
    10: ["pazdziernik", "pazdziernika", "paz", "october", "oct", "x"],
    11: ["listopad", "listopada", "lis", "november", "nov", "xi"],
    12: ["grudzien", "grudnia", "gru", "december", "dec", "xii"],
}
_MONTH_LOOKUP = {name: num for num, names in _MONTHS.items() for name in names}
_MONTH_ALT = "|".join(sorted(_MONTH_LOOKUP, key=len, reverse=True))
_YEAR = r"(?:19[6-9]\d|20[0-4]\d)"
# optional day (Europass: 01/09/2018, 15.03.2021, "15 marca 2021"), month (number, Polish/English
# name or Roman numeral) and the year
_DAY = r"(?:(?:0?[1-9]|[12]\d|3[01])\s*[./-]\s*(?=(?:0?[1-9]|1[0-2])\s*[./-]\s*(?:19|20)\d{{2}})|(?:0?[1-9]|[12]\d|3[01])\s+(?=[a-z]))?"
_DATE = (rf"(?:{_DAY}(?:(?P<m{{n}}>0?[1-9]|1[0-2])\s*[./-]\s*|(?P<mn{{n}}>{_MONTH_ALT})\.?\s+)?"
         rf"(?P<y{{n}}>{_YEAR}))")
_OPEN = r"(?P<open>obecnie|nadal|teraz|dzis|aktualnie|do chwili obecnej|do teraz|w trakcie|trwa|present|now|current|ongoing|today)"
_RANGE_RE = re.compile(
    rf"(?<![\d/])(?:od\s+|since\s+|from\s+)?{_DATE.format(n=1)}\s*(?:-|do|to|until|–)\s*(?:{_DATE.format(n=2)}|{_OPEN})(?![\d/])"
)
_SINCE_RE = re.compile(rf"(?:\bod\b|\bsince\b|\bfrom\b)\s+{_DATE.format(n=1)}(?!\s*(?:-|do|to)\s*\d)")
_SINGLE_RE = re.compile(rf"(?<![\d/.-]){_DATE.format(n=1)}(?![\d/-])")


@dataclass
class Entry:
    """A dated item: job, education, exhibition, project..."""

    section: str
    title: str
    start: dt.date
    end: dt.date
    ongoing: bool
    point: bool  # a single date (exhibition year) rather than a range
    lines: list[str] = field(default_factory=list)
    date_text: str = ""

    @property
    def months(self) -> int:
        """Duration in months (at least 1)."""
        return max(1, (self.end.year - self.start.year) * 12 + self.end.month - self.start.month + 1)

    @property
    def text(self) -> str:
        """Title and description."""
        return "\n".join([self.title] + self.lines)

    def months_since_end(self, today: dt.date) -> int:
        """Months between the end of the entry and ``today`` (0 if ongoing)."""
        if self.ongoing:
            return 0
        return max(0, (today.year - self.end.year) * 12 + today.month - self.end.month)


def _date_from(match: re.Match, n: int, end: bool) -> dt.date | None:
    year = match.group(f"y{n}")
    if not year:
        return None
    month = match.group(f"m{n}")
    month_name = match.group(f"mn{n}")
    if month:
        mon = int(month)
    elif month_name:
        mon = _MONTH_LOOKUP.get(month_name.lower().rstrip("."), 12 if end else 1)
    else:
        mon = 12 if end else 1
    return dt.date(int(year), mon, 1)


def find_dates(line: str, today: dt.date) -> tuple[dt.date, dt.date, bool, bool, str] | None:
    """Parse a date range / single date in a line.

    Returns ``(start, end, ongoing, point, matched_text)`` or ``None``. A bare year
    range ``2019-2021`` ends in December of the last year; an open range ends today.
    """
    folded = textnorm.fold(line).replace("—", "-")
    match = _RANGE_RE.search(folded)
    if match:
        start = _date_from(match, 1, end=False)
        if match.group("open"):
            return start, today, True, False, match.group(0)
        end = _date_from(match, 2, end=True)
        if end and start and end >= start:
            # "2019 - 2021" with no months: if the start year has no month, Jan is fine
            return start, min(end, today), end >= today, False, match.group(0)
    match = _SINCE_RE.search(folded)
    if match:
        return _date_from(match, 1, end=False), today, True, False, match.group(0)
    match = _SINGLE_RE.search(folded)
    if match:
        # single year / month-year ("2019 Wystawa ...", "... Kraków, 2019"): only at the start
        # or the end of the line, or on a short line - not "ok. 2000 uczestników" in a bullet
        body = folded.strip().lstrip("•*-– ").strip()
        at_edge = body.startswith(match.group(0).strip()) or body.rstrip(" .)").endswith(match.group(0).strip())
        if at_edge or len(body) < 45:
            start = _date_from(match, 1, end=False)
            end = _date_from(match, 1, end=True)
            if match.group("m1") or match.group("mn1"):
                end = start
            return start, min(end, today), False, True, match.group(0)
    return None


DATED_SECTIONS = ("experience", "education", "courses", "volunteering", "exhibitions", "awards",
                  "portfolio", "publications", "header", "summary")


def _strip_date(text: str, date_text: str) -> str:
    folded = textnorm.fold(text)
    pos = folded.find(date_text)
    if pos < 0:
        return text.strip(" |-–,")
    return (text[:pos] + " " + text[pos + len(date_text):]).strip(" |-–,:()•\t")


def timeline(sections: list[Section], today: dt.date | None = None) -> list[Entry]:
    """Build dated entries from every dated section."""
    today = today or dt.date.today()
    entries: list[Entry] = []
    for sec in sections:
        if sec.key not in DATED_SECTIONS:
            continue
        current: Entry | None = None
        pending: list[str] = []  # short non-bullet lines that may be the next entry's title
        # Layout of the section, decided by its first entry: "inline" = the date sits on the
        # title line ("Edukatorka 2016-2018"), so short lines that follow it are that entry's
        # organisation; otherwise the title precedes the dated line ("Edukatorka" /
        # "Galeria X   2016-2018") and the last pending line is the next entry's title.
        inline: bool | None = None
        for line in sec.lines:
            found = find_dates(line, today)
            is_bullet = line.startswith("•")
            if found and not (is_bullet and found[3]):
                start, end, ongoing, point, date_text = found
                rest = _strip_date(line, date_text)
                if inline is None:
                    inline = bool(rest) and not pending
                head: list[str] = []
                if pending and (current is None or not inline or not rest):
                    head = [pending.pop(-1)]
                if current is not None:
                    current.lines.extend(pending)
                elif pending:
                    head = pending[-1:] + head
                pending = []
                title = " - ".join(head + ([rest] if rest else []))
                current = Entry(sec.key, title, start, end, ongoing, point, [], date_text)
                entries.append(current)
            elif current is None:
                if not is_bullet:
                    pending.append(line)
            elif not current.title and not is_bullet:
                current.title = line
            elif not is_bullet and len(line) < 70 and not line.endswith((".", ",")):
                pending.append(line)
            else:
                current.lines.extend(pending)
                pending = []
                current.lines.append(line)
        if current is not None:
            # trailing short lines after the last date belong to that entry
            current.lines.extend(pending)
    return entries


def merged_months(entries: list[Entry]) -> int:
    """Total months covered by the entries, counting overlaps once."""
    spans = sorted((e.start, e.end) for e in entries if not e.point)
    total, cur_s, cur_e = 0, None, None
    for start, end in spans:
        if cur_e is None or start > cur_e:
            if cur_e is not None:
                total += (cur_e.year - cur_s.year) * 12 + cur_e.month - cur_s.month + 1
            cur_s, cur_e = start, end
        else:
            cur_e = max(cur_e, end)
    if cur_e is not None:
        total += (cur_e.year - cur_s.year) * 12 + cur_e.month - cur_s.month + 1
    return total


def gaps(entries: list[Entry], min_months: int = 6) -> list[tuple[dt.date, dt.date, int]]:
    """Gaps of at least ``min_months`` between (merged) job entries."""
    spans = sorted((e.start, e.end) for e in entries if not e.point)
    out, reach = [], None
    for start, end in spans:
        if reach is not None:
            months = (start.year - reach.year) * 12 + start.month - reach.month - 1
            if months >= min_months:
                out.append((reach, start, months))
        reach = end if reach is None else max(reach, end)
    return out
