"""Structured views of a CV, a job offer and a cover letter, built once and shared
by all metrics."""

from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass, field

from . import textnorm
from .extract import Document
from .lexicon import SkillHit, lexicons, phrase_set, regex_any, taxonomy
from .sections import (Contact, Entry, Section, extract_contact, find_dates, split_sections,
                       timeline)

LEVELS = ["A1", "A2", "B1", "B2", "C1", "C2", "native"]

# Sections whose lines describe what the person did (statements / bullets)
STATEMENT_SECTIONS = ("experience", "volunteering", "portfolio", "summary", "header")
EXPERIENCE_SECTIONS = ("experience", "volunteering")


def detect_language(text: str) -> str:
    """``pl`` or ``en`` - by diacritics and frequent function words."""
    folded = textnorm.tokens(text[:6000])
    pl = sum(1 for t in folded if t in {"i", "w", "z", "na", "do", "sie", "oraz", "dla", "jest", "od", "po"})
    en = sum(1 for t in folded if t in {"the", "and", "of", "to", "in", "for", "with", "is", "on", "at"})
    if textnorm.has_polish_diacritics(text):
        pl += 5
    return "pl" if pl >= en else "en"


def language_mentions(text: str) -> dict[str, str | None]:
    """Languages mentioned in the text with the highest level stated on the same line."""
    lex = lexicons()
    out: dict[str, str | None] = {}
    for line in text.split("\n"):
        folded = " " + textnorm.fold(line) + " "
        for lang, pats in lex["languages"].items():
            if not any(re.search(rf"(?<![a-z]){p}", folded) for p in pats):
                continue
            level = None
            for lvl in reversed(LEVELS):
                words = lex["level_words"][lvl]
                if any(re.search(rf"(?<![a-z0-9]){re.escape(w)}(?![a-z0-9])", folded) for w in words):
                    level = lvl
                    break
            prev = out.get(lang)
            if prev is None or (level and LEVELS.index(level) > LEVELS.index(prev)):
                out[lang] = level
            else:
                out.setdefault(lang, level)
    return out


# ------------------------------------------------------------------------- CV


@dataclass
class Statement:
    """A line describing an activity, with its section and (if any) entry."""

    text: str
    section: str
    entry: Entry | None
    is_bullet: bool


@dataclass
class CVAnalysis:
    """Everything metrics need to know about a CV."""

    doc: Document
    sections: list[Section]
    contact: Contact
    entries: list[Entry]
    statements: list[Statement]
    skill_hits: list[tuple[SkillHit, str, Entry | None]]  # hit, section, entry
    language: str
    languages: dict[str, str | None]
    today: dt.date
    words: int = 0

    def section(self, key: str) -> Section | None:
        """First section with the key."""
        return next((s for s in self.sections if s.key == key), None)

    def has_section(self, key: str) -> bool:
        """True if the section exists and has content."""
        return any(s.key == key and s.lines for s in self.sections)

    @property
    def jobs(self) -> list[Entry]:
        """Dated experience entries (ranges, not single dates)."""
        return [e for e in self.entries if e.section in EXPERIENCE_SECTIONS and not e.point]

    def headline(self) -> str:
        """Name block: the first lines of the header plus the largest-type line on page 1 and
        the two lines after it (in designed CVs the name may be read after a sidebar)."""
        head = self.sections[0].lines[:8] if self.sections and self.sections[0].key == "header" else []
        first = [ln for ln in self.doc.lines if ln.page == 1]
        if first and any(ln.size for ln in first):
            top = max(range(len(first)), key=lambda i: first[i].size)
            head = head + [ln.text for ln in first[top:top + 3]]
        return "\n".join(dict.fromkeys(head))

    def skills_by_name(self) -> dict[str, list[tuple[SkillHit, str, Entry | None]]]:
        """Skill hits grouped by canonical name."""
        out: dict[str, list] = {}
        for item in self.skill_hits:
            out.setdefault(item[0].skill.name, []).append(item)
        return out


def _entry_for_line(entries: list[Entry], section: str, text: str) -> Entry | None:
    for entry in entries:
        if entry.section == section and (text == entry.title or text in entry.lines):
            return entry
    return None


_LINK_RE = re.compile(r"(?:https?://|www\.)\S+|\S+@\S+\.\w+|\b[\w.-]+\.(?:com|pl|net|org|eu|art|io)/\S*")


def analyze_cv(doc: Document, today: dt.date | None = None) -> CVAnalysis:
    """Build the CV analysis."""
    today = today or dt.date.today()
    sections = split_sections(doc)
    entries = timeline(sections, today)
    statements, hits = [], []
    tax = taxonomy()
    for sec in sections:
        for line in sec.lines:
            entry = _entry_for_line(entries, sec.key, line)
            # contact links are not skills ("linkedin.com/in/...", "instagram.com/...")
            plain = _LINK_RE.sub(" ", line)
            for hit in tax.find_in_stems(textnorm.stems(plain)):
                hits.append((hit, sec.key, entry))
            if sec.key not in STATEMENT_SECTIONS:
                continue
            body = line.lstrip("• ").strip()
            if len(body.split()) < 4:
                continue
            if entry is not None and line == entry.title:
                continue
            found = find_dates(body, today)
            if found and len(body.split()) < 8:
                continue  # a date line, not a statement
            for sent in textnorm.sentences(body) if len(body) > 220 else [body]:
                if len(sent.split()) >= 4:
                    statements.append(Statement(sent, sec.key, entry, line.startswith("•")))
    return CVAnalysis(doc, sections, extract_contact(doc), entries, statements, hits,
                      detect_language(doc.text), language_mentions(doc.text), today,
                      len(doc.text.split()))


# ------------------------------------------------------------------- job offer


@dataclass
class Requirement:
    """One requirement or responsibility line of a job offer."""

    text: str
    kind: str  # "must", "nice", "duty"
    skills: list[str] = field(default_factory=list)
    years: int | None = None
    language: tuple[str, str | None] | None = None
    knockouts: list[str] = field(default_factory=list)

    @property
    def stems(self) -> list[str]:
        """Content stems without generic requirement words."""
        return textnorm.content_stems(self.text, textnorm.GENERIC_REQUIREMENT_WORDS)


@dataclass
class JobOffer:
    """Parsed job offer."""

    doc: Document
    title: str
    organisation: str
    requirements: list[Requirement]
    knockouts: dict[str, dict]  # key -> {label, line, soft, document_only}
    languages: dict[str, str | None]
    years_required: int | None
    seniority: str | None
    domains: dict[str, float]
    sections: list[Section]
    language: str

    @property
    def must(self) -> list[Requirement]:
        """Must-have requirements."""
        return [r for r in self.requirements if r.kind == "must"]

    @property
    def nice(self) -> list[Requirement]:
        """Nice-to-have requirements."""
        return [r for r in self.requirements if r.kind == "nice"]

    @property
    def duties(self) -> list[Requirement]:
        """Responsibilities."""
        return [r for r in self.requirements if r.kind == "duty"]

    def skills(self, kinds=("must", "nice", "duty")) -> set[str]:
        """Canonical skill names mentioned in the given requirement kinds."""
        return {s for r in self.requirements if r.kind in kinds for s in r.skills}


_YEARS_RE = re.compile(
    r"(?:min\.?|minimum|co najmniej|przynajmniej|at least|ponad|powyzej)?\s*(\d{1,2})\s*\+?\s*"
    r"(?:-\s*\d{1,2}\s*)?(?:lat|lata|letnie|roku|rok|years?|yrs)")
_ORG_RE = re.compile(
    r"(?i)(?:\bsp\.\s*z\s*o\.\s*o\.?|\bs\.a\.|\bsp\.\s*j\.|\b(?:fundacj\w+|stowarzyszeni\w+|muzeum|galeri\w+|"
    r"centrum|dom kultury|osrodek|ośrodek|szko[lł]\w*|przedszkol\w+|teatr\w*|biblioteka|agencj\w+|studio|"
    r"instytut|uniwersytet|akademi\w+|spółdzielni\w+|spoldzielni\w+|ltd|gmbh|inc|samopomocy|zak[lł]ad\w*|"
    r"urz[aą]d\w*|szpital\w*|poradni\w*|o[sś]rod\w+|dom\w* pomocy|warsztat\w* terapii)\b)")


def _seniority_of(text: str) -> str | None:
    folded = " " + textnorm.fold(text) + " "
    for level in ("lead", "senior", "junior", "mid"):
        for word in lexicons()["seniority_signals"][level]:
            if re.search(rf"(?<![a-z]){re.escape(word)}(?![a-z])", folded):
                return level
    return None


_SITE_RE = re.compile(r"(?i)pracuj\.pl|olx|indeed|linkedin|justjoin|nofluff|gowork|praca\.pl|aplikuj|zaloguj|"
                      r"cookies|menu|szukaj|strona główna|ogłoszenia|oferty pracy|\d{1,2}[./]\d{1,2}[./]\d{2,4}|https?://")


def _title_by_typography(doc: Document) -> str:
    """The largest-type line near the top of page 1 (web pages saved as PDF have menus first)."""
    cands = [ln for ln in doc.lines[:40] if ln.page == 1 and 1 <= len(ln.text.split()) <= 14
             and not _SITE_RE.search(ln.text) and ln.size > 0]
    if not cands:
        return ""
    best = max(cands, key=lambda ln: ln.size)
    body = sorted(ln.size for ln in doc.lines if ln.size > 0)
    median = body[len(body) // 2] if body else 0
    return best.text if best.size >= median * 1.2 else ""


def parse_job(doc: Document) -> JobOffer:
    """Parse a job offer into requirements, knock-outs, languages and seniority."""
    lex = lexicons()
    sections = split_sections(doc, kind="job_sections")
    nice_set, must_set = phrase_set("nice_markers"), phrase_set("must_markers")
    tax = taxonomy()
    requirements: list[Requirement] = []
    has_req_sections = any(s.key in ("requirements", "nice_to_have", "responsibilities") for s in sections)

    def add(line: str, kind: str) -> None:
        body = line.lstrip("• ").strip()
        if ";" in body and len(body) > 80:  # "wykształcenie wyższe; staż 2 lata; prawo jazdy"
            for part in body.split(";"):
                if part.strip():
                    add("• " + part.strip(), kind)
            return
        if not body or (len(body.split()) < 2 and not line.startswith("•")):
            return
        if kind == "must" and nice_set.any(body) and not must_set.any(body):
            kind = "nice"
        req = Requirement(body, kind, sorted({h.skill.name for h in tax.find_in_stems(textnorm.stems(body))}))
        match = _YEARS_RE.search(textnorm.fold(body))
        if match and ("doswiad" in textnorm.fold(body) or "experience" in textnorm.fold(body)):
            req.years = int(match.group(1))
        langs = language_mentions(body)
        if langs.get("polski", "x") in (None, "native"):
            langs.pop("polski")  # "native Polish" is not a language requirement worth checking
        if langs:
            name, level = next(iter(langs.items()))
            req.language = (name, level)
        requirements.append(req)

    for sec in sections:
        if sec.key == "requirements":
            for line in sec.lines:
                add(line, "must")
        elif sec.key == "nice_to_have":
            for line in sec.lines:
                add(line, "nice")
        elif sec.key == "responsibilities":
            for line in sec.lines:
                add(line, "duty")
        elif not has_req_sections and sec.key == "header":
            for line in sec.lines:
                if line.startswith("•") or must_set.any(line) or nice_set.any(line):
                    add(line, "must")
    if not requirements:  # unstructured offer: every sentence counts as a requirement
        for sent in textnorm.sentences(doc.text):
            add(sent, "must")

    # job-board noise ("similar offers") and legal notices are not part of the offer
    main_text = "\n".join(s.text for s in sections if s.key not in ("other", "clause"))
    knockouts: dict[str, dict] = {}
    for key, spec in lex["knockouts"].items():
        for req in requirements:
            if req.kind == "duty" and key != "availability_weekends":
                continue
            if regex_any(spec["job"], req.text):
                if key in knockouts and knockouts[key]["kind"] == "must":
                    continue
                knockouts[key] = {"label": spec["label"], "line": req.text, "kind": req.kind,
                                  "soft": bool(spec.get("soft")) or req.kind == "nice",
                                  "document_only": bool(spec.get("document_only"))}
                req.knockouts.append(key)
        if key not in knockouts and key in ("no_criminal_record", "sanepid", "portfolio_required"):
            # often stated outside the requirement list, e.g. in "required documents"
            if regex_any(spec["job"], main_text):
                knockouts[key] = {"label": spec["label"], "line": "", "kind": "must",
                                  "soft": False,
                                  "document_only": bool(spec.get("document_only"))}

    languages: dict[str, str | None] = {}
    for req in requirements:
        if req.language and req.kind != "duty":
            languages[req.language[0]] = req.language[1]
    years = [r.years for r in requirements if r.years and r.kind == "must"]
    lines = [ln for ln in doc.text.split("\n") if ln.strip()]
    title = ""
    for line in lines[:8]:
        # "Stanowisko: X", "... ogłasza nabór na stanowisko: X", "Position: X"
        match = re.search(r"(?i)(?:stanowisk\w*|position|role)\s*:\s*(.{3,120})", line)
        if match:
            title = match.group(1).strip(" .\"„”")
            break
    if not title:
        title = _title_by_typography(doc)
    if not title:
        title = next((ln for ln in lines[:5] if 2 <= len(ln.split()) <= 12 and not _ORG_RE.search(ln)),
                     lines[0] if lines else "")
    org = next((ln.strip() for ln in lines[:15] if _ORG_RE.search(ln) and len(ln) < 160), "")
    # "Dyrektor Środowiskowego Domu Samopomocy w X ogłasza nabór na stanowisko ..." -> the institution
    org = re.split(r"(?i)\s+(?:ogłasza|oglasza|poszukuje|zatrudni|szuka|zaprasza|announces|is hiring)\b", org)[0]
    org = re.sub(r"(?i)^(?:dyrektor\w*|kierownik\w*|prezes\w*|zarząd\w*)\s+", "", org).strip()
    # level: from the title, else from job-board meta lines ("umowa o pracę · specjalista")
    meta = "\n".join(ln for ln in lines[:10] if re.search(r"(?i)[·|]|etat|poziom|level|seniority", ln))
    return JobOffer(doc, title.strip(" -•"), org, requirements, knockouts, languages,
                    max(years) if years else None, _seniority_of(title) or _seniority_of(meta),
                    tax.domain_profile(doc.text), sections, detect_language(doc.text))


# ---------------------------------------------------------------- cover letter


@dataclass
class LetterAnalysis:
    """Parsed cover letter."""

    doc: Document
    paragraphs: list[str]
    words: int
    language: str


def analyze_letter(doc: Document) -> LetterAnalysis:
    """Split a cover letter into paragraphs (blank lines or long lines)."""
    paras, cur = [], []
    for line in doc.text.split("\n"):
        if not line.strip():
            if cur:
                paras.append(" ".join(cur))
                cur = []
        else:
            cur.append(line.strip())
    if cur:
        paras.append(" ".join(cur))
    return LetterAnalysis(doc, paras, len(doc.text.split()), detect_language(doc.text))
