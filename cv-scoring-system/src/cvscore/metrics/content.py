"""Content metrics: achievements, numbers, ownership, scale, evidence for skills,
career shape, education, courses and artistic output. None of them needs a job
offer."""

from __future__ import annotations

import re
import statistics

from .. import textnorm
from ..analysis import CVAnalysis
from ..lexicon import lexicons, phrase_set, taxonomy
from ..profiles import Profile
from .base import MetricResult, make, pct, ramp, short

# numbers that are not results: years, dates, phone numbers, postcodes, ordinals in titles
_DATEISH = re.compile(r"\b(?:19|20)\d{2}\b|\b\d{1,2}[./]\d{1,2}[./]\d{2,4}\b|\b\d{2}-\d{3}\b|\+?\d[\d\s-]{8,}\d")
_NUMBER = re.compile(r"(?<![\w.])(?:\d+[\d\s.,]*\d|\d)(?:\s?(?:%|proc|tys|k\b|mln|zł|pln|eur|osób|os\.|"
                     r"uczestnik|dzieci|uczni|edycj|wystaw|prac|obraz|godzin|h\b|warsztat|zaję|grup|klient|"
                     r"obserwuj|follower|post|projekt|wydarze|imprez|wolontariusz|osob|lat|miesi))?", re.IGNORECASE)
_SCOPE = re.compile(r"(?i)(\d[\d\s.,]*)\s*(?:\+\s*)?(tys\.?|k|mln)?\s*(osób|osoby|os\.|uczestnik\w*|uczestniczek|dzieci|"
                    r"uczni\w*|seniorów|podopieczn\w*|pacjent\w*|grup\w*|klient\w*|obserwuj\w*|followers?|wolontariusz\w*|"
                    r"edycj\w*|wydarze\w*|imprez\w*|warsztat\w*|zaję\w*|wystaw\w*|prac\b|obraz\w*|ilustracj\w*|projekt\w*|"
                    r"zł|pln|eur|euro|people|participants|students|members|events|editions|team)")
_PROPER = re.compile(r"(?<!^)(?<![.!?•]\s)\b[A-ZĄĆĘŁŃÓŚŹŻ][a-ząćęłńóśźż]{2,}")


def _numbers(text: str) -> list[str]:
    clean = _DATEISH.sub(" ", text)
    return [m.group(0).strip() for m in _NUMBER.finditer(clean) if any(ch.isdigit() for ch in m.group(0))]


def _is_result(text: str) -> bool:
    return phrase_set("result_words").any(text) or bool(_SCOPE.search(_DATEISH.sub(" ", text)))


def _target(profile: Profile, key: str, default: float) -> float:
    value = profile.target(key, default)
    return default if value is None else value


def achievement_ratio(cv: CVAnalysis, profile: Profile) -> MetricResult:
    """Statements that describe an outcome."""
    st = cv.statements
    if not st:
        return make("achievement_ratio", None, value="brak opisów", suggestion="Dodaj opisy działań przy stanowiskach.")
    ach = [s for s in st if _is_result(s.text) or (_numbers(s.text) and phrase_set("action_verbs").starts(s.text))]
    ratio = len(ach) / len(st)
    tgt = _target(profile, "achievement_ratio", 0.4)
    ev = [f"Osiągnięcia: {len(ach)} z {len(st)} punktów (cel {pct(tgt)})"] + [f"+ {short(s.text)}" for s in ach[:4]]
    weak = [s for s in st if s not in ach][:3]
    ev += [f"- {short(s.text)}" for s in weak]
    return make("achievement_ratio", ramp(ratio, 0, tgt), value=pct(ratio), evidence=ev,
                suggestion="Przy każdym stanowisku dopisz 1-2 efekty: ilu uczestników, ile edycji, co powstało, "
                           "jaki był odbiór (np. 'warsztaty dla 12 grup, 180 dzieci; 3 edycje na prośbę szkoły')."
                if ratio < tgt else "")


def quantification_ratio(cv: CVAnalysis, profile: Profile) -> MetricResult:
    """Statements containing a meaningful number."""
    st = cv.statements
    if not st:
        return make("quantification_ratio", None, value="brak opisów")
    num = [s for s in st if _numbers(s.text)]
    ratio = len(num) / len(st)
    tgt = _target(profile, "quantification_ratio", 0.4)
    ev = [f"Punkty z liczbą: {len(num)} z {len(st)} (cel {pct(tgt)} dla profilu)"] + [f"+ {short(s.text)}" for s in num[:4]]
    return make("quantification_ratio", ramp(ratio, 0, tgt), value=pct(ratio), evidence=ev,
                suggestion="Dodaj liczby tam, gdzie mają sens: liczba uczestników/grup, edycji, prac, wystaw, "
                           "obserwujących, budżet, czas realizacji." if ratio < tgt else "")


def action_verbs(cv: CVAnalysis, _profile: Profile) -> MetricResult:
    """Statements that start with a concrete action."""
    st = cv.statements
    if not st:
        return make("action_verbs", None, value="brak opisów")
    act = [s for s in st if phrase_set("action_verbs").starts(s.text)]
    ratio = len(act) / len(st)
    return make("action_verbs", ramp(ratio, 0.1, 0.6), value=pct(ratio),
                evidence=[f"Od czasownika akcji: {len(act)} z {len(st)}"] + [f"+ {short(s.text)}" for s in act[:3]],
                suggestion="Zaczynaj punkty od działania: 'Prowadziłam...', 'Zaprojektowałam...', 'Zorganizowałam...'."
                if ratio < 0.5 else "")


def duty_language(cv: CVAnalysis, _profile: Profile) -> MetricResult:
    """Share of 'responsible for' style statements."""
    st = cv.statements
    if not st:
        return make("duty_language", None, value="brak opisów")
    duty = [s for s in st if phrase_set("duty_phrases").starts(s.text, window=4)]
    ratio = len(duty) / len(st)
    return make("duty_language", ramp(ratio, 0.6, 0.05), value=pct(ratio),
                evidence=[f"Punkty-obowiązki: {len(duty)} z {len(st)}"] + [f"- {short(s.text)}" for s in duty[:3]],
                suggestion="Zamień 'odpowiedzialna za X' na 'zrobiłam X dla Y, efekt Z'." if ratio > 0.25 else "")


def specificity(cv: CVAnalysis, _profile: Profile) -> MetricResult:
    """Numbers, named things and tools per statement."""
    st = cv.statements
    if not st:
        return make("specificity", None, value="brak opisów")
    tax = taxonomy()
    vals = []
    for s in st:
        signals = sum([bool(_numbers(s.text)), bool(_PROPER.search(s.text)),
                       bool(tax.find_in_stems(textnorm.stems(s.text)))])
        vals.append(signals / 3)
    avg = sum(vals) / len(vals)
    vague = [s for s, v in zip(st, vals) if v == 0][:3]
    return make("specificity", ramp(avg, 0.1, 0.55), value=f"{avg:.2f}",
                evidence=[f"Średnio {avg * 3:.1f} z 3 sygnałów (liczba, nazwa własna, narzędzie/technika) na punkt"]
                + [f"ogólnikowe: {short(s.text)}" for s in vague],
                suggestion="Nazywaj konkretnie: instytucję, projekt, technikę, program, grupę odbiorców." if avg < 0.4 else "")


def ownership(cv: CVAnalysis, _profile: Profile) -> MetricResult:
    """Strong vs weak ownership language."""
    text = "\n".join(s.text for s in cv.statements) or cv.doc.text
    strong = phrase_set("ownership_strong").find(text)
    weak = phrase_set("ownership_weak").find(text)
    if not strong and not weak:
        return make("ownership", 50, value="brak sygnałów", evidence=["Nie znaleziono ani 'prowadziłam', ani 'pomagałam'"])
    ratio = len(strong) / (len(strong) + len(weak))
    score = 0.7 * ramp(ratio, 0.2, 0.8) + 0.3 * ramp(len(strong), 0, 5)
    return make("ownership", score, value=f"{len(strong)} mocnych / {len(weak)} słabych",
                evidence=[f"Mocne: {', '.join(sorted(set(strong))[:8])}", f"Słabe: {', '.join(sorted(set(weak))[:8])}"],
                suggestion="Gdy coś prowadziłaś samodzielnie, napisz to wprost ('prowadziłam', 'autorski program'), "
                           "zamiast 'pomagałam' / 'brałam udział'." if ratio < 0.6 else "")


def scope_signals(cv: CVAnalysis, _profile: Profile) -> MetricResult:
    """Explicit scale: people, groups, editions, budgets, followers."""
    found = []
    for s in cv.statements or [type("S", (), {"text": ln})() for ln in cv.doc.text.split("\n")]:
        for m in _SCOPE.finditer(_DATEISH.sub(" ", s.text)):
            found.append(m.group(0).strip())
    return make("scope_signals", ramp(len(found), 0, 6), value=f"{len(found)} sygnałów skali",
                evidence=[", ".join(found[:12])] if found else ["Brak informacji o skali (liczba osób, grup, edycji, budżet)"],
                suggestion="Podaj skalę: 'grupy 8-12 osób', '6 edycji festiwalu, ok. 2000 uczestników', "
                           "'profil 4,5 tys. obserwujących'." if len(found) < 4 else "")


def star_structure(cv: CVAnalysis, _profile: Profile) -> MetricResult:
    """Action + context + result in one statement."""
    st = cv.statements
    if not st:
        return make("star_structure", None, value="brak opisów")
    full = [s for s in st if phrase_set("action_verbs").starts(s.text) and len(s.text.split()) >= 7 and _is_result(s.text)]
    ratio = len(full) / len(st)
    return make("star_structure", ramp(ratio, 0, 0.35), value=pct(ratio),
                evidence=[f"Pełne (działanie + kontekst + wynik): {len(full)} z {len(st)}"] + [f"+ {short(s.text)}" for s in full[:3]])


def _claimed_and_used(cv: CVAnalysis):
    claimed, used = {}, {}
    for hit, section, entry in cv.skill_hits:
        name = hit.skill.name
        if section in ("skills", "languages", "summary", "header", "interests"):
            claimed.setdefault(name, hit.skill)
        elif section in ("experience", "volunteering", "portfolio", "exhibitions", "awards", "publications",
                         "education", "courses"):
            used.setdefault(name, set()).add(id(entry) if entry is not None else (section, hit.line))
    return claimed, used


def skill_evidence(cv: CVAnalysis, _profile: Profile) -> MetricResult:
    """Evidence-to-claim: listed skills confirmed elsewhere in the CV."""
    claimed, used = _claimed_and_used(cv)
    claimed = {k: v for k, v in claimed.items() if v.kind != "soft"}
    if not claimed:
        if used:
            return make("skill_evidence", 85, value="umiejętności opisane tylko w doświadczeniu",
                        evidence=["Brak osobnej listy umiejętności - wszystkie wynikają z opisów (to dobrze)"])
        return make("skill_evidence", None, value="brak rozpoznanych umiejętności")
    backed = [k for k in claimed if k in used]
    ratio = len(backed) / len(claimed)
    unbacked = sorted(set(claimed) - set(backed))
    return make("skill_evidence", ramp(ratio, 0.1, 0.8), value=f"{len(backed)}/{len(claimed)}",
                evidence=[f"Potwierdzone w doświadczeniu: {', '.join(sorted(backed)[:10])}",
                          f"Tylko zadeklarowane: {', '.join(unbacked[:10])}"],
                suggestion=("Pokaż w opisach stanowisk/projektów, gdzie używałaś: " + ", ".join(unbacked[:6]))
                if ratio < 0.7 else "")


def skill_depth(cv: CVAnalysis, _profile: Profile) -> MetricResult:
    """+1 listed, +2 used in a role, +2 used in two or more roles (max 5 per skill)."""
    claimed, used = _claimed_and_used(cv)
    names = (set(claimed) | set(used)) - {n for n, s in claimed.items() if s.kind == "soft"}
    if not names:
        return make("skill_depth", None, value="brak umiejętności")
    pts = {}
    for name in names:
        p = (1 if name in claimed else 0) + (2 if name in used else 0) + (2 if len(used.get(name, ())) >= 2 else 0)
        pts[name] = p
    avg = sum(pts.values()) / len(pts)
    top = sorted(pts.items(), key=lambda kv: -kv[1])[:8]
    return make("skill_depth", ramp(avg, 1, 4), value=f"{avg:.1f}/5",
                evidence=["Najgłębsze: " + ", ".join(f"{k} {v}/5" for k, v in top)])


_LEVEL = {"junior": 1, "mid": 2, "senior": 3, "lead": 4}


def _entry_level(title: str) -> int | None:
    folded = " " + textnorm.fold(title) + " "
    for level in ("lead", "senior", "junior", "mid"):
        for word in lexicons()["seniority_signals"][level]:
            if re.search(rf"(?<![a-z]){re.escape(word)}(?![a-z])", folded):
                return _LEVEL[level]
    return None


def career_trajectory(cv: CVAnalysis, _profile: Profile) -> MetricResult:
    """Responsibility over time, from title seniority words."""
    jobs = sorted(cv.jobs, key=lambda e: e.start)
    levels = [(e, _entry_level(e.title)) for e in jobs]
    known = [(e, lv) for e, lv in levels if lv is not None]
    if len(known) < 2:
        return make("career_trajectory", None, value="za mało danych",
                    evidence=[f"Stanowisk z datą: {len(jobs)}, z rozpoznanym poziomem: {len(known)}"])
    first = statistics.mean(lv for _, lv in known[: max(1, len(known) // 2)])
    last = statistics.mean(lv for _, lv in known[len(known) // 2:])
    score = 100 if last > first else 75 if last == first else 50
    return make("career_trajectory", score, value="rosnąca" if last > first else "stabilna" if last == first else "malejąca",
                evidence=[f"{e.start.year}: {short(e.title, 60)} (poziom {lv})" for e, lv in known[-5:]])


def tenure_stability(cv: CVAnalysis, _profile: Profile) -> MetricResult:
    """Median months per role and very short episodes (contracts are common in culture)."""
    jobs = cv.jobs
    if not jobs:
        return make("tenure_stability", None, value="brak datowanych stanowisk")
    months = [e.months for e in jobs]
    med = statistics.median(months)
    short_ones = [e for e in jobs if e.months < 6 and not e.ongoing]
    share = len(short_ones) / len(jobs)
    score = 0.6 * ramp(med, 4, 24) + 0.4 * ramp(share, 0.6, 0.1)
    return make("tenure_stability", score, value=f"mediana {med:.0f} mies.",
                evidence=[f"Stanowisk: {len(jobs)}, krótszych niż 6 mies.: {len(short_ones)}"]
                + [f"krótkie: {short(e.title, 60)} ({e.months} mies.)" for e in short_ones[:3]],
                explanation="Mediana czasu na stanowisku. W kulturze i edukacji umowy projektowe są normalne - "
                            "krótkie epizody warto opisać jako 'projekt' lub 'umowa zlecenie, cykl warsztatów'.")


def gap_transparency(cv: CVAnalysis, _profile: Profile) -> MetricResult:
    """Gaps over 6 months between roles, unless explained."""
    from ..sections import gaps  # pylint: disable=import-outside-toplevel

    found = gaps(cv.jobs)
    if not cv.jobs:
        return make("gap_transparency", None, value="brak datowanych stanowisk")
    explained = bool(re.search(r"(?i)urlop (macierzy|rodzicielsk|wychowawcz)|przerwa|opieka nad|freelanc|"
                               r"dzia[lł]alno[sś][cć] w[lł]asn|tw[oó]rczo[sś][cć] w[lł]asn|sabbatical|career break",
                               cv.doc.text))
    big = [g for g in found if g[2] >= 6]
    score = 100 if not big else (85 if explained else max(30, 100 - 20 * len(big)))
    return make("gap_transparency", score, value=f"{len(big)} przerw", evidence=[
        f"Przerwa {g[0]:%m.%Y} - {g[1]:%m.%Y} ({g[2]} mies.)" for g in big[:4]] + (
        ["W CV jest wyjaśnienie przerwy/działalności własnej"] if explained and big else []),
        suggestion="Przerwę można opisać jednym wierszem (np. 'działalność twórcza własna', 'urlop rodzicielski')."
        if big and not explained else "")


def recency(cv: CVAnalysis, _profile: Profile) -> MetricResult:
    """How recent the latest professional activity is."""
    dated = [e for e in cv.entries if e.section in ("experience", "volunteering", "exhibitions", "portfolio")]
    if not dated:
        return make("recency", None, value="brak dat")
    since = min(e.months_since_end(cv.today) for e in dated)
    return make("recency", ramp(since, 30, 6), value="trwa" if since == 0 else f"{since} mies. temu",
                evidence=[f"Najnowsza aktywność: {short(min(dated, key=lambda e: e.months_since_end(cv.today)).title, 70)}"])


_DEGREES = re.compile(r"(?i)\b(mgr|magister\w*|lic\.|licencjat\w*|in[zż]\.?|in[zż]ynier\w*|dr\b|doktor\w*|"
                      r"studia podyplomowe|podyplomow\w*|ma\b|ba\b|msc|bsc|master|bachelor|phd|dyplom\w*)")
_ART_SCHOOLS = re.compile(r"(?i)akademi\w* sztuk|asp\b|uniwersytet artystyczn|wydzia[lł] (grafiki|malarstwa|sztuk|edukacji artystycznej)|"
                          r"liceum plastyczn|zesp[oó][lł] szk[oó][lł] plastycznych|pedagogi|arteterapi|terapi\w* zaj")


def education_strength(cv: CVAnalysis, _profile: Profile) -> MetricResult:
    """Degrees and relevant schools in the education section."""
    edu = "\n".join(s.text for s in cv.sections if s.key == "education")
    if not edu.strip():
        return make("education_strength", 20, value="brak sekcji", suggestion="Dodaj sekcję Wykształcenie (uczelnia, kierunek, lata, dyplom).")
    degrees = sorted({m.group(0).lower() for m in _DEGREES.finditer(edu)})
    schools = sorted({m.group(0).lower() for m in _ART_SCHOOLS.finditer(edu)})
    entries = [e for e in cv.entries if e.section == "education"]
    score = 40 + min(35, 18 * len(degrees)) + min(15, 8 * len(schools)) + (10 if entries else 0)
    return make("education_strength", score, value=", ".join(degrees[:3]) or "bez stopnia",
                evidence=[f"Stopnie/dyplomy: {', '.join(degrees) or '-'}", f"Kierunkowe: {', '.join(schools) or '-'}",
                          f"Datowane wpisy: {len(entries)}"])


def courses_certificates(cv: CVAnalysis, _profile: Profile) -> MetricResult:
    """Courses, trainings and certificates (counted per line), recency bonus."""
    lines = [ln for s in cv.sections if s.key == "courses" for ln in s.lines if len(ln.split()) >= 2]
    edu_courses = [ln for s in cv.sections if s.key == "education" for ln in s.lines
                   if re.search(r"(?i)kurs|szkoleni|certyfikat|warsztat|webinar", ln)]
    items = lines + edu_courses
    recent = [e for e in cv.entries if e.section == "courses" and (cv.today.year - e.end.year) <= 5]
    score = ramp(len(items), 0, 6) * 0.8 + (20 if recent else 0)
    return make("courses_certificates", score, value=f"{len(items)} pozycji",
                evidence=[short(i, 90) for i in items[:6]] or ["Brak kursów i szkoleń"],
                suggestion="Wypisz kursy i szkolenia z datą i organizatorem (np. arteterapia, pierwsza pomoc, "
                           "kurs wychowawcy wypoczynku, Meta Blueprint, Adobe)." if len(items) < 3 else "")


def exhibitions_awards(cv: CVAnalysis, _profile: Profile) -> MetricResult:
    """Artistic and professional output: exhibitions, awards, publications, residencies."""
    keys = ("exhibitions", "awards", "publications")
    lines = [ln for s in cv.sections if s.key in keys for ln in s.lines if len(ln.split()) >= 2]
    elsewhere = [s.text for s in cv.statements if re.search(
        r"(?i)wystaw|wernisa|nagrod|wyró[zż]nieni|konkurs|plener|rezydencj|publikacj|exhibit|award", s.text)]
    total = len(lines) + len(elsewhere)
    return make("exhibitions_awards", ramp(total, 0, 8), value=f"{total} pozycji",
                evidence=[short(x, 90) for x in (lines + elsewhere)[:6]] or ["Brak wystaw, nagród i publikacji"],
                suggestion="Dodaj sekcję 'Wybrane wystawy i nagrody' (rok, tytuł, miejsce) - to dowód poziomu artystycznego."
                if total < 3 else "")


def summary_quality(cv: CVAnalysis, _profile: Profile) -> MetricResult:
    """Short professional summary naming the profession and speciality."""
    sec = cv.section("summary")
    if sec is None or not sec.lines:
        return make("summary_quality", 40, value="brak",
                    suggestion="Dodaj 2-4 zdania profilu: kim jesteś zawodowo, z kim pracujesz, w czym się specjalizujesz, "
                               "1 konkretny wynik.")
    text = sec.text
    words = len(text.split())
    skills = taxonomy().names(text)
    cl = phrase_set("cliches").count(text)
    score = 40 + (20 if 25 <= words <= 110 else 5) + min(30, 10 * len(skills)) + (10 if _numbers(text) else 0) - 8 * cl
    return make("summary_quality", score, value=f"{words} słów, {len(skills)} umiejętności",
                evidence=[short(text, 160), f"Umiejętności w profilu: {', '.join(sorted(skills)[:6]) or '-'}",
                          f"Frazesów: {cl}"])


ALL = [achievement_ratio, quantification_ratio, action_verbs, duty_language, specificity, ownership,
       scope_signals, star_structure, skill_evidence, skill_depth, career_trajectory, tenure_stability,
       gap_transparency, recency, education_strength, courses_certificates, exhibitions_awards, summary_quality]
