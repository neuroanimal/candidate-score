"""Document-level metrics: extraction quality, ATS parseability, structure, form,
Polish formal requirements (RODO clause) and personal data minimisation."""

from __future__ import annotations

import re
from collections import Counter

from .. import textnorm
from ..analysis import CVAnalysis
from ..lexicon import lexicons, phrase_set, regex_any
from ..profiles import Profile
from .base import MetricResult, band, make, pct, ramp, short

SECTION_LABELS = {
    "contact": "Kontakt", "summary": "Profil/O mnie", "experience": "Doświadczenie", "education": "Wykształcenie",
    "courses": "Kursy/certyfikaty", "skills": "Umiejętności", "languages": "Języki", "portfolio": "Portfolio/projekty",
    "exhibitions": "Wystawy", "awards": "Nagrody", "publications": "Publikacje", "volunteering": "Wolontariat",
    "interests": "Zainteresowania", "references": "Referencje", "clause": "Klauzula RODO",
}


def text_extractability(cv: CVAnalysis, _profile: Profile) -> MetricResult:
    """How reliable the text layer is (OCR and broken encodings lower it)."""
    pages = cv.doc.pages
    if cv.doc.kind != "pdf":
        return make("text_extractability", 100, value=cv.doc.kind.upper(), evidence=["Plik tekstowy/DOCX - brak problemu warstwy tekstowej"])
    scores, ev = [], []
    for p in pages:
        if p.method == "text":
            s = ramp(p.garbage, 0.35, 0.03)
            ev.append(f"Strona {p.number}: warstwa tekstowa, {p.chars} znaków, śmieci {pct(p.garbage)}")
        elif p.method == "image":
            ev.append(f"Strona {p.number}: strona graficzna (np. prace w portfolio) - pominięta w ocenie tekstu")
            continue
        elif p.method == "ocr":
            conf = p.ocr_confidence or 0
            s = min(60.0, conf * 0.6)
            ev.append(f"Strona {p.number}: brak użytecznego tekstu, OCR (pewność {conf:.0f}%)")
        else:
            s = 0.0
            ev.append(f"Strona {p.number}: brak tekstu")
        scores.append(s)
    score = sum(scores) / len(scores) if scores else 0
    sugg = ""
    if score < 80:
        sugg = ("Wyeksportuj PDF z InDesign jako 'Adobe PDF (Print)' lub 'Interactive' bez zamiany tekstu na "
                "krzywe (nie używaj Type > Create Outlines), z osadzonymi fontami i opcją 'Create Tagged PDF'. "
                "Nagłówki i nazwisko jako tekst, nie jako obraz.")
    if cv.doc.producer:
        ev.append(f"Program: {cv.doc.producer}")
    return make("text_extractability", score, value=f"{cv.doc.ocr_pages}/{len(pages)} stron z OCR", evidence=ev, suggestion=sugg)


def ats_parseability(cv: CVAnalysis, _profile: Profile) -> MetricResult:
    """Can an ATS find what it needs: sections, dates, contact, sane reading order."""
    checks = {
        "sekcja Doświadczenie rozpoznana": cv.has_section("experience") or cv.has_section("volunteering"),
        "sekcja Wykształcenie rozpoznana": cv.has_section("education"),
        "e-mail w tekście (nie tylko w ikonie)": bool(re.search(r"[\w.+-]+@[\w-]+\.", cv.doc.text)),
        "telefon w tekście": bool(cv.contact.phones),
        "daty stanowisk odczytane": len(cv.jobs) >= 1,
        "co najwyżej 2 kolumny": cv.doc.max_columns <= 2,
        "bez OCR": cv.doc.ocr_pages == 0,
        "mało tekstu w grafice": all(p.image_coverage < 0.5 for p in cv.doc.pages if p.method != "image"),
    }
    ok = sum(checks.values())
    score = 100.0 * ok / len(checks)
    ev = [("OK: " if v else "BRAK: ") + k for k, v in checks.items()]
    sugg = ""
    if score < 80:
        missing = [k for k, v in checks.items() if not v]
        sugg = ("Dla systemów ATS: " + "; ".join(missing) + ". Nagłówki sekcji pisz standardowo "
                "('Doświadczenie zawodowe', 'Wykształcenie', 'Umiejętności'), e-mail i telefon jako zwykły tekst. "
                "Rozważ drugą, prostą (jednokolumnową) wersję CV do formularzy online.")
    return make("ats_parseability", score, value=f"{ok}/{len(checks)}", evidence=ev, suggestion=sugg)


def layout_complexity(cv: CVAnalysis, _profile: Profile) -> MetricResult:
    """Columns and images; informative for creative CVs, important for ATS."""
    cols = cv.doc.max_columns
    cover = max((p.image_coverage for p in cv.doc.pages), default=0.0)
    score = {1: 100, 2: 80, 3: 55}.get(cols, 40) - ramp(cover, 0.15, 0.6) * 0.35
    ev = [f"Maks. kolumn na stronie: {cols}", f"Maks. pokrycie strony grafiką: {pct(cover)}",
          f"Obrazów łącznie: {sum(p.images for p in cv.doc.pages)}"]
    return make("layout_complexity", score, value=f"{cols} kol., grafika {pct(cover)}", evidence=ev,
                suggestion="Układ wielokolumnowy z dużą ilością grafiki jest atrakcyjny dla człowieka, ale słabiej "
                           "czytany przez ATS - wysyłaj go mailem, a do formularzy wersję prostą." if score < 70 else "")


def section_coverage(cv: CVAnalysis, profile: Profile) -> MetricResult:
    """Expected sections for the profile."""
    expected = list(profile.expected_sections)
    if cv.language == "pl" and "clause" not in expected:
        expected.append("clause")
    found = {s.key for s in cv.sections if s.lines}
    if "contact" in expected and (cv.contact.emails or cv.contact.phones):
        found.add("contact")
    if "portfolio" in expected and ({"exhibitions", "awards", "publications"} & found or cv.contact.portfolio):
        found.add("portfolio")
    if "courses" in expected and "courses" not in found and "education" in found:
        edu = "\n".join(s.text for s in cv.sections if s.key == "education")
        if re.search(r"(?i)kurs|szkoleni|certyfikat|warsztat", edu):
            found.add("courses")
    present = [k for k in expected if k in found]
    missing = [k for k in expected if k not in found]
    ev = ["Rozpoznane: " + ", ".join(SECTION_LABELS.get(s.key, s.key) + (f" ('{s.title}')" if s.title else "")
                                     for s in cv.sections if s.key != "header")]
    if missing:
        ev.append("Brak: " + ", ".join(SECTION_LABELS.get(k, k) for k in missing))
    return make("section_coverage", 100.0 * len(present) / max(1, len(expected)),
                value=f"{len(present)}/{len(expected)}", evidence=ev,
                suggestion=("Dodaj lub nazwij czytelnie sekcje: " + ", ".join(SECTION_LABELS.get(k, k) for k in missing))
                if missing else "")


def contact_completeness(cv: CVAnalysis, _profile: Profile) -> MetricResult:
    """E-mail, phone, city, LinkedIn/portfolio."""
    c = cv.contact
    parts = {"e-mail": (40, bool(c.emails)), "telefon": (30, bool(c.phones)), "miejscowość": (10, bool(c.city)),
             "LinkedIn / portfolio / strona": (20, bool(c.linkedin or c.portfolio or c.other_links))}
    score = sum(w for w, ok in parts.values() if ok)
    ev = [("OK: " if ok else "BRAK: ") + k for k, (_, ok) in parts.items()]
    ev += [f"e-mail: {e}" for e in c.emails[:2]] + [f"link: {u}" for u in (c.linkedin + c.portfolio)[:4]]
    return make("contact_completeness", score, value=f"{score}/100", evidence=ev,
                suggestion="Uzupełnij: " + ", ".join(k for k, (_, ok) in parts.items() if not ok) if score < 100 else "")


def length_fit(cv: CVAnalysis, profile: Profile) -> MetricResult:
    """Pages and words relative to experience and profile."""
    pages = cv.doc.text_pages  # picture pages (portfolio appendix) do not count
    words = cv.words
    years = sum(e.months for e in cv.jobs) / 12
    pmax = profile.target("pages_max", 2) + (1 if years > 12 else 0)
    wmin, wmax = profile.target("words_min", 250), profile.target("words_max", 900) * (1.3 if years > 12 else 1.0)
    s_pages = band(pages, profile.target("pages_min", 1), pmax, 0, pmax + 2)
    s_words = band(words, wmin, wmax, wmin * 0.3, wmax * 2.2)
    score = 0.5 * s_pages + 0.5 * s_words
    ev = [f"Stron: {pages} (oczekiwane 1-{pmax:.0f})", f"Słów: {words} (oczekiwane {wmin:.0f}-{wmax:.0f})",
          f"Łączny staż w datowanych wpisach: {years:.1f} lat"]
    if cv.doc.image_pages:
        ev.append(f"Stron graficznych (portfolio) pominiętych: {cv.doc.image_pages}")
    sugg = ""
    if words < wmin:
        sugg = "CV jest bardzo krótkie - dodaj opisy działań i efektów przy każdym stanowisku."
    elif words > wmax or pages > pmax:
        sugg = "CV jest długie - skróć opisy starszych stanowisk, przenieś pełną listę prac/wystaw do portfolio."
    return make("length_fit", score, value=f"{pages} str., {words} słów", evidence=ev, suggestion=sugg)


def readability(cv: CVAnalysis, profile: Profile) -> MetricResult:
    """Sentence/bullet length and share of long words (FOG-PL style)."""
    units = [s.text for s in cv.statements] or textnorm.sentences(cv.doc.text)
    if not units:
        return make("readability", None, value="brak tekstu")
    lens = [len(u.split()) for u in units]
    avg = sum(lens) / len(lens)
    limit = profile.target("bullet_words_max", 28)
    long_share = sum(1 for n in lens if n > limit) / len(lens)
    words = re.findall(r"[^\W\d_]+", " ".join(units))
    hard = sum(1 for w in words if textnorm.syllables(w) >= 5) / max(1, len(words))
    fog = 0.4 * (avg + 100 * hard)
    score = 0.4 * band(avg, 6, 20, 0, 40) + 0.3 * ramp(long_share, 0.5, 0.1) + 0.3 * ramp(fog, 18, 9)
    ev = [f"Średnio {avg:.1f} słów na zdanie/punkt", f"Za długich punktów (> {limit:.0f} słów): {pct(long_share)}",
          f"Trudnych słów (5+ sylab): {pct(hard)}, indeks FOG ~ {fog:.1f}"]
    longest = sorted(units, key=lambda u: -len(u.split()))[:2]
    ev += [f"Najdłuższy: {short(u)}" for u in longest if len(u.split()) > limit]
    return make("readability", score, value=f"{avg:.0f} sł./punkt, FOG {fog:.0f}", evidence=ev,
                suggestion="Dziel długie punkty na krótsze: jedno działanie i jeden efekt w punkcie." if score < 70 else "")


def scannability(cv: CVAnalysis, _profile: Profile) -> MetricResult:
    """Can a recruiter see role, organisation, dates and a result in seconds."""
    exp_lines = [s for s in cv.statements if s.section in ("experience", "volunteering")]
    bullet_share = sum(1 for s in exp_lines if s.is_bullet) / len(exp_lines) if exp_lines else 0.0
    headings = len([s for s in cv.sections if s.key != "header"])
    header = cv.headline()  # name block, also when a sidebar is read first
    from ..lexicon import taxonomy  # pylint: disable=import-outside-toplevel
    title_on_top = bool(taxonomy().find(header)) or bool(re.search(
        r"(?i)(grafi|malar|projektant|specjalist|organizator|pedagog|terapeut|instruktor|animator|developer|"
        r"architect|engineer|scientist|analyst|designer|manager|koordynator|edukator|artyst|ilustrator)", header))
    per_entry = [len(e.lines) for e in cv.jobs]
    avg_lines = sum(per_entry) / len(per_entry) if per_entry else 0
    checks = {
        "zawód/specjalizacja widoczne w nagłówku": title_on_top,
        "co najmniej 3 nagłówki sekcji": headings >= 3,
        "opisy stanowisk w punktach": bullet_share >= 0.5 or not exp_lines,
        "zwięzłe wpisy (<= 7 linii na stanowisko)": avg_lines <= 7,
        "daty przy stanowiskach": bool(cv.jobs),
    }
    score = 100.0 * sum(checks.values()) / len(checks)
    ev = [("OK: " if v else "BRAK: ") + k for k, v in checks.items()]
    ev.append(f"Punktory w opisach doświadczenia: {pct(bullet_share)}; średnio {avg_lines:.1f} linii na stanowisko")
    return make("scannability", score, value=f"{sum(checks.values())}/{len(checks)}", evidence=ev,
                suggestion="Pod imieniem i nazwiskiem wpisz zawód/specjalizację (np. 'Graficzka i edukatorka plastyczna'); "
                           "opisy stanowisk w 2-5 punktach." if score < 80 else "")


_FEM = re.compile(r"\b\w{3,}(?:łam|lam)\b", re.IGNORECASE)
_MASC = re.compile(r"\b\w{3,}(?:łem|lem)\b", re.IGNORECASE)
_NOT_VERBS = {"problem", "problemy", "emblem", "system", "tandem", "totem", "poem", "item", "dylemat", "dilemma",
              "skalam", "mlem", "rodem", "salam", "islam", "program", "gram", "telegram", "diagram", "schemat"}


def _date_style(date_text: str) -> str:
    if re.search(r"(?<!\d)\d{1,2}\s*[./-]\s*\d{4}", date_text):
        return "MM.RRRR"
    if re.search(r"[a-z]{3,}\.?\s+\d{4}", date_text):
        return "miesiąc słownie"
    if re.search(r"\b[ivx]+\s+\d{4}", date_text):
        return "miesiąc rzymski"
    return "RRRR"


def consistency(cv: CVAnalysis, _profile: Profile) -> MetricResult:
    """Date formats, chronological order, grammatical gender and person."""
    issues, ev = [], []
    styles = Counter(_date_style(e.date_text) for e in cv.entries if not e.point)
    if len(styles) > 1:
        issues.append("różne formaty dat")
    ev.append("Formaty dat: " + (", ".join(f"{k} x{v}" for k, v in styles.items()) or "brak"))
    starts = [e.start for e in cv.jobs]
    if len(starts) >= 3:
        desc = sum(1 for a, b in zip(starts, starts[1:]) if a >= b)
        asc = sum(1 for a, b in zip(starts, starts[1:]) if a <= b)
        if desc < len(starts) - 1 and asc < len(starts) - 1:
            issues.append("brak jednolitej kolejności chronologicznej")
        ev.append("Kolejność stanowisk: " + ("od najnowszego" if desc >= asc else "od najstarszego"))
        if asc > desc:
            issues.append("kolejność od najstarszego (zalecana odwrotna chronologia)")
    words = re.findall(r"\b\w+\b", cv.doc.text)
    fem = [w for w in words if _FEM.fullmatch(w) and w.lower() not in _NOT_VERBS]
    masc = [w for w in words if _MASC.fullmatch(w) and w.lower() not in _NOT_VERBS]
    if fem and masc and min(len(fem), len(masc)) >= 1 and cv.language == "pl":
        minority = fem if len(fem) < len(masc) else masc
        issues.append("mieszane formy rodzaju gramatycznego (-łam / -łem)")
        ev.append("Formy mniejszościowe: " + ", ".join(sorted(set(minority))[:6]))
    verbs_first = sum(1 for s in cv.statements if re.match(r"^\W*\w+(?:łam|łem|lam|lem|ałam|ałem)\b", s.text))
    nouns_first = sum(1 for s in cv.statements if re.match(r"^\W*\w+(?:anie|enie|cja|acja)\b", s.text))
    total = max(1, len(cv.statements))
    if verbs_first / total > 0.25 and nouns_first / total > 0.25:
        issues.append("punkty raz od czasownika, raz od rzeczownika")
    ev.append(f"Punkty od czasownika: {verbs_first}, od rzeczownika odczasownikowego: {nouns_first}")
    score = 100 - 22 * len(issues)
    return make("consistency", score, value=f"{len(issues)} problemów", evidence=ev + [f"Problem: {i}" for i in issues],
                suggestion="Ujednolić: " + "; ".join(issues) if issues else "")


def noise_ratio(cv: CVAnalysis, _profile: Profile) -> MetricResult:
    """Clichés, duplicated lines and long hobby lists."""
    cl = phrase_set("cliches").find(cv.doc.text)
    lines = [ln.strip().lower() for ln in cv.doc.text.split("\n") if len(ln.strip()) > 15]
    dup = [ln for ln, c in Counter(lines).items() if c > 1]
    interests = sum(len(s.lines) for s in cv.sections if s.key == "interests")
    per_100 = len(cl) / max(1, cv.words) * 100
    score = 100 - min(50, per_100 * 18) - min(25, 8 * len(dup)) - (10 if interests > 8 else 0)
    ev = [f"Frazesy ({len(cl)}): " + ", ".join(sorted(set(cl))[:10]) if cl else "Frazesów nie znaleziono"]
    if dup:
        ev.append("Powtórzone linie: " + "; ".join(short(d, 60) for d in dup[:3]))
    if interests:
        ev.append(f"Zainteresowania: {interests} linii")
    return make("noise_ratio", score, value=f"{len(cl)} frazesów", evidence=ev,
                suggestion="Zamiast cech ('komunikatywna, kreatywna') pokaż je w działaniu: co zrobiłaś, dla kogo, z jakim efektem."
                if cl else "")


_PL_NO_DIACRITICS = ("doswiadczenie", "umiejetnosci", "wyksztalcenie", "jezyk", "zajec", "srodowisk", "wspolprac",
                     "glown", "osob", "dzialan", "tworz", "rowniez", "miedzy", "wiedze", "zespol", "biegl")


def language_mechanics(cv: CVAnalysis, _profile: Profile) -> MetricResult:
    """Polish diacritics, ALL CAPS, doubled words, mixed languages."""
    text = cv.doc.text
    issues, ev = [], []
    if cv.language == "pl" and cv.words > 120:
        dia = sum(1 for ch in text if ch in textnorm.POLISH_DIACRITICS)
        bare = [w for w in re.findall(r"\b\w+\b", text.lower()) if w.startswith(_PL_NO_DIACRITICS)]
        if dia < cv.words * 0.02 and len(bare) >= 3:
            issues.append("tekst bez polskich znaków")
            ev.append("Przykłady: " + ", ".join(sorted(set(bare))[:5]))
    lines = [ln for ln in text.split("\n") if len(ln) > 25]
    caps = [ln for ln in lines if ln.isupper()]
    if lines and len(caps) / len(lines) > 0.15:
        issues.append("dużo linii pisanych WERSALIKAMI")
    doubled = re.findall(r"\b(\w{3,})\s+\1\b", text, flags=re.IGNORECASE)
    if doubled:
        issues.append("powtórzone słowa")
        ev.append("Powtórzenia: " + ", ".join(sorted(set(doubled))[:5]))
    if cv.language == "pl":
        en_lines = [ln for ln in text.split("\n") if len(ln.split()) >= 5 and
                    sum(1 for t in textnorm.tokens(ln) if t in {"the", "and", "of", "with", "for", "to"}) >= 2]
        if len(en_lines) > 3:
            issues.append("mieszanie języka polskiego i angielskiego")
    space_punct = len(re.findall(r"\s[,.;:](?!\d)", text))
    if space_punct > 3:
        issues.append(f"spacje przed przecinkiem/kropką ({space_punct})")
    score = 100 - 20 * len(issues)
    return make("language_mechanics", score, value=f"{len(issues)} problemów", evidence=ev + [f"Problem: {i}" for i in issues],
                suggestion="Popraw: " + "; ".join(issues) if issues else "")


def rodo_clause(cv: CVAnalysis, _profile: Profile, job=None) -> MetricResult:
    """Polish recruitment practice: current GDPR consent clause."""
    if cv.language != "pl" and not (job and job.language == "pl"):
        return make("rodo_clause", None, value="nie dotyczy (CV po angielsku)")
    pats = lexicons()["rodo_patterns"]
    folded = textnorm.fold(cv.doc.text)
    has_consent = "przetwarzanie" in folded and ("zgod" in folded or "rekrutac" in folded)
    current = any(p in folded for p in pats["current"])
    outdated = any(p in folded for p in pats["outdated"])
    future = any(p in folded for p in pats["future_recruitment"])
    if has_consent and current and not outdated:
        score, value = 100, "aktualna (RODO 2016/679)"
    elif has_consent and outdated:
        score, value = 40, "nieaktualna (ustawa z 1997 r.)"
    elif has_consent:
        score, value = 70, "jest, bez odwołania do RODO"
    else:
        score, value = 0, "brak"
    ev = [f"Klauzula: {value}", "Zgoda na przyszłe rekrutacje: " + ("tak" if future else "nie")]
    sugg = ""
    if score < 100:
        sugg = ("Dodaj na dole CV: 'Wyrażam zgodę na przetwarzanie moich danych osobowych dla potrzeb niezbędnych do "
                "realizacji procesu rekrutacji zgodnie z Rozporządzeniem Parlamentu Europejskiego i Rady (UE) 2016/679 "
                "z dnia 27 kwietnia 2016 r. (RODO).' Jeśli ogłoszenie podaje własną treść klauzuli - użyj tej treści.")
    return make("rodo_clause", score, value=value, evidence=ev, suggestion=sugg)


def personal_data(cv: CVAnalysis, _profile: Profile) -> MetricResult:
    """Data a recruiter does not need (risk of bias, RODO data minimisation)."""
    pats = lexicons()["personal_data"]
    found = {}
    for key, regexes in pats.items():
        hits = regex_any(regexes, cv.doc.text)
        if key == "pesel":
            hits = [h for h in hits if not re.fullmatch(r"(48)?\d{9}", h) and h[2:4] in
                    {f"{m:02d}" for m in list(range(1, 13)) + list(range(21, 33))}]
        if hits:
            found[key] = hits
    labels = {"pesel": "PESEL", "birth_date": "data urodzenia/wiek", "marital_status": "stan cywilny",
              "children": "informacja o dzieciach", "nationality": "obywatelstwo", "health": "stan zdrowia",
              "religion": "wyznanie"}
    penalty = {"pesel": 60, "birth_date": 15, "marital_status": 15, "children": 15, "nationality": 5,
               "health": 25, "religion": 25}
    score = 100 - sum(penalty[k] for k in found)
    photo = _photo_present(cv)
    if photo:
        score -= 5
    ev = [f"Znaleziono: {labels[k]}" for k in found] or ["Brak zbędnych danych osobowych"]
    ev.append("Zdjęcie: " + ("prawdopodobnie tak (informacyjnie - w Polsce częste, nie jest wymagane)" if photo else "nie wykryto"))
    return make("personal_data", score, value=", ".join(labels[k] for k in found) or "OK", evidence=ev,
                suggestion="Usuń z CV: " + ", ".join(labels[k] for k in found) + ". Pracodawca nie może ich wymagać na etapie CV."
                if found else "")


def _photo_present(cv: CVAnalysis) -> bool:
    if not cv.doc.pages:
        return False
    for (x0, y0, x1, y1) in cv.doc.pages[0].image_boxes:
        w, h = x1 - x0, y1 - y0
        if w <= 0 or h <= 0:
            continue
        area, ratio = w * h, h / w
        if 0.008 < area < 0.12 and 0.7 < ratio * 1.414 < 2.2 and y0 < 0.4:
            return True
    return False


def portfolio_presence(cv: CVAnalysis, profile: Profile) -> MetricResult:
    """Links to works - for artists the portfolio is the real CV."""
    c = cv.contact
    links = c.portfolio + [u for u in c.other_links if not re.search(r"(?i)\.(gov|edu)\b", u)]
    tech = "tech" in profile.mix
    if tech:
        links = links + [u for u in c.other_links if "github" in u.lower()]
    mention = bool(re.search(r"(?i)portfolio|behance|instagram|prace do wglądu|na życzenie|przykładowe prace", cv.doc.text))
    if links:
        score, value = 100, f"{len(set(links))} link(i)"
    elif cv.doc.image_pages:
        score, value = 80, f"portfolio w PDF ({cv.doc.image_pages} str.), bez linku"
    elif mention:
        score, value = 45, "wzmianka bez linku"
    else:
        score, value = 0, "brak"
    ev = [f"Link: {u}" for u in list(dict.fromkeys(links))[:6]] or ["Nie znaleziono linków do prac"]
    sugg = ""
    if score < 100:
        sugg = ("Dodaj klikalny link do portfolio (Behance, własna strona, PDF na Dysku Google, Instagram twórczy) "
                "w nagłówku CV - i wpisz adres także jako tekst, bo link schowany pod ikoną znika przy wydruku i w ATS.")
    return make("portfolio_presence", score, value=value, evidence=ev, suggestion=sugg)


ALL = [text_extractability, ats_parseability, layout_complexity, section_coverage, contact_completeness,
       length_fit, readability, scannability, consistency, noise_ratio, language_mechanics, rodo_clause,
       personal_data, portfolio_presence]
