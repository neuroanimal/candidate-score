"""Cover letter metrics. The offer is optional; without it personalisation and
alignment are reported as unknown."""

from __future__ import annotations

import re

from .. import similarity, textnorm
from ..analysis import CVAnalysis, JobOffer, LetterAnalysis
from ..lexicon import phrase_set, taxonomy
from ..profiles import Profile
from .base import MetricResult, band, make, pct, ramp, short
from .content import _numbers

_ORG_GENERIC = {"fundacja", "fundacji", "stowarzyszenie", "centrum", "osrodek", "dom", "kultury", "szkola", "sp", "z", "o",
                "muzeum", "galeria", "agencja", "studio", "spolka", "firma", "instytut", "teatr", "biblioteka"}


def _org_tokens(job: JobOffer | None) -> set[str]:
    if job is None or not job.organisation:
        return set()
    return {textnorm.stem(t) for t in textnorm.tokens(job.organisation) if len(t) >= 4 and t not in _ORG_GENERIC}


_NAMED_ORG = re.compile(
    r"\b(?:Fundacj\w*|Stowarzyszeni\w*|Muze(?:um|ach|ów|ami|a)|Galeri\w*|Centrum|Centrach|Dom\w*\s+Kultury|"
    r"Ośrod\w*|Teatr\w*|Bibliote\w*|Agencj\w*|Studi\w*|Instytu\w*|Szko\w*|Przedszko\w*|Firm\w*|Spółk\w*)"
    r"\s+((?:[„\"]?[A-ZĄĆĘŁŃÓŚŹŻ][\w-]+[”\"]?\s*){1,4})")


def foreign_organisations(letter_text: str, job: JobOffer | None, cv: CVAnalysis) -> list[str]:
    """Organisation names in the letter that are neither in the offer nor in the CV.

    The classic copy-paste mistake: a letter reused from another application still
    names the previous employer-to-be. Previous employers are in the CV, so they are
    not reported.
    """
    if job is None:
        return []
    known = set(textnorm.stems(job.doc.text)) | set(textnorm.stems(cv.doc.text))
    out = []
    for m in _NAMED_ORG.finditer(letter_text):
        name_stems = [s for s in textnorm.stems(m.group(1)) if len(s) >= 4]
        if name_stems and not any(s in known for s in name_stems):
            out.append(" ".join(m.group(0).split()))
    return list(dict.fromkeys(out))


def letter_personalization(letter: LetterAnalysis, cv: CVAnalysis, job: JobOffer | None, _p: Profile) -> MetricResult:
    """Organisation, position and addressee named."""
    text = letter.doc.text
    letter_stems = set(textnorm.stems(text))
    checks = {}
    org = _org_tokens(job)
    if org:
        checks["nazwa organizacji"] = bool(org & letter_stems)
    if job is not None and job.title:
        title = [s for s in textnorm.content_stems(job.title, textnorm.GENERIC_REQUIREMENT_WORDS) if len(s) >= 4]
        tst = set(title)
        # "na stanowisko grafika" names the role even if "specjalista ds. social media" is omitted
        checks["nazwa stanowiska"] = bool(tst) and (len(tst & letter_stems) / len(tst) >= 0.5 or title[0] in letter_stems)
    checks["imienny adresat"] = bool(re.search(r"(Szanown[ay] Pan[iu]?|Dear (Mr|Ms|Mrs|Dr))\s+[A-ZĄĆĘŁŃÓŚŹŻ]", text))
    checks["odniesienie do działalności organizacji"] = bool(re.search(
        r"(?i)(wasz|państwa|pani|pana|your)\w*\s+(projekt|działaln|misj|wystaw|warsztat|program|ofert|podejści|work|mission)", text))
    if job is None:
        score = 100 * sum(checks.values()) / len(checks) * 0.8
    else:
        weights = {"nazwa organizacji": 35, "nazwa stanowiska": 35, "imienny adresat": 10,
                   "odniesienie do działalności organizacji": 20}
        total = sum(weights[k] for k in checks)
        score = 100 * sum(weights[k] for k, v in checks.items() if v) / total
    foreign = foreign_organisations(text, job, cv)
    evidence = [("OK: " if v else "BRAK: ") + k for k, v in checks.items()]
    suggestion = ("Wymień nazwę organizacji i stanowiska oraz jedną konkretną rzecz z ich działalności, "
                  "która Cię przyciąga." if score < 70 else "")
    if foreign:
        score = min(score, 25)
        evidence.insert(0, "UWAGA: list wymienia organizację spoza oferty i CV: " + "; ".join(foreign[:3]))
        suggestion = ("Sprawdź, czy list nie jest skopiowany z innej aplikacji - wymienia: " + "; ".join(foreign[:3]) +
                      ". " + suggestion).strip()
    return make("letter_personalization", score, value=f"{sum(checks.values())}/{len(checks)}",
                evidence=evidence, suggestion=suggestion, critical=False)


def letter_jd_alignment(letter: LetterAnalysis, _cv: CVAnalysis, job: JobOffer | None, _p: Profile) -> MetricResult:
    """Share of must-haves addressed in the letter."""
    if job is None or not job.must:
        return make("letter_jd_alignment", None, value="brak oferty")
    index = similarity.LineIndex(textnorm.sentences(letter.doc.text))
    names = taxonomy().names(letter.doc.text)
    hit, ev = 0, []
    for req in job.must:
        ok = bool(set(req.skills) & names)
        best = index.best(req.text, top=1)
        if not ok and best and best[0][2] >= 0.45:
            ok = True
        hit += ok
        ev.append(("OK: " if ok else "BRAK: ") + short(req.text, 80))
    ratio = hit / len(job.must)
    return make("letter_jd_alignment", ramp(ratio, 0.1, 0.6), value=f"{hit}/{len(job.must)}", evidence=ev,
                suggestion="Odnieś się w liście do 2-3 najważniejszych wymagań, każde z krótkim przykładem z praktyki."
                if ratio < 0.5 else "")


def letter_cv_consistency(letter: LetterAnalysis, cv: CVAnalysis, _job: JobOffer | None, _p: Profile) -> MetricResult:
    """Claims in the letter backed by the CV, and the letter is not a copy of it."""
    tax = taxonomy()
    l_sk = {s for s in tax.names(letter.doc.text) if tax.skills[s].kind != "soft"}
    c_sk = tax.names(cv.doc.text)
    backed = l_sk & c_sk
    ratio = len(backed) / len(l_sk) if l_sk else 1.0
    sents = textnorm.sentences(letter.doc.text)
    cv_lines = [ln for ln in cv.doc.text.split("\n") if len(ln.split()) >= 5]
    index = similarity.LineIndex(cv_lines)
    copied = [s for s in sents if len(s.split()) >= 6 and (b := index.best(s, top=1)) and b[0][2] >= 0.85]
    copy_share = len(copied) / max(1, len(sents))
    score = 0.7 * ramp(ratio, 0.4, 0.95) + 0.3 * ramp(copy_share, 0.5, 0.1)
    return make("letter_cv_consistency", score, value=f"{len(backed)}/{len(l_sk)} potwierdzonych, kopiowanych {pct(copy_share)}",
                evidence=[f"Niepotwierdzone w CV: {', '.join(sorted(l_sk - c_sk)) or '-'}"]
                + [f"prawie dosłownie z CV: {short(s, 80)}" for s in copied[:2]],
                suggestion="Umiejętności, o których piszesz w liście, pokaż też w CV; w liście nie powtarzaj CV - "
                           "opowiedz jedną historię z efektem." if score < 70 else "")


def letter_structure(letter: LetterAnalysis, _cv: CVAnalysis, _job: JobOffer | None, _p: Profile) -> MetricResult:
    """Greeting, opening naming the role, body, closing, call to action, signature."""
    text = letter.doc.text
    paras = [p for p in letter.paragraphs if len(p.split()) >= 3]
    first = " ".join(paras[:2]) if paras else ""
    checks = {
        "powitanie": phrase_set("letter", "greeting").any("\n".join(text.split("\n")[:12])),
        "wstęp ze stanowiskiem/celem": bool(re.search(r"(?i)stanowisk|ofert|ogłoszen|rekrutac|position|role|vacancy", first)),
        "co najmniej 2 akapity treści": len([p for p in paras if len(p.split()) >= 25]) >= 2,
        "zakończenie grzecznościowe": phrase_set("letter", "closing").any(text),
        "zaproszenie do kontaktu (CTA)": phrase_set("letter", "call_to_action").any(text),
        "podpis": bool(re.search(r"\n\s*[A-ZĄĆĘŁŃÓŚŹŻ][a-ząćęłńóśźż]+\s+[A-ZĄĆĘŁŃÓŚŹŻ][a-ząćęłńóśźż-]+\s*$", text.strip()[-120:])),
    }
    score = 100 * sum(checks.values()) / len(checks)
    return make("letter_structure", score, value=f"{sum(checks.values())}/{len(checks)}",
                evidence=[("OK: " if v else "BRAK: ") + k for k, v in checks.items()],
                suggestion="Uzupełnij: " + ", ".join(k for k, v in checks.items() if not v) if score < 100 else "")


def letter_length(letter: LetterAnalysis, _cv: CVAnalysis, _job: JobOffer | None, p: Profile) -> MetricResult:
    """About 180-450 words and one page."""
    lo, hi = p.target("letter_words_min", 180), p.target("letter_words_max", 450)
    score = band(letter.words, lo, hi, 60, 900) - (25 if letter.doc.page_count > 1 else 0)
    return make("letter_length", score, value=f"{letter.words} słów, {letter.doc.page_count} str.",
                suggestion="Skróć list do jednej strony." if letter.words > hi else
                ("List jest bardzo krótki - dodaj przykład z praktyki." if letter.words < lo else ""))


def letter_specificity(letter: LetterAnalysis, _cv: CVAnalysis, _job: JobOffer | None, _p: Profile) -> MetricResult:
    """Examples, numbers and names vs generic phrases."""
    text = letter.doc.text
    nums = _numbers(text)
    tax_hits = taxonomy().names(text)
    proper = re.findall(r"(?<![.!?]\s)(?<!^)\b[A-ZĄĆĘŁŃÓŚŹŻ][a-ząćęłńóśźż]{3,}", text)
    generic = phrase_set("letter", "generic").find(text) + phrase_set("cliches").find(text)
    score = 30 + min(25, 8 * len(nums)) + min(25, 4 * len(tax_hits)) + min(20, 3 * len(set(proper))) - min(40, 7 * len(generic))
    return make("letter_specificity", score, value=f"{len(nums)} liczb, {len(generic)} ogólników",
                evidence=[f"Liczby: {', '.join(nums[:6]) or '-'}", f"Ogólniki: {', '.join(sorted(set(generic))[:8]) or '-'}"],
                suggestion="Zastąp ogólniki jednym konkretnym przykładem: co zrobiłaś, dla kogo, z jakim efektem."
                if score < 70 else "")


def letter_motivation(letter: LetterAnalysis, _cv: CVAnalysis, job: JobOffer | None, _p: Profile) -> MetricResult:
    """Why this organisation and this role."""
    text = letter.doc.text
    mot = phrase_set("letter", "motivation").find(text)
    org = _org_tokens(job)
    org_ctx = bool(org & set(textnorm.stems(text)))
    score = min(70, 15 * len(set(mot))) + (30 if org_ctx else 0)
    return make("letter_motivation", score, value=f"{len(set(mot))} sygnałów motywacji",
                evidence=[f"Sygnały: {', '.join(sorted(set(mot))[:8]) or '-'}",
                          "Organizacja wymieniona w kontekście motywacji" if org_ctx else "Brak nazwy organizacji"],
                suggestion="Napisz, dlaczego właśnie ta organizacja (jej projekt, odbiorcy, wartości) i ta rola."
                if score < 60 else "")


def letter_tone(letter: LetterAnalysis, _cv: CVAnalysis, _job: JobOffer | None, _p: Profile) -> MetricResult:
    """Polish courtesy forms capitalised, no colloquial greeting."""
    text = letter.doc.text
    if letter.language != "pl":
        formal = phrase_set("letter", "closing").any(text)
        return make("letter_tone", 100 if formal else 60, value="EN")
    issues = []
    lower_forms = re.findall(r"(?<![.!?]\s)\b(państwa|państwu|pani|panu|pana)\b", text)
    if lower_forms:
        issues.append(f"formy grzecznościowe małą literą ({len(lower_forms)})")
    if re.search(r"(?im)^\s*(hej|cześć|siema|hello)\b", text):
        issues.append("potoczne powitanie")
    if text.count("!") > 2:
        issues.append("dużo wykrzykników")
    if re.search(r"(?i)\bwitam\b", text):
        issues.append("'Witam' bywa odbierane jako niegrzeczne - lepiej 'Szanowni Państwo' / 'Dzień dobry'")
    score = 100 - 25 * len(issues)
    return make("letter_tone", score, value=f"{len(issues)} uwag", evidence=issues or ["Forma poprawna"],
                suggestion="; ".join(issues) if issues else "")


def letter_readability(letter: LetterAnalysis, _cv: CVAnalysis, _job: JobOffer | None, _p: Profile) -> MetricResult:
    """Sentence and paragraph length."""
    sents = textnorm.sentences(letter.doc.text)
    if not sents:
        return make("letter_readability", None)
    avg = sum(len(s.split()) for s in sents) / len(sents)
    long_par = [p for p in letter.paragraphs if len(p.split()) > 120]
    score = band(avg, 8, 22, 0, 45) - 15 * len(long_par)
    return make("letter_readability", score, value=f"{avg:.0f} słów/zdanie",
                evidence=[f"Akapitów > 120 słów: {len(long_par)}"])


ALL = [letter_personalization, letter_jd_alignment, letter_cv_consistency, letter_structure, letter_length,
       letter_specificity, letter_motivation, letter_tone, letter_readability]
