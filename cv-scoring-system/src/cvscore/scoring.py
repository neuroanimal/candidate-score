"""Scoring engine: runs all metrics, builds composite scores, confidence and
recommendations ranked by how much they would raise the overall score."""

from __future__ import annotations

import datetime as dt
import inspect
from dataclasses import dataclass, field
from pathlib import Path

from . import __version__, rewrite, similarity
from .analysis import CVAnalysis, JobOffer, LetterAnalysis, analyze_cv, analyze_letter, parse_job
from .extract import Document, extract
from .lexicon import taxonomy
from .metrics import content, document, fit, letter
from .metrics.base import REGISTRY, MetricResult
from .profiles import Profile, resolve

EVIDENCE_METRICS = ("skill_evidence", "achievement_ratio", "quantification_ratio", "ownership", "scope_signals",
                    "must_have_coverage", "experience_relevance", "specificity")
KNOCKOUT_CAP = 40.0


@dataclass
class Recommendation:
    """A suggestion with the estimated effect on the overall score."""

    metric: str
    label: str
    text: str
    gain: float
    current: float | None
    critical: bool = False


@dataclass
class Report:
    """Full scoring result for one application (CV + optional offer + optional letter)."""

    cv: CVAnalysis
    job: JobOffer | None
    letter: LetterAnalysis | None
    profile: Profile
    metrics: list[MetricResult]
    composites: dict[str, float | None]
    confidence: float
    recommendations: list[Recommendation] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    created: str = ""
    verdict: Verdict | None = None
    rewrites: list = field(default_factory=list)  # rewrite.Rewrite drafts for weak bullets

    def metric(self, metric_id: str) -> MetricResult | None:
        """Metric by id."""
        return next((m for m in self.metrics if m.id == metric_id), None)

    @property
    def overall(self) -> float | None:
        """Overall score."""
        return self.composites.get("overall")

    @property
    def critical(self) -> list[MetricResult]:
        """Metrics flagged as critical (failed knock-outs)."""
        return [m for m in self.metrics if m.critical]

    def to_dict(self) -> dict:
        """JSON-serialisable form."""
        cv = self.cv
        return {
            "version": __version__,
            "created": self.created,
            "cv": {"path": cv.doc.path, "pages": cv.doc.page_count, "words": cv.words, "language": cv.language,
                   "producer": cv.doc.producer, "ocr_pages": cv.doc.ocr_pages,
                   "sections": [{"key": s.key, "title": s.title, "lines": len(s.lines)} for s in cv.sections],
                   "entries": [{"section": e.section, "title": e.title, "start": e.start.isoformat(),
                                "end": e.end.isoformat(), "ongoing": e.ongoing, "months": e.months}
                               for e in cv.entries],
                   "contact": vars(cv.contact), "languages": cv.languages,
                   "skills": sorted({h.skill.name for h, _, _ in cv.skill_hits})},
            "job": None if self.job is None else {
                "path": self.job.doc.path, "title": self.job.title, "organisation": self.job.organisation,
                "seniority": self.job.seniority, "years_required": self.job.years_required,
                "languages": self.job.languages, "knockouts": self.job.knockouts, "domains": self.job.domains,
                "requirements": [{"kind": r.kind, "text": r.text, "skills": r.skills} for r in self.job.requirements]},
            "letter": None if self.letter is None else {"path": self.letter.doc.path, "words": self.letter.words},
            "profile": {"name": self.profile.name, "label": self.profile.label, "mix": self.profile.mix},
            "composites": {k: None if v is None else round(v, 1) for k, v in self.composites.items()},
            "confidence": round(self.confidence, 1),
            "metrics": [m.to_dict() for m in self.metrics],
            "recommendations": [vars(r) for r in self.recommendations],
            "verdict": None if self.verdict is None else vars(self.verdict),
            "rewrites": [vars(r) for r in self.rewrites],
            "warnings": self.warnings,
        }


@dataclass
class Verdict:
    """Recruiter-style screening decision (the "Przesiew" checklist from the method notes)."""

    decision: str  # "shortlista", "dopytaj", "odrzuć"
    reasons: list[str]
    sentence: str


def screening_verdict(metrics: list[MetricResult], job: JobOffer | None) -> Verdict | None:
    """Knock-outs -> must-haves -> relevance of the best role -> level, in that order.

    * a failed hard knock-out rejects;
    * must-have evidence: any requirement at 0, or an average below 2.0 of 3, prevents the
      shortlist (below 1.5: reject);
    * experience relevance below 1.6 of 3 rejects, 1.6-2.39 means "ask more";
    * a level two steps away from the offer prevents the shortlist.
    """
    if job is None:
        return None
    get = {m.id: m for m in metrics}
    reasons, decision = [], "shortlista"

    def lower(to: str) -> None:
        nonlocal decision
        order = ["shortlista", "dopytaj", "odrzuć"]
        decision = max(decision, to, key=order.index)

    ko = get.get("knockouts")
    if ko is not None and ko.critical:
        lower("odrzuć")
        reasons.append("niespełniony warunek konieczny")
    must = get.get("must_have_coverage")
    if must is not None and must.score is not None:
        avg = must.score / 100 * 3
        zeros = sum(1 for e in must.evidence if e.startswith("[0]"))
        if avg < 1.5:
            lower("odrzuć")
        elif avg < 2.0 or zeros:
            lower("dopytaj")
        reasons.append(f"wymagania obowiązkowe {avg:.1f}/3" + (f", bez dowodu: {zeros}" if zeros else ""))
    rel = get.get("experience_relevance")
    if rel is not None and rel.score is not None:
        val = rel.score / 100 * 3
        if val < 1.6:
            lower("odrzuć")
        elif val < 2.4:
            lower("dopytaj")
        reasons.append(f"trafność najlepszej roli {val:.2f}/3")
    sen = get.get("seniority_fit")
    if sen is not None and sen.score is not None and sen.score < 50:
        lower("dopytaj")
        reasons.append(f"poziom: {sen.value}")
    best_role = rel.evidence[0].split(" [")[0].rstrip(".…") if rel is not None and rel.evidence else "-"
    sentence = (f"{decision.capitalize()}: " + "; ".join(reasons) +
                (f". Najtrafniejsza rola: {best_role}." if best_role != "-" else "."))
    return Verdict(decision, reasons, sentence)


def _call(func, *args, job=None):
    params = inspect.signature(func).parameters
    if "job" in params:
        return func(*args, job=job)
    return func(*args)


def _weighted(metrics: list[MetricResult]) -> float | None:
    known = [m for m in metrics if m.score is not None and m.weight > 0]
    total = sum(m.weight for m in known)
    return sum(m.score * m.weight for m in known) / total if total else None


def composites(metrics: list[MetricResult], profile: Profile, has_job: bool, has_letter: bool) -> dict[str, float | None]:
    """Composite scores and the overall score."""
    by_group = {g: [m for m in metrics if m.group == g] for g in ("document", "content", "fit", "letter")}
    comp: dict[str, float | None] = {
        "document": _weighted(by_group["document"]),
        "content": _weighted(by_group["content"]),
        "cv_quality": _weighted(by_group["document"] + by_group["content"]),
        "job_fit": _weighted(by_group["fit"]) if has_job else None,
        "letter": _weighted(by_group["letter"]) if has_letter else None,
        "evidence_strength": _weighted([m for m in metrics if m.id in EVIDENCE_METRICS]),
    }
    gw = profile.group_weights or {"job_fit": 0.5, "cv_quality": 0.3, "letter": 0.2}
    parts = [(comp["cv_quality"], gw.get("cv_quality", 0.3))]
    if has_job:
        parts.append((comp["job_fit"], gw.get("job_fit", 0.5)))
    if has_letter:
        parts.append((comp["letter"], gw.get("letter", 0.2)))
    parts = [(v, w) for v, w in parts if v is not None]
    overall = sum(v * w for v, w in parts) / sum(w for _, w in parts) if parts else None
    if overall is not None and any(m.critical for m in metrics):
        overall = min(overall, KNOCKOUT_CAP)
    comp["overall"] = overall
    return comp


def _confidence(cv: CVAnalysis, metrics: list[MetricResult]) -> float:
    conf = 100.0
    pages = cv.doc.pages
    if pages:
        conf -= 30 * cv.doc.ocr_pages / len(pages)
        conf -= 40 * max((p.garbage for p in pages if p.method == "text"), default=0.0)
        low = [p.ocr_confidence for p in pages if p.ocr_confidence is not None and p.ocr_confidence < 75]
        conf -= 10 * len(low)
    weighted = [m for m in metrics if m.weight > 0]
    unknown = sum(m.weight for m in weighted if m.score is None) / max(1e-9, sum(m.weight for m in weighted))
    conf -= 30 * unknown
    if not cv.jobs:
        conf -= 15
    if len(cv.statements) < 5:
        conf -= 10
    if cv.words < 60:
        conf = min(conf, 20.0)  # almost nothing was read - the scores say little
    return max(10.0, min(100.0, conf))


def recommend(metrics: list[MetricResult], profile: Profile, has_job: bool, has_letter: bool,
              limit: int = 12) -> list[Recommendation]:
    """Suggestions ranked by the overall-score gain if the metric reached 90."""
    base = composites(metrics, profile, has_job, has_letter)["overall"] or 0.0
    recs = []
    for m in metrics:
        if not m.suggestion or (m.score is not None and m.score >= 85):
            continue
        trial = [MetricResult(x.id, x.label, x.group, 90.0 if x is m else x.score, weight=x.weight,
                              critical=False if x is m else x.critical) for x in metrics]
        new = composites(trial, profile, has_job, has_letter)["overall"] or 0.0
        recs.append(Recommendation(m.id, m.label, m.suggestion, round(new - base, 1), m.score, m.critical))
    recs.sort(key=lambda r: (not r.critical, -r.gain))
    return recs[:limit]


def score(cv_doc: Document, job_doc: Document | None = None, letter_doc: Document | None = None,
          profile: str | None = "auto", today: dt.date | None = None) -> Report:
    """Score already-extracted documents."""
    today = today or dt.date.today()
    cv = analyze_cv(cv_doc, today)
    job = parse_job(job_doc) if job_doc is not None else None
    let = analyze_letter(letter_doc) if letter_doc is not None else None
    domains = job.domains if job is not None and job.domains else taxonomy().domain_profile(cv_doc.text)
    prof = resolve(profile, domains)
    results: list[MetricResult] = []
    for func in document.ALL + content.ALL:
        results.append(_call(func, cv, prof, job=job))
    if job is not None:
        ctx = fit.FitContext(cv, job, prof)
        for func in fit.ALL:
            results.append(func(ctx, prof))
    if let is not None:
        for func in letter.ALL:
            results.append(func(let, cv, job, prof))
    for m in results:
        m.weight = REGISTRY[m.id][2] * prof.weight(m.id)
    comp = composites(results, prof, job is not None, let is not None)
    warnings = list(cv_doc.warnings) + (list(job_doc.warnings) if job_doc else []) + (
        list(letter_doc.warnings) if letter_doc else [])
    if cv.words < 60:
        warnings.insert(0, f"Z CV odczytano tylko {cv.words} słów - sprawdź plik poleceniem 'cvscore extract'.")
    if job is not None and not job.requirements:
        warnings.append("Nie rozpoznano wymagań w ofercie - dopasowanie jest przybliżone.")
    if not similarity.embeddings_available():
        warnings.append("Embeddingi semantyczne wyłączone (brak modelu w pamięci podręcznej) - używam TF-IDF i BM25.")
    return Report(cv, job, let, prof, results, comp, _confidence(cv, results),
                  recommend(results, prof, job is not None, let is not None), warnings,
                  dt.datetime.now().isoformat(timespec="seconds"), screening_verdict(results, job),
                  rewrite.suggest(cv))


def score_files(cv_path: str | Path, job_path: str | Path | None = None, letter_path: str | Path | None = None,
                profile: str | None = "auto", ocr: str = "auto", today: dt.date | None = None) -> Report:
    """Extract and score files."""
    cv_doc = extract(cv_path, ocr=ocr)
    job_doc = extract(job_path, ocr=ocr) if job_path else None
    letter_doc = extract(letter_path, ocr=ocr) if letter_path else None
    return score(cv_doc, job_doc, letter_doc, profile=profile, today=today)
