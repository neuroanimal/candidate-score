"""Batch scoring of many applications, CV-versus-offer matrices and outcome analysis.

Input layouts (first that applies):

1. ``manifest.yaml`` (or ``.csv``) in the folder - explicit list of applications::

       - id: 2026-03_fundacja
         cv: cv/cv_terapia.pdf
         letter: listy/list_fundacja.pdf
         job: oferty/oferta_fundacja.pdf
         outcome: rozmowa           # optional: rozmowa / odrzucona / brak odpowiedzi / zatrudniona ...

2. one sub-folder per application, files recognised by name or content
   (``outcome.txt`` / ``wynik.txt`` / ``meta.yaml`` with the outcome are optional);

3. a flat folder of CVs, letters and offers - every CV is scored against every
   offer (pool mode).
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from . import calibrate, evaluation, textnorm
from .extract import Document, extract
from .lexicon import phrase_set
from .report import STATUS_NAMES, _cls, _env, _fmt, write
from .scoring import Report, score

DOC_SUFFIXES = (".pdf", ".docx", ".txt")
POSITIVE = ("rozmow", "zaprosz", "interview", "invited", "zatrudn", "hired", "przyj", "oferta pracy", "offer", "etap 2",
            "drugi etap", "spotkani", "sukces")
NEGATIVE = ("odrzuc", "odmow", "reject", "brak odpowiedzi", "no response", "cisza", "ghost", "negatyw", "nie przesz")
GRADES = {"zatrudn": 3, "hired": 3, "offer": 3, "oferta pracy": 3, "przyj": 3, "rozmow": 2, "interview": 2,
          "zaprosz": 2, "invited": 2, "spotkani": 2, "etap 2": 2, "drugi etap": 2}


@dataclass
class Application:
    """One application: a CV, optionally a letter and the offer it was sent for."""

    id: str
    cv: Path
    job: Path | None = None
    letter: Path | None = None
    outcome: str | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def label(self) -> int | None:
        """1 = positive outcome (invited/hired), 0 = negative, None = unknown."""
        if not self.outcome:
            return None
        o = textnorm.fold(self.outcome)
        if any(n in o for n in NEGATIVE):
            return 0
        if any(p in o for p in POSITIVE):
            return 1
        return None

    @property
    def grade(self) -> int | None:
        """Graded outcome for NDCG: 3 hired/offer, 2 interview, 0 negative."""
        if self.label is None:
            return None
        o = textnorm.fold(self.outcome or "")
        return next((g for k, g in GRADES.items() if k in o), 1) if self.label else 0


# ------------------------------------------------------------ classification

_NAME_HINTS = {
    "cv": re.compile(r"(?i)(^|[^a-z])(cv|zyciorys|życiorys|resume|résumé|curriculum)([^a-z]|$)"),
    "letter": re.compile(r"(?i)(list|motywac|cover|letter|lm)([^a-z]|$)"),
    "job": re.compile(r"(?i)(oferta|ogloszen|ogłoszen|job|offer|stanowisk|rekrutac|nabor|nabór|praca|posting|vacancy|ad)([^a-z]|$)"),
}


def classify(doc: Document, filename: str = "") -> tuple[str, dict[str, float]]:
    """Guess whether a document is a ``cv``, ``letter`` or ``job`` offer."""
    scores = {"cv": 0.0, "letter": 0.0, "job": 0.0}
    for kind, rx in _NAME_HINTS.items():
        if rx.search(Path(filename).stem.replace("_", " ").replace("-", " ")):
            scores[kind] += 4.0
    text = doc.text[:8000]
    for kind in scores:
        scores[kind] += min(6.0, phrase_set("doc_type_cues", kind).count(text) * 0.8)
    if phrase_set("letter", "greeting").any("\n".join(text.split("\n")[:15])) and phrase_set("letter", "closing").any(text):
        scores["letter"] += 4
    years = len(re.findall(r"\b(19|20)\d{2}\b", text))
    scores["cv"] += min(4.0, years * 0.3)
    if re.search(r"(?im)^\s*(wymagania|oferujemy|zakres obowiązków|obowiązki|requirements)\s*:?\s*$", text):
        scores["job"] += 4
    if "moich danych osobowych" in textnorm.fold(text):
        scores["cv"] += 2
        scores["letter"] += 1
    best = max(scores, key=scores.get)
    return best, scores


def _read_outcome(folder: Path) -> str | None:
    for name in ("outcome.txt", "wynik.txt", "status.txt", "rezultat.txt"):
        p = folder / name
        if p.is_file():
            return p.read_text(encoding="utf-8", errors="replace").strip() or None
    for name in ("meta.yaml", "meta.yml"):
        p = folder / name
        if p.is_file():
            data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
            return data.get("outcome") or data.get("wynik")
    return None


class DocCache:
    """Extract every file once (OCR is slow)."""

    def __init__(self, ocr: str = "auto"):
        self.ocr = ocr
        self.docs: dict[Path, Document] = {}

    def get(self, path: Path) -> Document:
        """Extracted document."""
        path = Path(path).resolve()
        if path not in self.docs:
            self.docs[path] = extract(path, ocr=self.ocr)
        return self.docs[path]


def _docs_in(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in DOC_SUFFIXES
                  and p.name.lower() not in ("outcome.txt", "wynik.txt", "status.txt", "rezultat.txt", "notes.txt"))


class _Pool(dict):
    """Documents by type, de-duplicated by content (the same CV copied into several folders)."""

    def __init__(self):
        super().__init__(cv=[], letter=[], job=[])
        self._seen: set[str] = set()

    def add(self, kind: str, path: Path | None) -> None:
        """Add a file unless an identical one is already there."""
        if path is None:
            return
        digest = hashlib.md5(Path(path).read_bytes()).hexdigest()  # identity, not security
        if digest not in self._seen:
            self._seen.add(digest)
            self[kind].append(path)


def discover(root: Path, cache: DocCache) -> tuple[list[Application], dict[str, list[Path]]]:
    """Find applications and the pool of documents by type."""
    root = Path(root)
    pool = _Pool()
    apps: list[Application] = []
    manifest = next((root / n for n in ("manifest.yaml", "manifest.yml", "manifest.csv") if (root / n).is_file()), None)
    if manifest is not None:
        if manifest.suffix == ".csv":
            with open(manifest, encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
        else:
            rows = yaml.safe_load(manifest.read_text(encoding="utf-8")) or []
        for i, row in enumerate(rows, 1):
            paths = {k: (root / row[k]) if row.get(k) else None for k in ("cv", "job", "letter")}
            app = Application(str(row.get("id") or f"app{i:03d}"), paths["cv"], paths["job"], paths["letter"],
                              row.get("outcome") or row.get("wynik"))
            apps.append(app)
            for kind, path in (("cv", app.cv), ("job", app.job), ("letter", app.letter)):
                pool.add(kind, path)
        return apps, pool
    subdirs = sorted(d for d in root.iterdir() if d.is_dir() and not d.name.startswith((".", "_")))
    for folder in subdirs:
        files = _docs_in(folder)
        if not files:
            continue
        typed: dict[str, list[tuple[float, Path]]] = {"cv": [], "letter": [], "job": []}
        for f in files:
            kind, sc = classify(cache.get(f), f.name)
            typed[kind].append((sc[kind], f))
        if not typed["cv"]:
            continue
        pick = {k: (sorted(v, key=lambda x: -x[0])[0][1] if v else None) for k, v in typed.items()}
        app = Application(folder.name, pick["cv"], pick["job"], pick["letter"], _read_outcome(folder))
        for k, v in typed.items():
            if len(v) > 1:
                app.notes.append(f"{len(v)} plików typu {k} - użyto {pick[k].name}")
        apps.append(app)
        for k in ("cv", "letter", "job"):
            pool.add(k, pick[k])
    for f in _docs_in(root):
        kind, _ = classify(cache.get(f), f.name)
        pool.add(kind, f)
    return apps, pool


# ------------------------------------------------------------------- running


@dataclass
class BatchResult:
    """Everything produced by a batch run."""

    apps: list[Application]
    reports: dict[str, Report]
    matrix: dict[tuple[str, str], Report]
    pool: dict[str, list[Path]]
    analysis: dict


def run(root: Path, profile: str | None = "auto", ocr: str = "auto", cross: bool = True,
        max_cross: int = 1600, today=None) -> BatchResult:
    """Score all applications, and (``cross``) every CV against every offer."""
    cache = DocCache(ocr)
    apps, pool = discover(root, cache)
    reports: dict[str, Report] = {}
    for app in apps:
        reports[app.id] = score(cache.get(app.cv), cache.get(app.job) if app.job else None,
                                cache.get(app.letter) if app.letter else None, profile=profile, today=today)
    matrix: dict[tuple[str, str], Report] = {}
    if cross and pool["cv"] and pool["job"] and len(pool["cv"]) * len(pool["job"]) <= max_cross:
        for cv in pool["cv"]:
            for job in pool["job"]:
                matrix[(str(cv), str(job))] = score(cache.get(cv), cache.get(job), None, profile=profile, today=today)
    return BatchResult(apps, reports, matrix, pool, analyse_outcomes(apps, reports))


def analyse_outcomes(apps: list[Application], reports: dict[str, Report]) -> dict:
    """Which scores and metrics separate positive from negative outcomes."""
    labeled = [(a, reports[a.id]) for a in apps if a.label is not None and a.id in reports]
    out: dict = {"n_labeled": len(labeled), "n_positive": sum(a.label for a, _ in labeled)}
    if len(labeled) < 4 or out["n_positive"] in (0, len(labeled)):
        out["note"] = ("Za mało zakończonych rekrutacji z wynikiem (potrzeba >= 4, w tym pozytywne i negatywne), "
                       "żeby sprawdzić, które metryki przewidują zaproszenie.")
        return out
    labels = [a.label for a, _ in labeled]
    comp = {}
    for key in ("overall", "job_fit", "cv_quality", "letter", "evidence_strength"):
        vals = [r.composites.get(key) for _, r in labeled]
        if all(v is not None for v in vals):
            comp[key] = {"auc": evaluation.roc_auc(vals, labels), "r_pb": evaluation.point_biserial(vals, labels)}
            grades = [a.grade for a, _ in labeled]
            comp[key]["ndcg"] = evaluation.ndcg_at_k(vals, grades)
    out["composites"] = comp
    metric_ids = sorted({m.id for _, r in labeled for m in r.metrics})
    rows = []
    for mid in metric_ids:
        pairs = [(r.metric(mid).score, a.label) for a, r in labeled if r.metric(mid) and r.metric(mid).score is not None]
        if len(pairs) >= 4 and len({y for _, y in pairs}) == 2:
            xs, ys = zip(*pairs)
            rows.append({"metric": mid, "label": labeled[0][1].metric(mid).label, "n": len(pairs),
                         "auc": evaluation.roc_auc(xs, ys), "r_pb": evaluation.point_biserial(xs, ys)})
    rows.sort(key=lambda r: -abs((r["auc"] if r["auc"] == r["auc"] else 0.5) - 0.5))
    out["metrics"] = rows
    if len(labeled) >= calibrate.MIN_SAMPLES:
        try:
            model = calibrate.fit([r for _, r in labeled], labels)
            out["model"] = {"loo_auc": model["loo_auc"], "importance": model["importance"][:15], "n": model["n"]}
        except ValueError as exc:
            out["model_note"] = str(exc)
    return out


# ------------------------------------------------------------------- output


def _short(path) -> str:
    return Path(path).name if path else "–"


def write_batch(result: BatchResult, out_dir: Path) -> list[Path]:
    """Write per-application reports and the batch index (HTML + Markdown + JSON)."""
    out_dir = Path(out_dir)
    (out_dir / "apps").mkdir(parents=True, exist_ok=True)
    written = []
    rows = []
    for app in result.apps:
        rep = result.reports[app.id]
        safe = re.sub(r"[^\w.-]+", "_", app.id)
        written += write(rep, out_dir / "apps" / safe)
        rows.append({"id": app.id, "href": f"apps/{safe}.html", "cv": _short(app.cv), "job": _short(app.job),
                     "job_title": rep.job.title if rep.job else "", "letter": _short(app.letter),
                     "profile": rep.profile.label, "outcome": app.outcome or "",
                     "scores": {k: _fmt(rep.composites.get(k)) for k in ("overall", "job_fit", "cv_quality", "letter")},
                     "cls": _cls(rep.overall), "confidence": f"{rep.confidence:.0f}",
                     "critical": bool(rep.critical), "notes": app.notes,
                     "verdict": rep.verdict.decision if rep.verdict else "",
                     "recs": [r.label for r in rep.recommendations[:3]]})
    cvs = [str(p) for p in result.pool["cv"]]
    jobs = [str(p) for p in result.pool["job"]]
    matrix_rows = []
    best_for_job = []
    if result.matrix:
        for cv in cvs:
            cells = []
            for job in jobs:
                rep = result.matrix.get((cv, job))
                v = rep.composites.get("job_fit") if rep else None
                cells.append({"v": _fmt(v), "cls": _cls(v), "crit": bool(rep and rep.critical)})
            matrix_rows.append({"cv": _short(cv), "cells": cells})
        for job in jobs:
            ranked = sorted(((result.matrix[(cv, job)].composites.get("job_fit") or 0, cv) for cv in cvs
                             if (cv, job) in result.matrix), reverse=True)
            if ranked:
                rep = result.matrix[(ranked[0][1], job)]
                best_for_job.append({"job": _short(job), "title": rep.job.title if rep.job else "",
                                     "best": _short(ranked[0][1]), "score": f"{ranked[0][0]:.0f}",
                                     "runner": _short(ranked[1][1]) if len(ranked) > 1 else "–",
                                     "runner_score": f"{ranked[1][0]:.0f}" if len(ranked) > 1 else "–"})
    ctx = {"rows": rows, "jobs": [_short(j) for j in jobs], "matrix": matrix_rows, "best": best_for_job,
           "analysis": result.analysis, "status_names": STATUS_NAMES}
    html = _env().get_template("batch.html.j2").render(**ctx)
    p = out_dir / "index.html"
    p.write_text(html, encoding="utf-8")
    written.append(p)
    p = out_dir / "index.md"
    p.write_text(_batch_markdown(ctx), encoding="utf-8")
    written.append(p)
    p = out_dir / "results.json"
    p.write_text(json.dumps({"applications": [dict(r, report=f"apps/{re.sub(r'[^\\w.-]+', '_', r['id'])}.json")
                                              for r in rows],
                             "matrix": {f"{_short(c)} | {_short(j)}": r.composites for (c, j), r in result.matrix.items()},
                             "analysis": result.analysis}, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    written.append(p)
    return written


def _batch_markdown(ctx: dict) -> str:
    out = ["# Zestawienie aplikacji", "",
           "| Aplikacja | CV | Oferta | Ogólny | Dopasowanie | Jakość CV | List | Przesiew | Wynik rekrutacji |",
           "|---|---|---|---|---|---|---|---|---|"]
    for r in ctx["rows"]:
        s = r["scores"]
        out.append(f"| {r['id']} | {r['cv']} | {r['job_title'] or r['job']} | {s['overall']} | {s['job_fit']} | "
                   f"{s['cv_quality']} | {s['letter']} | {r['verdict']} | {r['outcome']} |")
    if ctx["best"]:
        out += ["", "## Najlepsza wersja CV dla każdej oferty", "", "| Oferta | Najlepsze CV | Dopasowanie | Drugie | Dopasowanie |",
                "|---|---|---|---|---|"]
        for b in ctx["best"]:
            out.append(f"| {b['title'] or b['job']} | {b['best']} | {b['score']} | {b['runner']} | {b['runner_score']} |")
    an = ctx["analysis"]
    out += ["", "## Co przewiduje wynik rekrutacji", ""]
    if an.get("note"):
        out.append(an["note"])
    for m in an.get("metrics", [])[:15]:
        out.append(f"- {m['label']}: AUC {m['auc']:.2f}, r = {m['r_pb']:.2f} (n = {m['n']})")
    return "\n".join(out) + "\n"
