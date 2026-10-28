"""Report writers: HTML (the deliverable), Markdown (easy to read back) and JSON."""

from __future__ import annotations

import json
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

from . import __version__, similarity
from .metrics.base import GROUPS
from .metrics.fit import LEVEL_NAMES
from .scoring import KNOCKOUT_CAP, Report

TEMPLATES = Path(__file__).parent / "templates"
STATUS_NAMES = {"good": "dobrze", "ok": "poprawnie", "weak": "słabo", "bad": "źle", "unknown": "nie zmierzono"}
COMPOSITE_LABELS = [("overall", "Wynik ogólny"), ("job_fit", "Dopasowanie do oferty"), ("cv_quality", "Jakość CV"),
                    ("evidence_strength", "Siła dowodów"), ("letter", "List motywacyjny")]


def _cls(score) -> str:
    if score is None:
        return "s-unknown"
    return "s-good" if score >= 80 else "s-ok" if score >= 60 else "s-weak" if score >= 35 else "s-bad"


def _fmt(score) -> str:
    return "–" if score is None else f"{score:.0f}"


def _env() -> Environment:
    return Environment(loader=FileSystemLoader(str(TEMPLATES)), autoescape=select_autoescape(["html", "j2"]),
                       trim_blocks=True, lstrip_blocks=True)


def _requirements(report: Report) -> tuple[list[dict], list[dict]]:
    """Requirement evidence rows and role relevance rows for the report."""
    if report.job is None:
        return [], []
    from .metrics.fit import FitContext  # pylint: disable=import-outside-toplevel

    ctx = FitContext(report.cv, report.job, report.profile)
    rows = []
    for ev in ctx.evidence:
        level = "?" if ev.level is None else str(ev.level)
        rows.append({"kind": {"must": "wymagane", "nice": "mile widziane", "duty": "obowiązek"}[ev.req.kind],
                     "text": ev.req.text, "level": level, "how": ev.how or LEVEL_NAMES.get(ev.level, ""),
                     "line": ev.line, "cls": _cls(None if ev.level is None else ev.level / 3 * 100)})
    roles = []
    for r in sorted(ctx.relevance, key=lambda x: -x["score"]):
        e = r["entry"]
        roles.append({"title": e.title, "period": f"{e.start:%m.%Y} – {'obecnie' if e.ongoing else format(e.end, '%m.%Y')}",
                      "problem": r["problem"], "context": r["context"], "scale": r["scale"],
                      "ownership": f"{r['ownership']:g}", "fresh": r["fresh"], "score": f"{r['score']:.2f}"})
    return rows, roles


def title_of(report: Report) -> str:
    """Short title: CV file name and offer title."""
    name = Path(report.cv.doc.path).name
    return f"{name} → {report.job.title}" if report.job is not None and report.job.title else name


def render_html(report: Report) -> str:
    """Render the HTML report."""
    groups = []
    for key, label in GROUPS.items():
        ms = [m for m in report.metrics if m.group == key]
        if not ms:
            continue
        comp = {"document": "document", "content": "content", "fit": "job_fit", "letter": "letter"}[key]
        groups.append({"label": label, "metrics": ms, "score": _fmt(report.composites.get(comp)),
                       "cls": _cls(report.composites.get(comp))})
    cards = [{"label": lbl, "value": _fmt(report.composites.get(k)), "cls": _cls(report.composites.get(k))}
             for k, lbl in COMPOSITE_LABELS if report.composites.get(k) is not None]
    cards.append({"label": "Pewność oceny", "value": f"{report.confidence:.0f}", "cls": _cls(report.confidence)})
    reqs, roles = _requirements(report)
    data = report.to_dict()
    return _env().get_template("report.html.j2").render(
        title=title_of(report), cv_path=report.cv.doc.path,
        job_path=report.job.doc.path if report.job else None,
        letter_path=report.letter.doc.path if report.letter else None,
        profile=report.profile.label, created=report.created, version=__version__, cards=cards,
        critical=report.critical, cap=int(KNOCKOUT_CAP), warnings=report.warnings, verdict=report.verdict,
        recommendations=report.recommendations, rewrites=report.rewrites, groups=groups, status_names=STATUS_NAMES,
        requirements=reqs, roles=roles, sections=data["cv"]["sections"], entries=data["cv"]["entries"],
        pages=report.cv.doc.pages, skills=data["cv"]["skills"], cv_text=report.cv.doc.text,
        confidence=f"{report.confidence:.0f}/100", embeddings=similarity.embeddings_available())


def _md_cell(text) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ")


def render_markdown(report: Report) -> str:
    """Render the Markdown report."""
    out = [f"# Ocena aplikacji: {title_of(report)}", "",
           f"- CV: `{report.cv.doc.path}`"]
    if report.job:
        out.append(f"- Oferta: `{report.job.doc.path}` ({report.job.title})")
    if report.letter:
        out.append(f"- List: `{report.letter.doc.path}`")
    out += [f"- Profil: {report.profile.label}", f"- Pewność oceny: {report.confidence:.0f}/100", "",
            "| Wynik | Wartość |", "|---|---|"]
    for k, lbl in COMPOSITE_LABELS:
        if report.composites.get(k) is not None:
            out.append(f"| {lbl} | {report.composites[k]:.0f} |")
    if report.verdict:
        out += ["", f"**Przesiew rekrutera: {report.verdict.decision.upper()}** - {report.verdict.sentence}"]
    if report.critical:
        out += ["", f"**Warunek konieczny niespełniony - wynik ogólny ograniczony do {KNOCKOUT_CAP:.0f}.**"]
    for w in report.warnings:
        out.append(f"\n> {w}")
    out += ["", "## Co poprawić najpierw", "", "| # | Metryka | Obecnie | Zysk | Co zrobić |", "|---|---|---|---|---|"]
    for i, r in enumerate(report.recommendations, 1):
        cur = "–" if r.current is None else f"{r.current:.0f}"
        out.append(f"| {i} | {_md_cell(r.label)} | {cur} | +{r.gain} | {_md_cell(r.text)} |")
    if report.rewrites:
        out += ["", "## Szkice przeredagowania słabych punktów", "", "| Obecnie | Szkic | Dlaczego |", "|---|---|---|"]
        for r in report.rewrites:
            out.append(f"| {_md_cell(r.original)} | {_md_cell(r.draft)} | {_md_cell(r.why)} |")
    for key, label in GROUPS.items():
        ms = [m for m in report.metrics if m.group == key]
        if not ms:
            continue
        out += ["", f"## {label}", "", "| Metryka | Wynik | Status | Wartość | Dowody |", "|---|---|---|---|---|"]
        for m in ms:
            ev = "; ".join(m.evidence[:4])
            out.append(f"| {_md_cell(m.label)} | {_fmt(m.score)} | {STATUS_NAMES[m.status]} | {_md_cell(m.value)} | {_md_cell(ev)} |")
    if report.job is not None:
        reqs, _ = _requirements(report)
        out += ["", "## Wymagania i dowody", "", "| Typ | Wymaganie | Poziom | Gdzie |", "|---|---|---|---|"]
        for r in reqs:
            out.append(f"| {r['kind']} | {_md_cell(r['text'])} | {r['level']} | {_md_cell(r['how'])} |")
    return "\n".join(out) + "\n"


def compare(reports: list[Report], names: list[str]) -> tuple[str, str]:
    """HTML and Markdown comparison of several CV versions (optionally against one offer)."""
    comps = []
    for key, label in COMPOSITE_LABELS + [("document", "Dokument"), ("content", "Treść")]:
        vals = [r.composites.get(key) for r in reports]
        if all(v is None for v in vals):
            continue
        best = max((v for v in vals if v is not None), default=None)
        comps.append({"label": label, "cells": [{"v": _fmt(v), "best": v is not None and v == best and len(vals) > 1}
                                                 for v in vals]})
    rows = []
    for m in reports[0].metrics:
        vals = [(r.metric(m.id).score if r.metric(m.id) else None) for r in reports]
        known = [v for v in vals if v is not None]
        if len(known) < 2:
            continue
        spread = (max(known) - min(known)) * m.weight
        if spread < 1:
            continue
        best = max(known)
        rows.append({"label": m.label, "spread": f"{spread:.0f}",
                     "cells": [{"v": _fmt(v), "best": v == best} for v in vals], "_s": spread})
    rows.sort(key=lambda r: -r["_s"])
    job = reports[0].job.title if reports[0].job is not None else ""
    html = _env().get_template("compare.html.j2").render(names=names, composites=comps, metrics=rows, job=job)
    md = [f"# Porównanie wersji CV{' dla: ' + job if job else ''}", "",
          "| Wynik | " + " | ".join(names) + " |", "|---" * (len(names) + 1) + "|"]
    md += [f"| {c['label']} | " + " | ".join(x["v"] for x in c["cells"]) + " |" for c in comps]
    md += ["", "| Metryka | " + " | ".join(names) + " | Różnica |", "|---" * (len(names) + 2) + "|"]
    md += [f"| {r['label']} | " + " | ".join(x["v"] for x in r["cells"]) + f" | {r['spread']} |" for r in rows]
    return html, "\n".join(md) + "\n"


def write(report: Report, out_base: str | Path, formats=("html", "md", "json")) -> list[Path]:
    """Write the report as ``<out_base>.html/.md/.json``; returns the written paths."""
    base = Path(out_base)
    if base.suffix in (".html", ".md", ".json"):
        base = base.with_suffix("")
    base.parent.mkdir(parents=True, exist_ok=True)
    written = []
    if "html" in formats:
        p = base.with_suffix(".html")
        p.write_text(render_html(report), encoding="utf-8")
        written.append(p)
    if "md" in formats:
        p = base.with_suffix(".md")
        p.write_text(render_markdown(report), encoding="utf-8")
        written.append(p)
    if "json" in formats:
        p = base.with_suffix(".json")
        p.write_text(json.dumps(report.to_dict(), ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        written.append(p)
    return written
