"""Command line interface.

    cvscore score CV.pdf [--job OFERTA.pdf] [--letter LIST.pdf] [--out reports/nazwa]
    cvscore batch FOLDER [--out reports/batch]
    cvscore compare CV1.pdf CV2.pdf [--job OFERTA.pdf]   # which CV version is better and why
    cvscore extract PLIK.pdf            # what the tool reads from a PDF (debug InDesign export)
    cvscore job OFERTA.pdf              # parsed requirements and knock-outs
    cvscore calibrate FOLDER --out model.json
    cvscore metrics | cvscore profiles
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

from . import __version__


def _today(value: str | None) -> dt.date | None:
    return dt.date.fromisoformat(value) if value else None


def cmd_score(args) -> int:
    """Score one application."""
    from . import report, scoring  # pylint: disable=import-outside-toplevel

    rep = scoring.score_files(args.cv, args.job, args.letter, profile=args.profile, ocr=args.ocr,
                              today=_today(args.today))
    out = args.out or str(Path("reports") / Path(args.cv).stem)
    paths = report.write(rep, out, formats=tuple(args.formats.split(",")))
    print(f"Profil: {rep.profile.label}")
    for key in ("overall", "job_fit", "cv_quality", "evidence_strength", "letter"):
        val = rep.composites.get(key)
        if val is not None:
            print(f"  {key:18} {val:5.1f}")
    print(f"  {'confidence':18} {rep.confidence:5.1f}")
    if rep.critical:
        print("  UWAGA: niespełniony warunek konieczny (knock-out)")
    if rep.verdict:
        print(f"Przesiew: {rep.verdict.sentence}")
    print("Najważniejsze poprawki:")
    for r in rep.recommendations[:5]:
        print(f"  +{r.gain:4.1f}  {r.label}: {r.text[:110]}")
    for p in paths:
        print(f"Zapisano: {p.resolve()}")
    return 0


def cmd_compare(args) -> int:
    """Compare CV versions."""
    from . import report, scoring  # pylint: disable=import-outside-toplevel

    reps = [scoring.score_files(cv, args.job, None, profile=args.profile, ocr=args.ocr, today=_today(args.today))
            for cv in args.cvs]
    names = [Path(cv).name for cv in args.cvs]
    html, md = report.compare(reps, names)
    base = Path(args.out)
    base.parent.mkdir(parents=True, exist_ok=True)
    base.with_suffix(".html").write_text(html, encoding="utf-8")
    base.with_suffix(".md").write_text(md, encoding="utf-8")
    key = "job_fit" if args.job else "cv_quality"
    for name, rep in sorted(zip(names, reps), key=lambda x: -(x[1].composites.get(key) or 0)):
        print(f"  {rep.composites.get(key) or 0:5.1f}  {name}")
    print(f"Zapisano: {base.with_suffix('.html').resolve()}\nZapisano: {base.with_suffix('.md').resolve()}")
    return 0


def cmd_batch(args) -> int:
    """Score a folder of applications."""
    from . import batch  # pylint: disable=import-outside-toplevel

    res = batch.run(Path(args.folder), profile=args.profile, ocr=args.ocr, cross=not args.no_cross,
                    today=_today(args.today))
    paths = batch.write_batch(res, Path(args.out))
    print(f"Aplikacji: {len(res.apps)}, CV: {len(res.pool['cv'])}, ofert: {len(res.pool['job'])}, "
          f"listów: {len(res.pool['letter'])}, par w macierzy: {len(res.matrix)}")
    index = next(p for p in paths if p.name == "index.html")
    print(f"Zestawienie: {index.resolve()}")
    print(f"Plików zapisano: {len(paths)} w {Path(args.out).resolve()}")
    return 0


def cmd_extract(args) -> int:
    """Show extracted text and diagnostics."""
    from .extract import extract  # pylint: disable=import-outside-toplevel
    from .sections import split_sections, timeline  # pylint: disable=import-outside-toplevel

    doc = extract(args.file, ocr=args.ocr)
    print(f"# {doc.path}  ({doc.kind}, {doc.page_count} str., program: {doc.producer or '-'})")
    for p in doc.pages:
        print(f"  strona {p.number}: {p.method}, {p.chars} znaków, kolumn {p.columns}, grafika {p.image_coverage:.0%}"
              + (f", OCR {p.ocr_confidence}%" if p.ocr_confidence is not None else ""))
    for w in doc.warnings:
        print(f"  ! {w}")
    print(f"  linki: {', '.join(doc.all_links()) or '-'}")
    secs = split_sections(doc)
    print("  sekcje: " + ", ".join(f"{s.key}({len(s.lines)})" for s in secs))
    for e in timeline(secs, _today(args.today)):
        print(f"  [{e.section}] {e.start:%m.%Y}-{'teraz' if e.ongoing else format(e.end, '%m.%Y')} {e.title[:80]}")
    print("-" * 80)
    print(doc.text)
    return 0


def cmd_job(args) -> int:
    """Show the parsed offer."""
    from .analysis import parse_job  # pylint: disable=import-outside-toplevel
    from .extract import extract  # pylint: disable=import-outside-toplevel
    from .profiles import detect  # pylint: disable=import-outside-toplevel

    job = parse_job(extract(args.file, ocr=args.ocr))
    print(f"Stanowisko: {job.title}\nOrganizacja: {job.organisation or '-'}\nPoziom: {job.seniority or '-'}, "
          f"staż: {job.years_required or '-'}, języki: {job.languages or '-'}")
    print(f"Profil: {detect(job.domains).label}")
    for kind in ("must", "nice", "duty"):
        for r in job.requirements:
            if r.kind == kind:
                print(f"  [{kind:4}] {r.text}" + (f"  -> {', '.join(r.skills)}" if r.skills else ""))
    for key, info in job.knockouts.items():
        print(f"  knock-out: {info['label']} ({key}{', miękki' if info['soft'] else ''}"
              f"{', dokument' if info['document_only'] else ''})")
    return 0


def cmd_calibrate(args) -> int:
    """Learn metric importance from outcomes."""
    from . import batch, calibrate  # pylint: disable=import-outside-toplevel

    res = batch.run(Path(args.folder), profile=args.profile, ocr=args.ocr, cross=False, today=_today(args.today))
    pairs = [(res.reports[a.id], a.label) for a in res.apps if a.label is not None]
    try:
        model = calibrate.fit([p[0] for p in pairs], [p[1] for p in pairs])
    except ValueError as exc:
        print(f"Nie można skalibrować: {exc}")
        return 1
    path = calibrate.save(model, Path(args.out))
    print(f"n = {model['n']} (pozytywnych {model['n_positive']}), AUC (leave-one-out) = {model['loo_auc']:.2f}")
    for item in model["importance"][:12]:
        print(f"  {item['coef']:+.2f}  {item['metric']}")
    print(f"Zapisano: {path.resolve()}")
    return 0


def cmd_metrics(_args) -> int:
    """List all metrics."""
    from .metrics.base import GROUPS, REGISTRY  # pylint: disable=import-outside-toplevel

    for group, label in GROUPS.items():
        print(f"\n{label}")
        for mid, (lbl, grp, weight, desc) in REGISTRY.items():
            if grp == group:
                print(f"  {mid:26} w={weight:<4} {lbl}: {desc}")
    print(f"\nRazem: {len(REGISTRY)} metryk")
    return 0


def cmd_profiles(_args) -> int:
    """List profiles."""
    from .profiles import get_profile, profile_names  # pylint: disable=import-outside-toplevel

    for name in profile_names():
        p = get_profile(name)
        print(f"{name:14} {p.label}\n{'':14} sekcje: {', '.join(p.expected_sections)}")
        if p.metric_weights:
            print(f"{'':14} wagi: " + ", ".join(f"{k} x{v}" for k, v in p.metric_weights.items()))
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Argument parser."""
    parser = argparse.ArgumentParser(prog="cvscore", description="Ocena CV, listów motywacyjnych i dopasowania do ofert (bez LLM).")
    parser.add_argument("--version", action="version", version=f"cvscore {__version__}")
    sub = parser.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--ocr", choices=("auto", "always", "never"), default="auto", help="OCR stron bez tekstu (domyślnie auto)")
        p.add_argument("--profile", default="auto", help="auto | tech | creative_arts | social_media | events | education | "
                                                          "therapy | general, można łączyć: creative_arts+therapy")
        p.add_argument("--today", help="data odniesienia RRRR-MM-DD (testy, ocena wsteczna)")

    p = sub.add_parser("score", help="oceń jedno CV (opcjonalnie z ofertą i listem)")
    p.add_argument("cv")
    p.add_argument("--job", "-j")
    p.add_argument("--letter", "-l")
    p.add_argument("--out", "-o", help="ścieżka bazowa raportu (bez rozszerzenia)")
    p.add_argument("--formats", default="html,md,json")
    common(p)
    p.set_defaults(func=cmd_score)

    p = sub.add_parser("compare", help="porównaj kilka wersji CV (opcjonalnie dla jednej oferty)")
    p.add_argument("cvs", nargs="+")
    p.add_argument("--job", "-j")
    p.add_argument("--out", "-o", default="reports/porownanie")
    common(p)
    p.set_defaults(func=cmd_compare)

    p = sub.add_parser("batch", help="oceń folder aplikacji")
    p.add_argument("folder")
    p.add_argument("--out", "-o", default="reports/batch")
    p.add_argument("--no-cross", action="store_true", help="bez macierzy każde CV x każda oferta")
    common(p)
    p.set_defaults(func=cmd_batch)

    p = sub.add_parser("extract", help="pokaż, co narzędzie odczytuje z pliku")
    p.add_argument("file")
    common(p)
    p.set_defaults(func=cmd_extract)

    p = sub.add_parser("job", help="pokaż rozpoznane wymagania oferty")
    p.add_argument("file")
    common(p)
    p.set_defaults(func=cmd_job)

    p = sub.add_parser("calibrate", help="naucz wagi metryk na wynikach rekrutacji")
    p.add_argument("folder")
    p.add_argument("--out", "-o", default="model.json")
    common(p)
    p.set_defaults(func=cmd_calibrate)

    sub.add_parser("metrics", help="lista metryk").set_defaults(func=cmd_metrics)
    sub.add_parser("profiles", help="lista profili").set_defaults(func=cmd_profiles)
    return parser


def main(argv=None) -> int:
    """Entry point."""
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
