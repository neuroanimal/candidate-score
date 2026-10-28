"""End-to-end scoring, metric behaviour and reports."""

import json

from cvscore import report, scoring
from cvscore.extract import from_text
from cvscore.metrics.base import REGISTRY

GOOD_CV = """Anna Przykładowa
Graficzka i specjalistka ds. social media
anna@example.com | 600 100 200 | 00-001 Warszawa | behance.net/anna

Profil
Graficzka z 6-letnim doświadczeniem w instytucjach kultury. Projektuję identyfikację wizualną i prowadzę social media.

Doświadczenie zawodowe
03.2021 – obecnie
Graficzka i specjalistka ds. social media, Centrum Kultury Przykład
• Zaprojektowałam identyfikację wizualną 4 festiwali i 120 plakatów w Photoshopie, Illustratorze i InDesignie
• Prowadziłam profile na Facebooku i Instagramie; liczba obserwujących wzrosła z 2 tys. do 9 tys.
• Przygotowałam pliki do druku dla 3 drukarni, 0 reklamacji w 2 lata

Wykształcenie
2014 – 2019 Akademia Sztuk Pięknych, Wydział Grafiki, magister

Umiejętności
Adobe Photoshop, Illustrator, InDesign, Canva, Meta Business Suite

Języki
Angielski – B2

Wyrażam zgodę na przetwarzanie moich danych osobowych zgodnie z Rozporządzeniem Parlamentu Europejskiego i Rady (UE) 2016/679 (RODO).
"""

WEAK_CV = """Anna Przykładowa
anna@example.com

Doświadczenie
2021 – obecnie Centrum Kultury
• Odpowiedzialna za grafikę
• Pomagałam przy social media
• Udział w organizacji wydarzeń

Umiejętności
Komunikatywna, kreatywna, sumienna, odporna na stres, Photoshop
"""

OFFER = """Grafik / specjalista ds. social media
Centrum Kultury Przykładowe, Warszawa

Obowiązki:
• projektowanie plakatów i grafik do social media
• prowadzenie profili na Facebooku i Instagramie
• przygotowanie plików do druku

Wymagania:
• min. 2 lata doświadczenia
• bardzo dobra znajomość Adobe Photoshop, Illustrator, InDesign
• portfolio
• prawo jazdy kat. B
"""


def run(cv, job=OFFER, letter=None, today=None):
    """Score text documents."""
    return scoring.score(from_text(cv, "cv.txt"), from_text(job, "job.txt") if job else None,
                         from_text(letter, "letter.txt") if letter else None, today=today)


def test_good_beats_weak(today):
    """Good beats weak."""
    good, weak = run(GOOD_CV, today=today), run(WEAK_CV, today=today)
    assert good.composites["cv_quality"] > weak.composites["cv_quality"] + 15
    assert good.composites["job_fit"] > weak.composites["job_fit"] + 15
    for mid in ("achievement_ratio", "quantification_ratio", "ownership", "noise_ratio", "must_have_coverage"):
        assert good.metric(mid).score > weak.metric(mid).score, mid


def test_all_metrics_present_and_bounded(today):
    """All metrics present and bounded."""
    rep = run(GOOD_CV, letter="Szanowni Państwo,\n\nchciałabym aplikować.\n\nZ wyrazami szacunku\nAnna Przykładowa",
              today=today)
    assert {m.id for m in rep.metrics} == set(REGISTRY)
    for m in rep.metrics:
        assert m.score is None or 0 <= m.score <= 100, m.id
    assert 0 <= rep.confidence <= 100
    json.dumps(rep.to_dict(), default=str)


def test_knockout_caps_overall(today):
    """Knockout caps overall."""
    rep = run(GOOD_CV.replace("behance.net/anna", ""), today=today)
    ko = rep.metric("knockouts")
    assert ko.critical  # no driving licence in the CV
    assert rep.overall <= scoring.KNOCKOUT_CAP
    assert rep.recommendations[0].critical


def test_unknown_is_not_zero(today):
    """Unknown is not zero."""
    rep = run(GOOD_CV, job=None, today=today)
    assert rep.composites["job_fit"] is None
    assert all(m.group != "fit" for m in rep.metrics)
    assert rep.overall == rep.composites["cv_quality"]


def test_profile_detection(today):
    """Profile detection."""
    assert "social_media" in run(GOOD_CV, today=today).profile.mix
    rep = scoring.score(from_text(GOOD_CV), profile="tech", today=today)
    assert rep.profile.name == "tech"


def test_recommendations_sorted_by_gain(today):
    """Recommendations sorted by gain."""
    rep = run(WEAK_CV, today=today)
    gains = [r.gain for r in rep.recommendations if not r.critical]
    assert gains == sorted(gains, reverse=True)
    assert rep.recommendations


def test_reports_written(tmp_path, today):
    """Reports written."""
    rep = run(GOOD_CV, today=today)
    paths = report.write(rep, tmp_path / "r")
    assert {p.suffix for p in paths} == {".html", ".md", ".json"}
    html = (tmp_path / "r.html").read_text(encoding="utf-8")
    assert "<table>" in html and 'lang="pl"' in html and "<caption>" in html
    assert "Co poprawić najpierw" in html


def test_rodo_and_personal_data(today):
    """Rodo and personal data."""
    rep = run(WEAK_CV + "\nPESEL 92030512345\nStan cywilny: panna\n", today=today)
    assert rep.metric("rodo_clause").score == 0
    assert rep.metric("personal_data").score < 50


def test_pdf_end_to_end(sample_pdfs, today):
    """Pdf end to end."""
    rep = scoring.score_files(sample_pdfs / "cv_2col.pdf", sample_pdfs / "oferta_sds_instruktor.pdf",
                              sample_pdfs / "list_sds.pdf", today=today)
    assert "therapy" in rep.profile.mix
    assert rep.composites["job_fit"] > 70 and not rep.critical
    tech = scoring.score_files(sample_pdfs / "cv_tech.pdf", sample_pdfs / "oferta_sds_instruktor.pdf", today=today)
    assert tech.composites["job_fit"] < rep.composites["job_fit"] - 30


def test_english_documents(root, today):
    """English CV and offer are parsed with the same rules."""
    rep = scoring.score_files(root / "samples/text/cv_en.txt", root / "samples/text/oferta_en.txt", today=today)
    assert rep.cv.language == "en" and len(rep.cv.jobs) == 2 and rep.cv.jobs[0].ongoing
    assert rep.metric("rodo_clause").score is None
    assert rep.metric("experience_relevance").score > 80
    assert rep.metric("knockouts").score is None  # only a nice-to-have licence: not a knock-out
    assert not rep.critical


def test_empty_cv_low_confidence(today):
    """Nearly empty extraction is flagged instead of being scored with confidence."""
    rep = run("Anna", today=today)
    assert rep.confidence <= 20 and "odczytano tylko" in rep.warnings[0]


def test_screening_verdict(today):
    """Good CV with all knock-outs met is shortlisted; a weak one is not."""
    offer = OFFER.replace("• prawo jazdy kat. B\n", "")
    good = run(GOOD_CV, job=offer, today=today)
    weak = run(WEAK_CV, job=offer, today=today)
    assert good.verdict.decision in ("shortlista", "dopytaj")
    assert weak.verdict.decision == "odrzuć"
    assert run(GOOD_CV, job=None, today=today).verdict is None
    assert run(GOOD_CV.replace("behance.net/anna", ""), today=today).verdict.decision == "odrzuć"


def test_compare_versions(today):
    """The tailored CV wins the comparison on tailoring."""
    tailored = run(GOOD_CV, today=today)
    generic = run(GOOD_CV.replace("Graficzka i specjalistka ds. social media", "Artystka")
                  .replace("Graficzka z", "Artystka z").replace("prowadzę social media", "maluję"), today=today)
    html, md = report.compare([tailored, generic], ["a.pdf", "b.pdf"])
    assert "<table>" in html and "Dopasowanie CV do oferty" in md
    assert tailored.metric("tailoring").score > generic.metric("tailoring").score


def test_letter_copied_from_other_application(today):
    """A letter naming another organisation (not in the offer, not a past employer) is flagged."""
    letter = ("Szanowni Państwo,\n\nz radością aplikuję do Fundacji Słoneczna Polana na stanowisko grafika.\n\n"
              "Z wyrazami szacunku\nAnna Przykładowa")
    rep = run(GOOD_CV, letter=letter, today=today)
    m = rep.metric("letter_personalization")
    assert m.score <= 25 and "Słoneczna Polana" in m.evidence[0]
    ok = run(GOOD_CV, letter=letter.replace("Fundacji Słoneczna Polana", "Centrum Kultury Przykładowego"), today=today)
    assert ok.metric("letter_personalization").score > 25
