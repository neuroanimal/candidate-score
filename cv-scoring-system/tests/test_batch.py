"""Batch mode, classification, outcomes and calibration."""

import shutil
import sys
from pathlib import Path

import pytest

from cvscore import batch, calibrate
from cvscore.extract import from_text

sys.path.insert(0, str(Path(__file__).parent))


def test_classify():
    """Classify."""
    assert batch.classify(from_text("Wymagania:\n• Photoshop\nOferujemy:\n• umowę"), "x.pdf")[0] == "job"
    assert batch.classify(from_text("Szanowni Państwo,\nchciałabym...\nZ wyrazami szacunku"), "x.pdf")[0] == "letter"
    assert batch.classify(from_text("Doświadczenie zawodowe\n2019-2021 Firma\nWykształcenie\n2014-2019"), "x.pdf")[0] == "cv"
    assert batch.classify(from_text("tekst"), "CV_Anna.pdf")[0] == "cv"


def test_outcome_labels():
    """Outcome labels."""
    app = batch.Application("a", cv=None, outcome="zaproszenie na rozmowę")
    assert app.label == 1 and app.grade == 2
    assert batch.Application("b", cv=None, outcome="brak odpowiedzi").label == 0
    assert batch.Application("c", cv=None, outcome="zatrudniona").grade == 3
    assert batch.Application("d", cv=None, outcome="?").label is None


def test_batch_folder(sample_pdfs, tmp_path, today):
    """Batch folder."""
    root = tmp_path / "apps"
    for name, cv, job, outcome in (("a1", "cv_2col.pdf", "oferta_sds_instruktor.pdf", "rozmowa"),
                                   ("a2", "cv_2col.pdf", "oferta_koordynator_wydarzen.pdf", "odrzucona"),
                                   ("a3", "cv_tech.pdf", "oferta_automation_developer.pdf", None)):
        d = root / name
        d.mkdir(parents=True)
        shutil.copy(sample_pdfs / cv, d / ("CV_" + cv))
        shutil.copy(sample_pdfs / job, d / job)
        if outcome:
            (d / "wynik.txt").write_text(outcome, encoding="utf-8")
    shutil.copy(sample_pdfs / "list_sds.pdf", root / "a1" / "list_motywacyjny.pdf")
    res = batch.run(root, today=today)
    assert [a.id for a in res.apps] == ["a1", "a2", "a3"]
    assert res.apps[0].letter is not None and res.apps[0].label == 1
    assert len(res.pool["cv"]) == 2  # identical CV copies are de-duplicated
    assert len(res.matrix) == 2 * 3
    paths = batch.write_batch(res, tmp_path / "out")
    assert (tmp_path / "out" / "index.html").is_file() and len(paths) >= 3 * 3 + 3
    assert res.analysis["n_labeled"] == 2


def test_manifest(sample_pdfs, tmp_path):
    """Manifest."""
    (tmp_path / "manifest.yaml").write_text(
        f"- id: m1\n  cv: {sample_pdfs / 'cv_2col.pdf'}\n  job: {sample_pdfs / 'oferta_animator_dzieci.pdf'}\n"
        "  outcome: rozmowa\n", encoding="utf-8")
    apps, pool = batch.discover(tmp_path, batch.DocCache())
    assert apps[0].id == "m1" and apps[0].label == 1 and pool["job"]


class _FakeReport:
    def __init__(self, metrics):
        self.metrics = metrics

    def metric(self, mid):
        """Metric by id."""
        return next((m for m in self.metrics if m.id == mid), None)


def test_calibrate_learns_signal():
    """Calibrate learns signal."""
    from cvscore.metrics.base import make  # pylint: disable=import-outside-toplevel

    reports, labels = [], []
    for i in range(16):
        y = i % 2
        reports.append(_FakeReport([make("must_have_coverage", 30 + 50 * y + i), make("readability", 50 + (i * 7) % 20),
                                    make("tailoring", None if i % 5 == 0 else 60)]))
        labels.append(y)
    model = calibrate.fit(reports, labels)
    assert model["loo_auc"] > 0.8
    assert model["importance"][0]["metric"] == "must_have_coverage"
    assert calibrate.predict(model, reports[1]) > calibrate.predict(model, reports[0])
    with pytest.raises(ValueError):
        calibrate.fit(reports[:3], labels[:3])


def test_batch_outcome_model(tmp_path, today):
    """With 8+ outcomes the batch analysis includes a learned model and separating metrics."""
    from test_scoring import GOOD_CV, OFFER, WEAK_CV  # pylint: disable=import-outside-toplevel

    for i in range(10):
        d = tmp_path / f"app{i:02d}"
        d.mkdir()
        good = i % 2 == 0
        (d / "cv.txt").write_text(GOOD_CV if good else WEAK_CV, encoding="utf-8")
        (d / "oferta.txt").write_text(OFFER, encoding="utf-8")
        (d / "wynik.txt").write_text("rozmowa" if good else "odrzucona", encoding="utf-8")
    res = batch.run(tmp_path, today=today, cross=False)
    an = res.analysis
    assert an["n_labeled"] == 10 and an["composites"]["cv_quality"]["auc"] == 1.0
    assert "model" in an and an["model"]["loo_auc"] > 0.8
    batch.write_batch(res, tmp_path / "out")
    assert "Regresja logistyczna" in (tmp_path / "out" / "index.html").read_text(encoding="utf-8")
