"""Learning metric weights from real outcomes (ML, no LLM).

Given applications with known outcomes (invited / not invited), fit an L2-regularised
logistic regression on the metric scores. With the small samples a single person
produces, the model is evaluated with leave-one-out cross-validation and its
coefficients are reported as *hints* on which metrics matter for this person's
market - not as truth. Unknown metric values are imputed with the training mean,
plus a missing-indicator feature, so "not measured" never silently becomes 0.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import evaluation
from .scoring import Report

MIN_SAMPLES = 8


def _features(reports: list[Report], metric_ids: list[str] | None = None):
    if metric_ids is None:
        metric_ids = sorted({m.id for r in reports for m in r.metrics})
    rows = []
    for r in reports:
        row = []
        for mid in metric_ids:
            m = r.metric(mid)
            row.append(np.nan if m is None or m.score is None else m.score / 100)
        rows.append(row)
    return np.array(rows, dtype=float), metric_ids


def fit(reports: list[Report], labels: list[int], c: float = 0.5) -> dict:
    """Fit the model; returns a JSON-serialisable dict with LOO-AUC and coefficients."""
    from sklearn.linear_model import LogisticRegression  # pylint: disable=import-outside-toplevel

    if len(reports) < MIN_SAMPLES or len(set(labels)) < 2:
        raise ValueError(f"Potrzeba co najmniej {MIN_SAMPLES} aplikacji z wynikiem, w tym pozytywnych i negatywnych.")
    x, ids = _features(reports)
    keep = [j for j in range(x.shape[1]) if np.isfinite(x[:, j]).sum() >= max(3, len(reports) // 2)]
    x, ids = x[:, keep], [ids[j] for j in keep]
    means = np.nanmean(x, axis=0)
    stds = np.nanstd(x, axis=0)
    stds[stds == 0] = 1.0

    def prep(mat, mu, sd):
        miss = ~np.isfinite(mat)
        filled = np.where(miss, mu, mat)
        return np.hstack([(filled - mu) / sd, miss.astype(float)])

    y = np.array(labels)
    loo = []
    for i in range(len(y)):
        tr = np.arange(len(y)) != i
        if len(set(y[tr])) < 2:
            loo.append(0.5)
            continue
        mu = np.nanmean(x[tr], axis=0)
        sd = np.nanstd(x[tr], axis=0)
        sd[sd == 0] = 1.0
        mu = np.where(np.isfinite(mu), mu, 0.5)
        model = LogisticRegression(C=c, class_weight="balanced", max_iter=2000)
        model.fit(prep(x[tr], mu, sd), y[tr])
        loo.append(float(model.predict_proba(prep(x[i:i + 1], mu, sd))[0, 1]))
    means = np.where(np.isfinite(means), means, 0.5)
    model = LogisticRegression(C=c, class_weight="balanced", max_iter=2000)
    model.fit(prep(x, means, stds), y)
    coefs = model.coef_[0][: len(ids)]
    return {
        "metric_ids": ids, "means": means.tolist(), "stds": stds.tolist(),
        "coef": model.coef_[0].tolist(), "intercept": float(model.intercept_[0]), "C": c,
        "n": int(len(y)), "n_positive": int(y.sum()),
        "loo_auc": evaluation.roc_auc(loo, list(y)),
        "importance": sorted(({"metric": m, "coef": float(w)} for m, w in zip(ids, coefs)),
                             key=lambda d: -abs(d["coef"])),
    }


def predict(model: dict, report: Report) -> float:
    """Learned probability-like score 0..100 for a report."""
    x, _ = _features([report], model["metric_ids"])
    mu, sd = np.array(model["means"]), np.array(model["stds"])
    miss = ~np.isfinite(x)
    filled = np.where(miss, mu, x)
    feats = np.hstack([(filled - mu) / sd, miss.astype(float)])[0]
    z = float(np.dot(feats, model["coef"]) + model["intercept"])
    return 100.0 / (1.0 + np.exp(-z))


def save(model: dict, path: Path) -> Path:
    """Write the model JSON."""
    path = Path(path)
    path.write_text(json.dumps(model, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load(path: Path) -> dict:
    """Read a model JSON."""
    return json.loads(Path(path).read_text(encoding="utf-8"))
