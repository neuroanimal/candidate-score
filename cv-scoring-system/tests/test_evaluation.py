"""Evaluation metrics against known values."""

import math

from cvscore import evaluation as ev


def test_ranking_metrics():
    """Ranking metrics."""
    scores = [0.9, 0.8, 0.3, 0.1]
    labels = [1, 0, 1, 0]
    assert ev.precision_at_k(scores, labels, 2) == 0.5
    assert ev.recall_at_k(scores, labels, 3) == 1.0
    assert ev.mrr(scores, labels) == 1.0
    assert math.isclose(ev.average_precision(scores, labels), (1 + 2 / 3) / 2)
    assert ev.ndcg_at_k([3, 2, 1], [3, 2, 1]) == 1.0
    assert ev.ndcg_at_k([1, 2, 3], [3, 2, 1]) < 1.0


def test_auc_pairwise_and_correlation():
    """Auc pairwise and correlation."""
    assert ev.roc_auc([0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0]) == 1.0
    assert ev.roc_auc([0.1, 0.2, 0.8, 0.9], [1, 1, 0, 0]) == 0.0
    assert ev.pairwise_accuracy([1, 2, 3], [1, 2, 3]) == 1.0
    assert ev.point_biserial([1, 2, 3, 4], [0, 0, 1, 1]) > 0.8
    assert ev.spearman([1, 2, 3], [10, 20, 30]) == 1.0


def test_kappa_and_adverse_impact():
    """Kappa and adverse impact."""
    assert ev.cohen_kappa([1, 2, 3, 1], [1, 2, 3, 1]) == 1.0
    assert ev.cohen_kappa([1, 2, 3, 3], [1, 2, 3, 2], weighted=True) > ev.cohen_kappa([1, 2, 3, 3], [1, 2, 3, 2])
    air = ev.adverse_impact_ratio([1, 1, 0, 1, 0, 0], ["a", "a", "a", "b", "b", "b"])
    assert air["a"] == 1.0 and math.isclose(air["b"], 0.5)
