"""Evaluation metrics for the scoring process itself (do the scores predict
outcomes, do raters agree, is selection fair). Pure Python/NumPy so they can be
unit-tested without data.

* ranking:     ``ndcg_at_k``, ``precision_at_k``, ``recall_at_k``, ``average_precision``, ``mrr``
* pairs:       ``pairwise_accuracy`` (does the score order pairs as the labels do)
* agreement:   ``cohen_kappa`` (optionally quadratic-weighted), ``spearman``
* prediction:  ``roc_auc``, ``point_biserial``
* fairness:    ``adverse_impact_ratio`` (the four-fifths rule)
"""

from __future__ import annotations

import math
from itertools import combinations

import numpy as np


def _order(scores):
    return sorted(range(len(scores)), key=lambda i: -scores[i])


def dcg(relevances) -> float:
    """Discounted cumulative gain with ``2^rel - 1`` gains."""
    return sum((2 ** r - 1) / math.log2(i + 2) for i, r in enumerate(relevances))


def ndcg_at_k(scores, relevances, k: int | None = None) -> float:
    """NDCG@k of a ranking by ``scores`` against graded ``relevances``."""
    k = k or len(scores)
    ranked = [relevances[i] for i in _order(scores)][:k]
    ideal = sorted(relevances, reverse=True)[:k]
    best = dcg(ideal)
    return dcg(ranked) / best if best > 0 else 0.0


def precision_at_k(scores, labels, k: int) -> float:
    """Share of positives among the top-k by score."""
    top = _order(scores)[:k]
    return sum(1 for i in top if labels[i]) / max(1, len(top))


def recall_at_k(scores, labels, k: int) -> float:
    """Share of all positives found in the top-k."""
    pos = sum(1 for x in labels if x)
    if not pos:
        return 0.0
    top = _order(scores)[:k]
    return sum(1 for i in top if labels[i]) / pos


def average_precision(scores, labels) -> float:
    """Average precision of the ranking."""
    hits, total = 0, 0.0
    for rank, i in enumerate(_order(scores), 1):
        if labels[i]:
            hits += 1
            total += hits / rank
    pos = sum(1 for x in labels if x)
    return total / pos if pos else 0.0


def mrr(scores, labels) -> float:
    """Reciprocal rank of the first positive."""
    for rank, i in enumerate(_order(scores), 1):
        if labels[i]:
            return 1.0 / rank
    return 0.0


def pairwise_accuracy(scores, targets) -> float:
    """Share of pairs with different targets that the scores order the same way."""
    good = total = 0
    for a, b in combinations(range(len(scores)), 2):
        if targets[a] == targets[b]:
            continue
        total += 1
        if (scores[a] - scores[b]) * (targets[a] - targets[b]) > 0:
            good += 1
    return good / total if total else float("nan")


def roc_auc(scores, labels) -> float:
    """ROC AUC as the probability a positive outranks a negative (ties count half)."""
    pos = [s for s, y in zip(scores, labels) if y]
    neg = [s for s, y in zip(scores, labels) if not y]
    if not pos or not neg:
        return float("nan")
    wins = sum(1.0 if p > n else 0.5 if p == n else 0.0 for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def point_biserial(values, labels) -> float:
    """Correlation between a continuous metric and a binary outcome."""
    x = np.asarray(values, dtype=float)
    y = np.asarray(labels, dtype=float)
    if len(x) < 3 or x.std() == 0 or y.std() == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def spearman(a, b) -> float:
    """Spearman rank correlation."""
    def ranks(v):
        order = np.argsort(v, kind="mergesort")
        r = np.empty(len(v))
        r[order] = np.arange(len(v))
        return r
    ra, rb = ranks(np.asarray(a, float)), ranks(np.asarray(b, float))
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def cohen_kappa(rater_a, rater_b, weighted: bool = False) -> float:
    """Cohen's kappa between two raters; ``weighted`` = quadratic weights (ordinal scales)."""
    cats = sorted(set(rater_a) | set(rater_b))
    idx = {c: i for i, c in enumerate(cats)}
    n = len(cats)
    if n < 2:
        return 1.0
    obs = np.zeros((n, n))
    for a, b in zip(rater_a, rater_b):
        obs[idx[a], idx[b]] += 1
    obs /= obs.sum()
    exp = np.outer(obs.sum(axis=1), obs.sum(axis=0))
    if weighted:
        w = np.array([[(i - j) ** 2 / (n - 1) ** 2 for j in range(n)] for i in range(n)])
    else:
        w = 1 - np.eye(n)
    denom = (w * exp).sum()
    return float(1 - (w * obs).sum() / denom) if denom else 1.0


def adverse_impact_ratio(selected, groups) -> dict:
    """Selection rate of each group divided by the highest group rate.

    A ratio below 0.8 (the four-fifths rule) is a warning sign of adverse impact.
    """
    rates = {}
    for g in set(groups):
        members = [s for s, gg in zip(selected, groups) if gg == g]
        rates[g] = sum(1 for s in members if s) / len(members)
    top = max(rates.values()) if rates else 0
    return {g: (r / top if top else float("nan")) for g, r in rates.items()}
