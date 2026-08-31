from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence
from typing import Any

import numpy as np
from scipy.stats import beta, chi2_contingency


def one_sided_exact_lower(successes: int, total: int, confidence: float = 0.95) -> float:
    """One-sided Clopper-Pearson lower bound."""
    if not 0 <= successes <= total or total <= 0:
        raise ValueError("require 0 <= successes <= total and total > 0")
    if successes == 0:
        return 0.0
    return float(beta.ppf(1.0 - confidence, successes, total - successes + 1))


def cohens_kappa(left: Sequence[str], right: Sequence[str], labels: Sequence[str]) -> float | None:
    if len(left) != len(right) or not left:
        raise ValueError("non-empty paired sequences required")
    n = len(left)
    observed = sum(a == b for a, b in zip(left, right)) / n
    left_counts = Counter(left)
    right_counts = Counter(right)
    expected = sum(left_counts[label] * right_counts[label] for label in labels) / (n * n)
    if math.isclose(expected, 1.0):
        return None
    return (observed - expected) / (1.0 - expected)


def confusion_matrix(
    targets: Sequence[str], predictions: Sequence[str], labels: Sequence[str]
) -> dict[str, dict[str, int]]:
    matrix = {target: {prediction: 0 for prediction in labels} for target in labels}
    for target, prediction in zip(targets, predictions):
        matrix[target][prediction] += 1
    return matrix


def option_position_dependence(positions: Sequence[int], correct: Sequence[bool]) -> dict[str, Any]:
    table = np.zeros((4, 2), dtype=int)
    for position, outcome in zip(positions, correct):
        table[position, int(bool(outcome))] += 1
    per_position = {
        str(position): float(table[position, 1] / table[position].sum())
        if table[position].sum()
        else None
        for position in range(4)
    }
    if np.any(table.sum(axis=1) == 0) or np.any(table.sum(axis=0) == 0):
        chi2, p_value, cramers_v = 0.0, 1.0, 0.0
    else:
        chi2, p_value, _, _ = chi2_contingency(table, correction=False)
        cramers_v = math.sqrt(chi2 / table.sum())
    finite = [value for value in per_position.values() if value is not None]
    return {
        "table_position_by_incorrect_correct": table.tolist(),
        "accuracy_by_position": per_position,
        "accuracy_range": max(finite) - min(finite) if finite else 0.0,
        "chi_square": float(chi2),
        "p_value": float(p_value),
        "cramers_v": float(cramers_v),
    }


def atomic_task_metrics(
    rows: Sequence[dict[str, Any]], labels: Sequence[str], prediction_key: str
) -> dict[str, Any]:
    targets = [row["target"] for row in rows]
    predictions = [row[prediction_key] for row in rows]
    correct = [target == prediction for target, prediction in zip(targets, predictions)]
    successes = sum(correct)
    return {
        "n": len(rows),
        "correct": successes,
        "accuracy": successes / len(rows),
        "one_sided_95_exact_lower": one_sided_exact_lower(successes, len(rows)),
        "confusion_matrix": confusion_matrix(targets, predictions, labels),
        "option_position_dependence": option_position_dependence(
            [row["correct_option_position"] for row in rows], correct
        ),
    }


def bootstrap_mean_ci(
    values: Sequence[float], *, resamples: int, seed: int, confidence: float = 0.95
) -> dict[str, float]:
    if not values:
        raise ValueError("values cannot be empty")
    array = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(array), size=(resamples, len(array)))
    sampled = array[indices].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    return {
        "mean": float(array.mean()),
        "lower": float(np.quantile(sampled, alpha)),
        "upper": float(np.quantile(sampled, 1.0 - alpha)),
        "resamples": resamples,
        "seed": seed,
    }
