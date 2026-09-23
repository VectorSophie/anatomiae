"""Classification and inter-evaluator agreement metrics.

Pure Python, no scikit-learn dependency. `macro_f1` deliberately matches
scikit-learn's `f1_score(average="macro")` semantics - averaging over the
union of labels present in y_true and y_pred, with F1 = 0 for a label that
is never predicted or never true - because Faulborn et al.'s released
eval.py uses exactly that, and the classifier reconstruction is validated
against their reported numbers.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence


def _check_lengths(a: Sequence, b: Sequence) -> None:
    if len(a) != len(b):
        raise ValueError(f"length mismatch: {len(a)} vs {len(b)}")
    if not a:
        raise ValueError("empty input")


def per_label_f1(y_true: Sequence[str], y_pred: Sequence[str]) -> dict[str, float]:
    _check_lengths(y_true, y_pred)
    labels = sorted(set(y_true) | set(y_pred))
    out = {}
    for label in labels:
        tp = sum(t == label and p == label for t, p in zip(y_true, y_pred, strict=True))
        fp = sum(t != label and p == label for t, p in zip(y_true, y_pred, strict=True))
        fn = sum(t == label and p != label for t, p in zip(y_true, y_pred, strict=True))
        denom = 2 * tp + fp + fn
        out[label] = (2 * tp / denom) if denom else 0.0
    return out


def macro_f1(y_true: Sequence[str], y_pred: Sequence[str]) -> float:
    scores = per_label_f1(y_true, y_pred)
    return sum(scores.values()) / len(scores)


def accuracy(y_true: Sequence[str], y_pred: Sequence[str]) -> float:
    _check_lengths(y_true, y_pred)
    return sum(t == p for t, p in zip(y_true, y_pred, strict=True)) / len(y_true)


def cohen_kappa(a: Sequence[str], b: Sequence[str]) -> float:
    """Cohen's kappa between two raters over the same items. Returns NaN
    when expected agreement is 1 (both raters use a single identical label
    everywhere) - kappa is undefined there, and silently returning 0 or 1
    would misstate what was measured."""
    _check_lengths(a, b)
    n = len(a)
    observed = sum(x == y for x, y in zip(a, b, strict=True)) / n
    ca, cb = Counter(a), Counter(b)
    expected = sum((ca[k] / n) * (cb[k] / n) for k in set(ca) | set(cb))
    if expected == 1.0:
        return float("nan")
    return (observed - expected) / (1 - expected)


def confusion_counts(a: Sequence[str], b: Sequence[str]) -> dict[tuple[str, str], int]:
    _check_lengths(a, b)
    return dict(Counter(zip(a, b, strict=True)))
