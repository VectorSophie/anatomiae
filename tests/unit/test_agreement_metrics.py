import math

import pytest

from anatomiae.metrics.agreement import (
    accuracy,
    cohen_kappa,
    confusion_counts,
    macro_f1,
    per_label_f1,
)


def test_perfect_agreement():
    y = ["agree", "disagree", "neutral", "agree"]
    assert macro_f1(y, y) == 1.0
    assert accuracy(y, y) == 1.0
    assert cohen_kappa(y, y) == pytest.approx(1.0)


def test_macro_f1_matches_hand_computed_sklearn_semantics():
    # y_true: a a b b ; y_pred: a b b c
    # label a: tp=1 fp=0 fn=1 -> 2/3 ; label b: tp=1 fp=1 fn=1 -> 1/2
    # label c: never true, predicted once -> tp=0 fp=1 fn=0 -> 0
    # sklearn macro averages over union {a,b,c} -> (2/3 + 1/2 + 0)/3
    y_true = ["a", "a", "b", "b"]
    y_pred = ["a", "b", "b", "c"]
    scores = per_label_f1(y_true, y_pred)
    assert scores == pytest.approx({"a": 2 / 3, "b": 1 / 2, "c": 0.0})
    assert macro_f1(y_true, y_pred) == pytest.approx((2 / 3 + 1 / 2 + 0) / 3)


def test_cohen_kappa_hand_computed():
    # observed = 3/4; rater a: {x:2, y:2}, rater b: {x:3, y:1}
    # expected = (2/4)(3/4) + (2/4)(1/4) = 0.5 ; kappa = (0.75-0.5)/(1-0.5) = 0.5
    a = ["x", "x", "y", "y"]
    b = ["x", "x", "x", "y"]
    assert cohen_kappa(a, b) == pytest.approx(0.5)


def test_cohen_kappa_undefined_when_single_shared_label():
    a = ["neutral"] * 5
    assert math.isnan(cohen_kappa(a, a))


def test_length_mismatch_and_empty_rejected():
    with pytest.raises(ValueError, match="length mismatch"):
        macro_f1(["a"], ["a", "b"])
    with pytest.raises(ValueError, match="empty"):
        cohen_kappa([], [])


def test_confusion_counts():
    counts = confusion_counts(["a", "a", "b"], ["a", "b", "b"])
    assert counts == {("a", "a"): 1, ("a", "b"): 1, ("b", "b"): 1}
