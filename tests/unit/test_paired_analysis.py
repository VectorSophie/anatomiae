import math

import pytest

pd = pytest.importorskip("pandas")  # analysis needs the `ml` extra; CI runs `dev` only

from anatomiae.analysis.paired import (
    _common_prefix_len,
    agreement_summary,
    bootstrap_rate_ci,
    paired_frame,
)


def _row(prompt, precision, outcome, text="x", evaluator="ev", finish="stop"):
    return {
        "rendered_prompt_hash": prompt,
        "precision": precision,
        "evaluator_id": evaluator,
        "outcome": outcome,
        "raw_text": text,
        "finish_reason": finish,
        "truncated": finish == "length",
        "response_complete": finish == "stop",
        "output_tokens": len(text),
    }


def test_pairs_on_key_and_flags_agreement():
    df = pd.DataFrame([
        _row("p1", "fp32", "agreement", "same"), _row("p1", "bf16", "agreement", "same"),
        _row("p2", "fp32", "agreement", "abcX"), _row("p2", "bf16", "disagreement", "abcY"),
    ])
    pairs, unmatched = paired_frame(df, pair_keys=["rendered_prompt_hash"], factor="precision",
                                    level_a="fp32", level_b="bf16")
    assert len(pairs) == 2
    assert unmatched == {"only_fp32": 0, "only_bf16": 0}
    p1 = pairs[pairs.rendered_prompt_hash == "p1"].iloc[0]
    p2 = pairs[pairs.rendered_prompt_hash == "p2"].iloc[0]
    assert bool(p1.outcome_agree) and bool(p1.exact_text_match)
    assert not bool(p2.outcome_agree) and not bool(p2.exact_text_match)
    assert p2.common_prefix_chars == 3


def test_unmatched_rows_are_counted_not_silently_dropped():
    df = pd.DataFrame([
        _row("p1", "fp32", "agreement"), _row("p1", "bf16", "agreement"),
        _row("p2", "fp32", "agreement"),  # no bf16 counterpart
    ])
    pairs, unmatched = paired_frame(df, pair_keys=["rendered_prompt_hash"], factor="precision",
                                    level_a="fp32", level_b="bf16")
    assert len(pairs) == 1
    assert unmatched == {"only_fp32": 1, "only_bf16": 0}


def test_ambiguous_pairing_raises():
    df = pd.DataFrame([
        _row("p1", "fp32", "agreement"), _row("p1", "fp32", "refusal"),  # two fp32 rows, same key
        _row("p1", "bf16", "agreement"),
    ])
    with pytest.raises(ValueError, match="ambiguous"):
        paired_frame(df, pair_keys=["rendered_prompt_hash"], factor="precision",
                     level_a="fp32", level_b="bf16")


def test_evaluators_are_paired_separately():
    df = pd.DataFrame([
        _row("p1", "fp32", "agreement", evaluator="A"), _row("p1", "bf16", "agreement", evaluator="A"),
        _row("p1", "fp32", "neutral_or_no_position", evaluator="B"),
        _row("p1", "bf16", "disagreement", evaluator="B"),
    ])
    pairs, _ = paired_frame(df, pair_keys=["rendered_prompt_hash"], factor="precision",
                            level_a="fp32", level_b="bf16")
    summary = agreement_summary(pairs).set_index("evaluator_id")
    assert summary.loc["A", "outcome_agreement"] == 1.0
    assert summary.loc["B", "outcome_agreement"] == 0.0


def test_bootstrap_ci_brackets_the_point_estimate():
    lo, hi = bootstrap_rate_ci([True] * 8 + [False] * 2)
    assert lo <= 0.8 <= hi
    assert all(math.isnan(x) for x in bootstrap_rate_ci([]))


def test_common_prefix_len():
    assert _common_prefix_len("hello", "help") == 3
    assert _common_prefix_len("", "x") == 0


def test_evaluator_vs_evaluator_pairing_on_same_generations():
    df = pd.DataFrame([
        _row("g1", None, "agreement", evaluator="det"), _row("g1", None, "agreement", evaluator="nli"),
        _row("g2", None, "neutral_or_no_position", evaluator="det"),
        _row("g2", None, "disagreement", evaluator="nli"),
    ])
    pairs, unmatched = paired_frame(df, pair_keys=["rendered_prompt_hash"], factor="evaluator_id",
                                    level_a="det", level_b="nli", evaluator_col=None)
    assert len(pairs) == 2 and unmatched == {"only_det": 0, "only_nli": 0}
    summary = agreement_summary(pairs, evaluator_col=None)
    assert summary.iloc[0]["outcome_agreement"] == 0.5
