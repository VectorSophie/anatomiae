"""Paired comparisons: the same items under two levels of one factor.

Precision (fp32 vs bf16), backend (transformers vs vllm) and prompt format
all share one shape: pair rows that are identical except for one factor,
then ask two separate questions -

- surface divergence: did the text change? (exact match, length, first
  divergence point) - needs `include_text=True` frames;
- dependent-variable divergence: did the *measurement* change? (per
  evaluator: outcome agreement, Cohen's kappa).

Byte-identical text is not the bar; the second question is the one that
matters for whether a factor materially changes political measurement.
Pairing is strict: an ambiguous key (two rows per level) raises, and rows
present at only one level are counted and reported, never silently dropped.
"""

from __future__ import annotations

import random

import pandas as pd

from anatomiae.metrics.agreement import cohen_kappa


def _common_prefix_len(a: str, b: str) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def paired_frame(
    df: pd.DataFrame,
    *,
    pair_keys: list[str],
    factor: str,
    level_a: object,
    level_b: object,
    evaluator_col: str | None = "evaluator_id",
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Returns (pairs, unmatched_counts). One row per (pair key, evaluator)
    with `_a`/`_b` suffixed columns for each level. Pass evaluator_col=None
    when the factor *is* the evaluator (evaluator-vs-evaluator agreement on
    the same generations), pairing on the generation alone."""
    keys = [*pair_keys, *([evaluator_col] if evaluator_col else [])]
    a = df[df[factor] == level_a]
    b = df[df[factor] == level_b]
    for side, level in ((a, level_a), (b, level_b)):
        dups = side[side.duplicated(keys, keep=False)]
        if len(dups):
            raise ValueError(
                f"{len(dups)} rows at {factor}={level!r} share a pairing key {keys} - "
                "pairing would be ambiguous; add a key column that disambiguates them."
            )
    pairs = a.merge(b, on=keys, suffixes=("_a", "_b"), how="inner")
    a_keys = set(map(tuple, a[keys].to_numpy()))
    b_keys = set(map(tuple, b[keys].to_numpy()))
    unmatched = {f"only_{level_a}": len(a_keys - b_keys), f"only_{level_b}": len(b_keys - a_keys)}

    pairs["outcome_agree"] = pairs["outcome_a"] == pairs["outcome_b"]
    pairs["both_complete"] = pairs["response_complete_a"] & pairs["response_complete_b"]
    pairs["both_truncated"] = pairs["truncated_a"] & pairs["truncated_b"]
    pairs["finish_reason_agree"] = pairs["finish_reason_a"] == pairs["finish_reason_b"]
    if "raw_text_a" in pairs:
        pairs["exact_text_match"] = pairs["raw_text_a"] == pairs["raw_text_b"]
        pairs["common_prefix_chars"] = [
            _common_prefix_len(x, y) for x, y in zip(pairs["raw_text_a"], pairs["raw_text_b"], strict=True)
        ]
        pairs["output_tokens_delta"] = pairs["output_tokens_b"] - pairs["output_tokens_a"]
    return pairs, unmatched


def bootstrap_rate_ci(values: list[bool], *, n_boot: int = 2000, seed: int = 0) -> tuple[float, float]:
    """Percentile 95% CI for a proportion, resampling units (pairs)."""
    if not values:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    n = len(values)
    stats = sorted(sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot))
    return stats[int(0.025 * n_boot)], stats[int(0.975 * n_boot) - 1]


def agreement_summary(
    pairs: pd.DataFrame,
    *,
    group_by: list[str] | None = None,
    evaluator_col: str | None = "evaluator_id",
) -> pd.DataFrame:
    """Per evaluator (and optional strata): outcome agreement rate with a
    bootstrap CI, Cohen's kappa, and - when text is present - the exact
    text match rate."""
    group_cols = [*([evaluator_col] if evaluator_col else []), *(group_by or [])]
    if not group_cols:
        pairs = pairs.assign(_all="all")
        group_cols = ["_all"]
    rows = []
    for key, g in pairs.groupby(group_cols, dropna=False):
        key = key if isinstance(key, tuple) else (key,)
        agree = list(g["outcome_agree"])
        lo, hi = bootstrap_rate_ci(agree)
        row = dict(zip(group_cols, key, strict=True))
        row.update(
            n_pairs=len(g),
            outcome_agreement=sum(agree) / len(agree),
            outcome_agreement_ci95_low=lo,
            outcome_agreement_ci95_high=hi,
            cohen_kappa=cohen_kappa(list(g["outcome_a"]), list(g["outcome_b"])),
        )
        if "exact_text_match" in g:
            row["exact_text_match_rate"] = g["exact_text_match"].mean()
        rows.append(row)
    return pd.DataFrame(rows)
