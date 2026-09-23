"""Normalize immutable generation records + evaluator results into an
analysis-ready table, and export it to multiple formats from one source
of truth (never hand-copied numbers - project spec §29/§56).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from anatomiae.evaluators.base import EvaluationResult
from anatomiae.evaluators.taxonomy import is_position_outcome
from anatomiae.inference.schema import GenerationRecord


def build_analysis_frame(
    records: list[GenerationRecord],
    evaluations: list[EvaluationResult],
    *,
    include_text: bool = False,
) -> pd.DataFrame:
    """One row per (generation record, evaluator) pair - a record scored by
    two evaluators yields two rows, not two columns, so adding an
    evaluator never requires reshaping existing rows (long format).

    Adds derived completeness columns (`truncated`, `response_complete`,
    `stance_observed_before_truncation`) from `finish_reason` - a truncated
    response must never be silently treated as equivalent to a complete
    one when reporting stance outcomes (e.g. `neutral_or_no_position` on a
    truncated response does not mean "the model has no position," it may
    mean "the model hadn't stated one yet when generation was cut off" -
    these are different claims and must stay distinguishable downstream).
    """
    records_by_key = {r.cache_key: r for r in records}

    rows = []
    for ev in evaluations:
        record = records_by_key.get(ev.generation_cache_key)
        if record is None:
            raise ValueError(
                f"EvaluationResult references cache_key {ev.generation_cache_key!r} "
                "that is not among the provided records - evaluations must reference "
                "records from the same immutable cache, never a dangling key."
            )
        truncated = record.finish_reason == "length"
        response_complete = record.finish_reason == "stop"
        rows.append(
            {
                "cache_key": record.cache_key,
                "item_id": record.request.item_id,
                "variant_id": record.request.variant_id,
                "template_id": record.request.template_id,
                "model_id": record.request.model_id,
                "model_revision": record.request.model_revision,
                "backend": record.request.backend,
                "precision": record.request.precision,
                "seed": record.request.decoding.seed,
                "temperature": record.request.decoding.temperature,
                "rendered_prompt_hash": record.request.rendered_prompt_hash,
                "finish_reason": record.finish_reason,
                "truncated": truncated,
                "response_complete": response_complete,
                "stance_observed_before_truncation": (
                    (truncated and is_position_outcome(ev.outcome)) if truncated else None
                ),
                "input_tokens": record.input_tokens,
                "output_tokens": record.output_tokens,
                "latency_seconds": record.latency_seconds,
                "gpu_physical_index": record.gpu.physical_index if record.gpu else None,
                "gpu_power_limit_w": record.gpu.power_limit_w if record.gpu else None,
                "error": record.error,
                "evaluator_id": ev.evaluator_id,
                "evaluator_version": ev.evaluator_version,
                "outcome": ev.outcome,
                "native_label": ev.native_label,
                "confidence": ev.confidence,
            }
        )
        if include_text:
            # Opt-in: needed for surface-divergence analysis (exact match,
            # length, first divergence), too bulky for default exports.
            rows[-1]["raw_text"] = record.raw_text
    return pd.DataFrame(rows)


def export_table(df: pd.DataFrame, base_path: Path | str, *, formats: tuple[str, ...] = ("csv", "parquet", "md")) -> dict[str, Path]:
    """Write the same DataFrame to multiple formats from one call - the
    project's "no hand-copied result numbers, one source of truth" rule."""
    base_path = Path(base_path)
    base_path.parent.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}

    if "csv" in formats:
        path = base_path.with_suffix(".csv")
        df.to_csv(path, index=False)
        written["csv"] = path
    if "parquet" in formats:
        path = base_path.with_suffix(".parquet")
        df.to_parquet(path, index=False)
        written["parquet"] = path
    if "md" in formats:
        path = base_path.with_suffix(".md")
        path.write_text(df.to_markdown(index=False))
        written["md"] = path
    if "tex" in formats:
        path = base_path.with_suffix(".tex")
        path.write_text(df.to_latex(index=False))
        written["tex"] = path

    return written


def stratify_by_completeness(
    df: pd.DataFrame, *, group_by: list[str] | None = None
) -> dict[str, pd.DataFrame]:
    """Outcome rates reported three ways - all outputs, completed outputs
    only, and truncated outputs separately - never collapsed into one
    number and never silently dropping the truncated slice. This is the
    minimum reporting the project's truncation-awareness rule requires for
    any free-form stance analysis (see docs/results/faulborn_reproduction.md
    for why: a high finish_reason=length rate can otherwise masquerade as
    a real "no stance" finding)."""
    return {
        "all": outcome_rate(df, group_by=group_by),
        "completed_only": outcome_rate(df[df["response_complete"]], group_by=group_by),
        "truncated_only": outcome_rate(df[df["truncated"]], group_by=group_by),
    }


def outcome_rate(df: pd.DataFrame, *, outcome_col: str = "outcome", group_by: list[str] | None = None) -> pd.DataFrame:
    """Per-outcome rate, optionally grouped (e.g. by model_id/backend). The
    denominator is always the full group size (all outcomes), matching the
    project's denominators-discipline rule - callers who want a
    position-only rate must filter first and say so explicitly, not rely
    on this function to guess which denominator they meant."""
    group_by = group_by or []
    group_cols = [*group_by, outcome_col]
    counts = df.groupby(group_cols, dropna=False).size().rename("n").reset_index()
    totals = df.groupby(group_by, dropna=False).size().rename("total") if group_by else pd.Series({"total": len(df)})
    if group_by:
        merged = counts.merge(totals.reset_index(), on=group_by)
    else:
        merged = counts.copy()
        merged["total"] = len(df)
    merged["rate"] = merged["n"] / merged["total"]
    return merged
