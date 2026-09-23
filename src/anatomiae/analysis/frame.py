"""Normalize immutable generation records + evaluator results into an
analysis-ready table, and export it to multiple formats from one source
of truth (never hand-copied numbers - project spec §29/§56).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from anatomiae.evaluators.base import EvaluationResult
from anatomiae.inference.schema import GenerationRecord


def build_analysis_frame(
    records: list[GenerationRecord],
    evaluations: list[EvaluationResult],
) -> pd.DataFrame:
    """One row per (generation record, evaluator) pair - a record scored by
    two evaluators yields two rows, not two columns, so adding an
    evaluator never requires reshaping existing rows (long format)."""
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
        rows.append(
            {
                "cache_key": record.cache_key,
                "model_id": record.request.model_id,
                "model_revision": record.request.model_revision,
                "backend": record.request.backend,
                "precision": record.request.precision,
                "seed": record.request.decoding.seed,
                "temperature": record.request.decoding.temperature,
                "rendered_prompt_hash": record.request.rendered_prompt_hash,
                "finish_reason": record.finish_reason,
                "input_tokens": record.input_tokens,
                "output_tokens": record.output_tokens,
                "latency_seconds": record.latency_seconds,
                "gpu_physical_index": record.gpu.physical_index if record.gpu else None,
                "gpu_power_limit_w": record.gpu.power_limit_w if record.gpu else None,
                "error": record.error,
                "evaluator_id": ev.evaluator_id,
                "evaluator_version": ev.evaluator_version,
                "outcome": ev.outcome,
                "confidence": ev.confidence,
            }
        )
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
