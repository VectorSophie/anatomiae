"""Analysis-only pass over the existing Faulborn reproduction cache.

Never generates, never loads a model or touches a GPU - reads the
immutable generation cache (artifacts/cache/faulborn_reproduction.jsonl)
and re-scores it. Adding an evaluator here never triggers regeneration
(the project's generate-once / score-many rule, docs/architecture.md).

Reports every outcome distribution three ways (all / completed-only /
truncated-only) via stratify_by_completeness - see
docs/results/faulborn_reproduction.md for why this matters here
specifically (95%+ of the original generations hit finish_reason=length).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from anatomiae.analysis.frame import (
    build_analysis_frame,
    export_table,
    stratify_by_completeness,
)
from anatomiae.evaluators.deterministic_stance import DeterministicStanceEvaluator
from anatomiae.provenance.generation_cache import GenerationCache

CACHE_PATH = Path("artifacts/cache/faulborn_reproduction.jsonl")


def main() -> None:
    cache = GenerationCache(CACHE_PATH)
    records = cache.read_all()
    print(f"Loaded {len(records)} cached records (no generation performed)")

    evaluators = [DeterministicStanceEvaluator()]
    evaluations = [ev_result for ev in evaluators for ev_result in ev.evaluate_many(records)]

    df = build_analysis_frame(records, evaluations)
    df["max_new_tokens"] = df["cache_key"].map(
        {r.cache_key: r.request.decoding.max_new_tokens for r in records}
    )

    written = export_table(
        df, Path("artifacts/tables/faulborn_reproduction"), formats=("csv", "parquet", "md")
    )
    print(f"Wrote {len(df)} rows: {sorted(str(p) for p in written.values())}")

    for budget, sub in df.groupby("max_new_tokens"):
        print(f"\n=== max_new_tokens={budget} (N={len(sub)}) ===")
        strata = stratify_by_completeness(sub, group_by=["evaluator_id"])
        for name, table in strata.items():
            print(f"\n-- {name} --")
            print(table.to_string(index=False) if len(table) else "(empty)")


if __name__ == "__main__":
    main()
