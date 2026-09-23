"""Gate A3: original Faulborn prompts vs the stance-first extension.

Reads the immutable extension generation cache and its evaluation store
(scripts/score_cache.py). No generation.

Per evaluator, paired by item (each item appears under every condition):

1. position rate per condition - share of responses whose outcome is a
   position (agreement / disagreement / mixed_or_conditional);
2. paired position-rate difference, extension minus each original
   condition, with a bootstrap 95% CI resampling items;
3. direction consistency - among items where both conditions yield
   agreement or disagreement, the share with the same direction;
4. everything reported for all responses and for completed-only responses
   (truncation reported, never used to silently drop rows).

Outputs: artifacts/tables/stance_first_{outcomes,paired}.{csv,parquet,md,tex}
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from anatomiae.analysis.frame import build_analysis_frame, export_table
from anatomiae.evaluators.taxonomy import POSITION_OUTCOMES
from anatomiae.provenance.evaluation_store import EvaluationStore
from anatomiae.provenance.generation_cache import GenerationCache

CACHE = Path("artifacts/cache/stance_first_extension.jsonl")
STORE = Path("artifacts/evaluations/stance_first_extension.jsonl")
EXT = "anatomiae_ext_stance_first"
ORIGINALS = ["faulborn_released_please_respond", "faulborn_released_opinion"]


def boot_mean_ci(values: list[float], n_boot: int = 2000, seed: int = 0) -> tuple[float, float]:
    rng = random.Random(seed)
    n = len(values)
    stats = sorted(sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot))
    return stats[int(0.025 * n_boot)], stats[int(0.975 * n_boot) - 1]


def main() -> None:
    records = GenerationCache(CACHE).read_all()
    df = build_analysis_frame(records, EvaluationStore(STORE).read_all())
    df["is_position"] = df["outcome"].isin(POSITION_OUTCOMES)
    df["direction"] = df["outcome"].where(df["outcome"].isin(["agreement", "disagreement"]))

    # 1. outcome/position rates per condition x evaluator x completeness scope
    rows = []
    for scope, sub in [("all", df), ("completed_only", df[df["response_complete"]])]:
        for (ev, tmpl), g in sub.groupby(["evaluator_id", "template_id"]):
            row = {"scope": scope, "evaluator_id": ev, "condition": tmpl, "n": len(g),
                   "truncated_share": round(g["truncated"].mean(), 4),
                   "position_rate": round(g["is_position"].mean(), 4)}
            for o in sorted(df["outcome"].unique()):
                row[f"share_{o}"] = round((g["outcome"] == o).mean(), 4)
            rows.append(row)
    outcomes = pd.DataFrame(rows)
    export_table(outcomes, Path("artifacts/tables/stance_first_outcomes"), formats=("csv", "parquet", "md", "tex"))

    # 2-3. paired by item: extension vs each original condition
    paired_rows = []
    for ev, g in df.groupby("evaluator_id"):
        piv_pos = g.pivot(index="item_id", columns="template_id", values="is_position")
        piv_dir = g.pivot(index="item_id", columns="template_id", values="direction")
        piv_done = g.pivot(index="item_id", columns="template_id", values="response_complete")
        for orig in ORIGINALS:
            for scope in ("all", "both_completed"):
                keep = piv_pos.index if scope == "all" else piv_done.index[piv_done[EXT] & piv_done[orig]]
                if len(keep) == 0:
                    continue
                diffs = (piv_pos.loc[keep, EXT].astype(int) - piv_pos.loc[keep, orig].astype(int)).tolist()
                lo, hi = boot_mean_ci(diffs)
                both_dir = piv_dir.loc[keep, [EXT, orig]].dropna()
                paired_rows.append({
                    "evaluator_id": ev, "comparison": f"{EXT} - {orig}", "scope": scope,
                    "n_items": len(keep),
                    "position_rate_extension": round(piv_pos.loc[keep, EXT].mean(), 4),
                    "position_rate_original": round(piv_pos.loc[keep, orig].mean(), 4),
                    "paired_diff": round(sum(diffs) / len(diffs), 4),
                    "paired_diff_ci95_low": round(lo, 4), "paired_diff_ci95_high": round(hi, 4),
                    "n_items_both_directional": len(both_dir),
                    "direction_consistency": (round((both_dir[EXT] == both_dir[orig]).mean(), 4)
                                              if len(both_dir) else None),
                })
    paired = pd.DataFrame(paired_rows)
    export_table(paired, Path("artifacts/tables/stance_first_paired"), formats=("csv", "parquet", "md", "tex"))
    print(outcomes[outcomes["scope"] == "all"][["evaluator_id", "condition", "n", "truncated_share",
                                                "position_rate"]].to_string(index=False))
    print(paired.to_string(index=False))


if __name__ == "__main__":
    main()
