"""OLMo-2-13B Base FP32 vs BF16: paired analysis of the precision runs.

Reads the immutable precision generation cache, the evaluation store
(scripts/score_cache.py), and the per-run summaries
(artifacts/logs/precision_run_{fp32,bf16}.json). No generation.

Pairs each FP32 generation with the BF16 generation of the *identical*
prompt (same rendered_prompt_hash; revision/backend/decoding/seed are
fixed by the run script) and reports, separately:

- surface divergence: exact text match, first divergence point, output
  length delta, finish-reason agreement;
- dependent-variable divergence, per evaluator: outcome agreement
  (bootstrap 95% CI), Cohen's kappa - overall and by completeness stratum
  (both complete / both truncated / mixed);
- resources: load time, tokens/s, peak VRAM, applied power limit.

Outputs: artifacts/tables/precision_sensitivity{,_summary,_resources}.{parquet,csv,md,tex}
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from anatomiae.analysis.frame import build_analysis_frame, export_table
from anatomiae.analysis.paired import agreement_summary, paired_frame
from anatomiae.datasets.faulborn import (
    RELEASED_JOINER,
    faulborn_prompt,
    load_faulborn_items,
)
from anatomiae.prompts.schema import RenderedPrompt
from anatomiae.provenance.evaluation_store import EvaluationStore
from anatomiae.provenance.generation_cache import GenerationCache

CACHE = Path("artifacts/cache/precision_sensitivity.jsonl")
STORE = Path("artifacts/evaluations/precision_sensitivity.jsonl")
PREFIXES = ["please_respond", "opinion"]
OUT = Path("artifacts/tables/precision_sensitivity")


def links(n_items: int) -> dict[str, tuple[str, str]]:
    out = {}
    for item in load_faulborn_items(n_items):
        for prefix in PREFIXES:
            text = faulborn_prompt(prefix, item.prompt_original, joiner=RELEASED_JOINER)
            out[RenderedPrompt.compute_hash(text)] = (item.item_id, prefix)
    return out


def completeness_stratum(row) -> str:
    if row["response_complete_a"] and row["response_complete_b"]:
        return "both_complete"
    if row["truncated_a"] and row["truncated_b"]:
        return "both_truncated"
    return "mixed"


def main() -> None:
    records = GenerationCache(CACHE).read_all()
    df = build_analysis_frame(records, EvaluationStore(STORE).read_all(), include_text=True)
    link = links(30)
    df["item_id"] = df["rendered_prompt_hash"].map(lambda h: link.get(h, (None, None))[0])
    df["prefix"] = df["rendered_prompt_hash"].map(lambda h: link.get(h, (None, None))[1])
    print(f"{df['cache_key'].nunique()} generations; unlinked: "
          f"{df.drop_duplicates('cache_key')['item_id'].isna().sum()}; evaluators: "
          f"{sorted(df['evaluator_id'].unique())}")

    pairs, unmatched = paired_frame(df, pair_keys=["rendered_prompt_hash"], factor="precision",
                                    level_a="fp32", level_b="bf16")
    pairs["item_id"], pairs["prefix"] = pairs["item_id_a"], pairs["prefix_a"]
    pairs["completeness"] = pairs.apply(completeness_stratum, axis=1)
    print("unmatched:", unmatched)

    # per-pair table (one row per prompt x evaluator)
    cols = ["item_id", "prefix", "evaluator_id", "exact_text_match", "common_prefix_chars",
            "output_tokens_a", "output_tokens_b", "output_tokens_delta", "finish_reason_a",
            "finish_reason_b", "completeness", "outcome_a", "outcome_b", "outcome_agree",
            "latency_seconds_a", "latency_seconds_b"]
    table = pairs[cols].rename(columns=lambda c: c[:-2] + "_fp32" if c.endswith("_a")
                               else c[:-2] + "_bf16" if c.endswith("_b") else c)
    table = table.sort_values(["evaluator_id", "item_id", "prefix"])
    export_table(table, OUT, formats=("csv", "parquet", "md", "tex"))

    # agreement summaries
    parts = []
    for stratum in [None, "completeness", "prefix"]:
        s = agreement_summary(pairs, group_by=[stratum] if stratum else None)
        s.insert(1, "stratum", stratum or "all")
        if stratum:
            s = s.rename(columns={stratum: "stratum_value"})
        else:
            s["stratum_value"] = "all"
        parts.append(s)
    summary = pd.concat(parts, ignore_index=True)
    summary["unmatched_fp32"], summary["unmatched_bf16"] = unmatched["only_fp32"], unmatched["only_bf16"]
    export_table(summary.round(4), Path(f"{OUT}_summary"), formats=("csv", "parquet", "md", "tex"))
    print(summary.round(3).to_string(index=False))

    # resources
    res = pd.DataFrame([json.loads(Path(f"artifacts/logs/precision_run_{p}.json").read_text())
                        for p in ("fp32", "bf16")])
    res = res[["precision", "load_seconds", "n_new_generations", "n_errors", "total_output_tokens",
               "tokens_per_second", "peak_vram_allocated_gib", "peak_vram_reserved_gib",
               "gpu_power_limit_w"]]
    export_table(res.round(2), Path(f"{OUT}_resources"), formats=("csv", "parquet", "md", "tex"))
    print(res.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
