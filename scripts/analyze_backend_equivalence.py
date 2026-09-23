"""Gate B analysis: Transformers vs vLLM, paired on identical requests.

Reads the immutable caches and evaluation stores; no generation. The 13B
Base Transformers-BF16 side comes from the precision run (same revision,
prompts, decoding); every other record from the backend-equivalence cache.

Pairs on (model_id, rendered_prompt_hash) and reports, separately:
- input equality: rendered prompt hash (by construction) and prompt token
  count as each backend tokenized it;
- surface divergence: exact text match, first divergence point, length
  delta, finish-reason agreement;
- dependent-variable divergence per evaluator: outcome agreement with a
  bootstrap CI and Cohen's kappa - all / by model / by completeness;
- every outcome disagreement, classified.

Outputs: artifacts/tables/backend_equivalence{,_surface,_summary,_disagreements}.*
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from anatomiae.analysis.frame import build_analysis_frame, export_table
from anatomiae.analysis.paired import agreement_summary, paired_frame
from anatomiae.provenance.evaluation_store import EvaluationStore
from anatomiae.provenance.generation_cache import GenerationCache

OUT = Path("artifacts/tables/backend_equivalence")
SHORT = {"allenai/OLMo-2-0425-1B-Instruct": "OLMo-2-1B-Instruct",
         "allenai/OLMo-2-1124-13B-SFT": "OLMo-2-13B-SFT",
         "allenai/OLMo-2-1124-13B": "OLMo-2-13B-Base"}


def completeness(row) -> str:
    if row["response_complete_a"] and row["response_complete_b"]:
        return "both_complete"
    if row["truncated_a"] and row["truncated_b"]:
        return "both_truncated"
    return "mixed"


def main() -> None:
    be = GenerationCache("artifacts/cache/backend_equivalence.jsonl").read_all()
    prec = [r for r in GenerationCache("artifacts/cache/precision_sensitivity.jsonl").read_all()
            if r.request.precision == "bf16" and r.request.backend == "transformers"]
    evals = (EvaluationStore("artifacts/evaluations/backend_equivalence.jsonl").read_all()
             + EvaluationStore("artifacts/evaluations/precision_sensitivity.jsonl").read_all())
    keys = {r.cache_key for r in be + prec}
    evals = [e for e in evals if e.generation_cache_key in keys]  # precision store also holds the FP32 side
    df = build_analysis_frame(be + prec, evals, include_text=True)
    df["model"] = df["model_id"].map(SHORT)
    df["prefix"] = df["template_id"].str.removeprefix("faulborn_released_")

    pairs, unmatched = paired_frame(df, pair_keys=["model_id", "rendered_prompt_hash"], factor="backend",
                                    level_a="transformers", level_b="vllm")
    # precision-run records predate the link fields; the vLLM side of the same prompt hash carries them
    for c in ("model", "item_id", "prefix"):
        pairs[c] = pairs[f"{c}_a"].combine_first(pairs[f"{c}_b"])
    assert pairs["item_id"].notna().all() and pairs["prefix"].notna().all()
    pairs["input_tokens_equal"] = pairs["input_tokens_a"] == pairs["input_tokens_b"]
    pairs["completeness"] = pairs.apply(completeness, axis=1)
    print("unmatched:", unmatched)

    # surface divergence: one row per generation pair (evaluator-independent)
    gen = pairs.drop_duplicates(["model_id", "rendered_prompt_hash"])
    surface = gen.groupby("model").agg(
        n_pairs=("exact_text_match", "size"),
        input_tokens_equal=("input_tokens_equal", "mean"),
        exact_text_match=("exact_text_match", "mean"),
        finish_reason_agree=("finish_reason_agree", "mean"),
        median_first_divergence_char_if_differs=(
            "common_prefix_chars", lambda s: s[~gen.loc[s.index, "exact_text_match"]].median()),
        mean_abs_output_tokens_delta=("output_tokens_delta", lambda s: s.abs().mean()),
        truncated_share_transformers=("truncated_a", "mean"),
        truncated_share_vllm=("truncated_b", "mean"),
        errors=("error_a", lambda s: int(s.notna().sum() + gen.loc[s.index, "error_b"].notna().sum())),
    ).reset_index()
    export_table(surface.round(4), Path(f"{OUT}_surface"), formats=("csv", "parquet", "md", "tex"))
    print(surface.round(3).to_string(index=False))

    cols = ["model", "item_id", "prefix", "evaluator_id", "input_tokens_a", "input_tokens_b",
            "exact_text_match", "common_prefix_chars", "output_tokens_a", "output_tokens_b",
            "finish_reason_a", "finish_reason_b", "completeness", "outcome_a", "outcome_b", "outcome_agree"]
    table = pairs[cols].rename(columns=lambda c: c[:-2] + "_transformers" if c.endswith("_a")
                               else c[:-2] + "_vllm" if c.endswith("_b") else c)
    export_table(table.sort_values(["model", "evaluator_id", "item_id", "prefix"]), OUT,
                 formats=("csv", "parquet", "md", "tex"))

    dis = table[~table["outcome_agree"]].copy()
    pos = {"agreement", "disagreement"}
    dis["kind"] = ["direction_flip" if {a, b} == pos
                   else "position_vs_nonposition" if (a in pos) != (b in pos) else "nonposition_relabel"
                   for a, b in zip(dis["outcome_transformers"], dis["outcome_vllm"], strict=True)]
    export_table(dis, Path(f"{OUT}_disagreements"), formats=("csv", "parquet", "md"))
    print(dis.groupby(["model", "kind"]).size().to_string())
    print("disagreements on byte-identical text:", int(dis["exact_text_match"].sum()))

    parts = []
    for strata in [["model"], ["model", "completeness"], ["model", "prefix"]]:
        s = agreement_summary(pairs, group_by=strata)
        s.insert(1, "stratum", "+".join(strata[1:]) or "all")
        s["stratum_value"] = s[strata[1]] if len(strata) > 1 else "all"
        parts.append(s.drop(columns=strata[1:]))
    summary = pd.concat(parts, ignore_index=True)
    summary["unmatched_transformers"] = unmatched["only_transformers"]
    summary["unmatched_vllm"] = unmatched["only_vllm"]
    export_table(summary.round(4), Path(f"{OUT}_summary"), formats=("csv", "parquet", "md", "tex"))
    print(summary[summary.stratum == "all"].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
