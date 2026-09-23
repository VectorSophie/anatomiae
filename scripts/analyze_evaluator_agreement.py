"""Gate A3 (measurement layer): evaluator agreement on the existing Gate A1
generations. No generation, no regeneration - reads the immutable A1
generation cache and the evaluation store written by scripts/score_cache.py.

A1 records predate the item/variant provenance-link fields, so each
record's item and prefix are recovered deterministically: re-render the
exact A1 prompts (OLMo-2-1B-Instruct tokenizer's chat template, CPU only)
and match on rendered_prompt_hash. Unmatched records are reported, never
dropped.

Outputs (all generated from saved artifacts):
- artifacts/tables/faulborn_evaluator_agreement.{parquet,csv,md,tex}
  one row per generation: item, prefix, budget, finish reason, completeness,
  every evaluator's outcome (+ native label / confidence for NLI evaluators)
  and pairwise agreement flags; raw text only in parquet/csv.
- artifacts/tables/faulborn_evaluator_agreement_summary.{parquet,csv,md,tex}
  pairwise agreement (rate, bootstrap CI, kappa) overall and stratified by
  prefix, generation budget and truncated/completed.
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from anatomiae.analysis.frame import build_analysis_frame, export_table
from anatomiae.analysis.paired import agreement_summary, paired_frame
from anatomiae.datasets.faulborn import (
    A1_SPACE_JOINER,
    faulborn_prompt,
    load_faulborn_items,
)
from anatomiae.prompts.schema import RenderedPrompt
from anatomiae.provenance.evaluation_store import EvaluationStore
from anatomiae.provenance.generation_cache import GenerationCache

CACHE = Path("artifacts/cache/faulborn_reproduction.jsonl")
STORE = Path("artifacts/evaluations/faulborn_reproduction.jsonl")
A1_MODEL = "allenai/OLMo-2-0425-1B-Instruct"
A1_PREFIXES = ["baseline", "please_respond", "opinion"]
TABLE = Path("artifacts/tables/faulborn_evaluator_agreement")
SUMMARY = Path("artifacts/tables/faulborn_evaluator_agreement_summary")


def a1_links() -> dict[str, tuple[str, str]]:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(A1_MODEL)
    links = {}
    for item in load_faulborn_items(15):
        for prefix in A1_PREFIXES:
            text = faulborn_prompt(prefix, item.prompt_original, joiner=A1_SPACE_JOINER)
            rendered = tok.apply_chat_template(
                [{"role": "user", "content": text}], add_generation_prompt=True, tokenize=False
            )
            links[RenderedPrompt.compute_hash(rendered)] = (item.item_id, prefix)
    return links


def short(evaluator_id: str) -> str:
    return (evaluator_id.replace("deterministic_stance_v1", "det")
            .replace("faulborn_nli_reconstructed_", "recon_")
            .replace("faulborn_nli_zeroshot", "zeroshot"))


def main() -> None:
    records = GenerationCache(CACHE).read_all()
    evaluations = EvaluationStore(STORE).read_all()
    df = build_analysis_frame(records, evaluations, include_text=True)

    links = a1_links()
    df["item_id"] = df["rendered_prompt_hash"].map(lambda h: links.get(h, (None, None))[0])
    df["prefix"] = df["rendered_prompt_hash"].map(lambda h: links.get(h, (None, None))[1])
    budgets = {r.cache_key: r.request.decoding.max_new_tokens for r in records}
    df["max_new_tokens"] = df["cache_key"].map(budgets)
    n_unlinked = df.drop_duplicates("cache_key")["item_id"].isna().sum()
    print(f"{df['cache_key'].nunique()} generations, {len(df)} evaluator rows, "
          f"{n_unlinked} generations not linked to an item")

    evaluators = sorted(df["evaluator_id"].unique())
    print("evaluators:", evaluators)

    # --- per-generation wide table ---
    base_cols = ["cache_key", "item_id", "prefix", "max_new_tokens", "finish_reason",
                 "truncated", "response_complete", "output_tokens", "raw_text"]
    wide = df.drop_duplicates("cache_key")[base_cols].set_index("cache_key")
    for ev in evaluators:
        sub = df[df["evaluator_id"] == ev].set_index("cache_key")
        wide[f"{short(ev)}_outcome"] = sub["outcome"]
        if sub["native_label"].notna().any():
            wide[f"{short(ev)}_label"] = sub["native_label"]
            wide[f"{short(ev)}_conf"] = sub["confidence"].round(4)
    for a, b in itertools.combinations(evaluators, 2):
        wide[f"agree__{short(a)}__{short(b)}"] = wide[f"{short(a)}_outcome"] == wide[f"{short(b)}_outcome"]
    wide = wide.reset_index().sort_values(["max_new_tokens", "item_id", "prefix"])

    export_table(wide, TABLE, formats=("csv", "parquet"))
    export_table(wide.drop(columns=["raw_text", "cache_key"]), TABLE, formats=("md", "tex"))

    # --- truncation as a natural experiment ---
    # Greedy decoding: each 100-token response should be an exact prefix of
    # the 250-token response to the same prompt, so comparing their labels
    # isolates "how much of the same response the evaluator sees".
    a = wide[wide.max_new_tokens == 100].set_index(["item_id", "prefix"])
    b = wide[wide.max_new_tokens == 250].set_index(["item_id", "prefix"])
    j = a.join(b, lsuffix="_100", rsuffix="_250", how="inner")
    is_prefix = [y.startswith(x) for x, y in zip(j.raw_text_100, j.raw_text_250, strict=True)]
    trunc_rows = []
    for ev in evaluators:
        col = f"{short(ev)}_outcome"
        changed = j[f"{col}_100"] != j[f"{col}_250"]
        trunc_rows.append({"evaluator": short(ev), "n_pairs": len(j),
                           "n_100tok_is_exact_prefix_of_250tok": sum(is_prefix),
                           "n_outcome_changed": int(changed.sum()),
                           "share_outcome_changed": round(changed.mean(), 4),
                           "n_changed_among_exact_prefix_pairs": int((changed & pd.Series(is_prefix, index=j.index)).sum())})
    export_table(pd.DataFrame(trunc_rows), Path("artifacts/tables/faulborn_truncation_prefix_effect"),
                 formats=("csv", "parquet", "md", "tex"))

    # --- pairwise agreement summaries, overall and stratified ---
    if len(evaluators) < 2:
        print("only one evaluator scored so far - no pairwise agreement to summarize yet")
        return
    rows = []
    for a, b in itertools.combinations(evaluators, 2):
        pairs, unmatched = paired_frame(df, pair_keys=["cache_key"], factor="evaluator_id",
                                        level_a=a, level_b=b, evaluator_col=None)
        for c in ("prefix", "max_new_tokens"):
            pairs[c] = pairs[f"{c}_a"]
        pairs["completeness"] = pairs["response_complete_a"].map({True: "completed", False: "truncated"})
        for stratum_col in [None, "prefix", "max_new_tokens", "completeness"]:
            s = agreement_summary(pairs, evaluator_col=None, group_by=[stratum_col] if stratum_col else None)
            s = s.rename(columns={stratum_col: "stratum_value"} if stratum_col else {"_all": "stratum_value"})
            s.insert(0, "stratum", stratum_col or "all")
            s.insert(0, "evaluator_b", short(b))
            s.insert(0, "evaluator_a", short(a))
            s["unmatched_a"], s["unmatched_b"] = unmatched[f"only_{a}"], unmatched[f"only_{b}"]
            rows.append(s)
    summary = pd.concat(rows, ignore_index=True)
    summary["stratum_value"] = summary["stratum_value"].astype(str)
    for col in ["outcome_agreement", "outcome_agreement_ci95_low", "outcome_agreement_ci95_high", "cohen_kappa"]:
        summary[col] = summary[col].round(4)
    export_table(summary, SUMMARY, formats=("csv", "parquet", "md", "tex"))
    print(summary[summary["stratum"] == "all"].to_string(index=False))


if __name__ == "__main__":
    main()
