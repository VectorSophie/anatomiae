"""Regenerate every result figure from saved tables. Figures whose source
table does not exist yet are skipped with a message, never faked.

    uv run python scripts/make_figures.py

Outputs figures/generated/<name>.{svg,pdf,png} + <name>.provenance.json.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from anatomiae.plots.figures import (
    SERIES,
    agreement_dotplot,
    confusion_heatmap,
    curve_small_multiples,
    dot_grid,
    outcome_stack,
    save,
    share_stack,
)

T = Path("artifacts/tables")
OUT = Path("figures/generated")
PRIMARY = "recon_as_written_seed42"  # the reconstruction that reproduces Faulborn's reported numbers


PRIMARY_NLI = "faulborn_nli_reconstructed_as_written_seed42"


def need(path: Path) -> bool:
    if not path.exists():
        print(f"skip: {path} not found")
        return False
    return True


def amber_curves() -> None:
    src = T / "amber_eval_curves.parquet"
    if not need(src):
        return
    df = pd.read_parquet(src)
    fig = curve_small_multiples(
        df, x="index",
        panels=[("hellaswag", "HellaSwag acc_norm (selection criterion)"), ("arc", "ARC-Challenge acc_norm"),
                ("truthfulqa", "TruthfulQA MC2"), ("mmlu", "MMLU acc (mean over subjects)")],
        markers={"early (010)": 10, "mid (180)": 180},
        chance={"hellaswag": 0.25, "arc": 0.25, "mmlu": 0.25},
        title="Amber per-checkpoint evaluations and the pilot checkpoint choice",
        subtitle=f"Published eval_*.json from {df['checkpoint'].nunique()} IFM/Amber branches; "
                 "early checkpoint chosen by a rule fixed before inspection",
    )
    save(fig, OUT / "amber_eval_curves", source_artifact=src,
         description="Amber per-checkpoint benchmark curves with selected pilot checkpoints")


def classifier_validation() -> None:
    src = T / "faulborn_classifier_validation.parquet"
    if not need(src):
        return
    df = pd.read_parquet(src)
    df = df[df["conf_threshold"].isin([0.0, 0.9])].copy()
    df["category"] = df["classifier"].str.replace("reconstructed_", "") + "  | conf>=" + df["conf_threshold"].astype(str)
    ours = df.rename(columns={"macro_f1": "outcome_agreement", "macro_f1_ci95_low": "outcome_agreement_ci95_low",
                              "macro_f1_ci95_high": "outcome_agreement_ci95_high", "n_retained": "n_pairs"})
    ours["series"] = "anatomiae (95% bootstrap CI)"
    theirs = df.assign(outcome_agreement=df["faulborn_reported_f1"],
                       outcome_agreement_ci95_low=df["faulborn_reported_f1"],
                       outcome_agreement_ci95_high=df["faulborn_reported_f1"],
                       n_pairs=df["faulborn_reported_n"], series="Faulborn et al. reported")
    fig = agreement_dotplot(
        pd.concat([ours, theirs]), category_col="category", series_col="series",
        title="Faulborn stance classifier: reconstruction vs. reported macro-F1",
        subtitle="Released 264-response test split; as_written reproduces the released data flow, "
                 "leakage_free excludes test responses from training",
        xlabel="Macro-F1 on the released test split",
    )
    save(fig, OUT / "faulborn_classifier_validation", source_artifact=src,
         description="Macro-F1 of zero-shot and reconstructed classifiers vs. Faulborn's reported values")


def evaluator_agreement() -> None:
    wide_src = T / "faulborn_evaluator_agreement.parquet"
    sum_src = T / "faulborn_evaluator_agreement_summary.parquet"
    if not (need(wide_src) and need(sum_src)):
        return
    wide = pd.read_parquet(wide_src)
    if f"{PRIMARY}_outcome" not in wide:
        print(f"skip: {PRIMARY} not scored yet")
        return
    fig = confusion_heatmap(
        wide, "det_outcome", f"{PRIMARY}_outcome",
        label_a="Deterministic rule-based evaluator", label_b="Faulborn NLI classifier (reconstructed, as_written)",
        title="Same 90 responses, two evaluators",
        subtitle="OLMo-2-1B-Instruct, 15 Faulborn items x 3 prefixes x 2 budgets; counts of generations",
    )
    save(fig, OUT / "faulborn_evaluator_confusion", source_artifact=wide_src,
         description="Confusion of deterministic vs reconstructed Faulborn classifier outcomes")

    long = pd.concat([
        pd.DataFrame({"evaluator": name, "outcome": wide[col]})
        for name, col in [("deterministic", "det_outcome"), ("zero-shot BART-MNLI", "zeroshot_outcome"),
                          ("reconstructed (as_written, seed 42)", f"{PRIMARY}_outcome"),
                          ("reconstructed (leakage_free, seed 42)", "recon_leakage_free_seed42_outcome")]
        if col in wide
    ])
    fig = outcome_stack(long, group_col="evaluator",
                        title="Outcome distribution depends on the evaluator",
                        subtitle="Identical 90 cached responses; no regeneration")
    save(fig, OUT / "faulborn_outcome_by_evaluator", source_artifact=wide_src,
         description="Outcome shares per evaluator on identical responses")

    summary = pd.read_parquet(sum_src)
    pairs = {("det", PRIMARY): "deterministic vs reconstructed",
             ("det", "zeroshot"): "deterministic vs zero-shot",
             ("recon_as_written_seed42", "recon_leakage_free_seed42"): "as_written vs leakage_free"}
    summary["pair"] = [pairs.get((a, b)) for a, b in zip(summary.evaluator_a, summary.evaluator_b, strict=True)]
    summary = summary.dropna(subset=["pair"])
    for stratum, fname, label in [("prefix", "faulborn_agreement_by_prefix", "prompt prefix"),
                                  ("completeness", "faulborn_agreement_by_completeness", "completed vs truncated")]:
        s = summary[summary["stratum"] == stratum]
        fig = agreement_dotplot(s, category_col="stratum_value", series_col="pair",
                                title=f"Evaluator agreement by {label}",
                                subtitle="Share of generations where two evaluators assign the same outcome")
        save(fig, OUT / fname, source_artifact=sum_src, description=f"Evaluator agreement by {label}")


def precision() -> None:
    src = T / "precision_sensitivity_summary.parquet"
    if not need(src):
        return
    s = pd.read_parquet(src)
    s = s[s["stratum"] == "all"].copy()
    s["series"] = "FP32 vs BF16"
    fig = agreement_dotplot(s, category_col="evaluator_id", series_col="series",
                            title="OLMo-2-13B Base: does inference precision change the measured outcome?",
                            subtitle="60 paired prompts, same revision/prompt/backend/greedy decoding; "
                                     "outcome agreement per evaluator")
    save(fig, OUT / "precision_outcome_agreement", source_artifact=src,
         description="FP32 vs BF16 outcome agreement per evaluator")


def stance_first() -> None:
    src = T / "stance_first_outcomes.parquet"
    if not need(src):
        return
    o = pd.read_parquet(src)
    o = o[(o["scope"] == "all") & o["evaluator_id"].isin(["deterministic_stance_v1", PRIMARY_NLI])]
    names = {"faulborn_released_please_respond": "released please_respond",
             "faulborn_released_opinion": "released opinion",
             "anatomiae_ext_stance_first": "stance-first (extension)"}
    evs = {"deterministic_stance_v1": "deterministic", PRIMARY_NLI: "reconstructed s42"}
    rows = []
    for _, r in o.iterrows():
        for col in [c for c in o.columns if c.startswith("share_")]:
            rows += [{"group": f"{names[r.condition]} / {evs[r.evaluator_id]}", "outcome": col[6:]}]
            rows[-1]["weight"] = r[col] * r["n"]
    long = pd.DataFrame(rows)
    long = long.loc[long.index.repeat(long["weight"].round().astype(int))][["group", "outcome"]]
    fig = outcome_stack(long, group_col="group",
                        title="Asking for the stance first changes what is measured",
                        subtitle="OLMo-2-1B-Instruct, same 30 Faulborn items per condition; greedy, 400 tokens")
    save(fig, OUT / "stance_first_outcomes", source_artifact=src,
         description="Outcome shares per elicitation condition and evaluator")


def backend() -> None:
    src = T / "backend_equivalence_summary.parquet"
    if not need(src):
        return
    s = pd.read_parquet(src)
    s = s[s["stratum"] == "all"].copy()
    s["evaluator"] = (s["evaluator_id"].str.replace("faulborn_nli_reconstructed_", "recon ")
                      .str.replace("faulborn_nli_", "").str.replace("deterministic_stance_v1", "deterministic"))
    fig = agreement_dotplot(s, category_col="evaluator", series_col="model",
                            title="Transformers vs vLLM: does the backend change the measured outcome?",
                            subtitle="60 paired prompts per model; same revision, rendered prompt, BF16, greedy decoding")
    save(fig, OUT / "backend_outcome_agreement", source_artifact=src,
         description="Transformers vs vLLM outcome agreement per evaluator and model")


STAGE_LABEL = {"0_base": "Base", "1_sft": "SFT", "2_dpo": "DPO", "3_rlvr2": "RLVR2"}
V2_PRIMARY = "faulborn_nli_reconstructed_as_written_seed42"
V2_DET = "deterministic_stance_v1"


def _cond(df: pd.DataFrame) -> pd.Series:
    return df["stage"].map(STAGE_LABEL) + " · " + df["render"]


def olmo_v2() -> None:
    src = T / "olmo_stages_v2_completion.parquet"
    if not need(src):
        return
    c = pd.read_parquet(src)
    c["completed_201_600_rate"] = c["completed_within_600_rate"] - c["completed_within_200_rate"]
    w = c.assign(group=_cond(c) + " · " + c["prompt"])
    cols = ["empty_rate", "completed_within_200_rate", "completed_201_600_rate", "truncated_at_600_rate"]
    shares = w.groupby("group", sort=False)[cols].mean()
    fig = share_stack(shares, [("empty_rate", "empty (immediate end)", "#c2c1b6"),
                               ("completed_within_200_rate", "finished within 200 tokens", SERIES[0]),
                               ("completed_201_600_rate", "finished in 201–600 tokens", SERIES[2]),
                               ("truncated_at_600_rate", "cut off at 600 tokens", SERIES[1])],
                      title="How each OLMo-2-13B stage answers, before any stance label",
                      subtitle="88 statements + 88 inversions per row; greedy, vLLM, BF16; from the generation cache")
    save(fig, OUT / "olmo_v2_response_mechanics", source_artifact=src,
         description="Empty / completion / truncation shares per stage, render and prompt")

    src = T / "olmo_stages_v2.parquet"
    if not need(src):
        return
    t = pd.read_parquet(src)
    t["condition"] = _cond(t)
    for ev, name, label in ((V2_DET, "olmo_v2_explicit_stance_rate", "explicit stance rate (deterministic evaluator)"),
                            (V2_PRIMARY, "olmo_v2_position_rate", "position rate (reconstructed Faulborn classifier)")):
        s = t[t["evaluator_id"] == ev]
        fig = dot_grid(s, category_col="condition", series_col="prompt", value_col="position_rate",
                       title=f"Released prompts vs stance-first: {label}",
                       subtitle="Share of responses taking a position; 176 prompts per point",
                       xlabel="share of responses")
        save(fig, OUT / name, source_artifact=src, description=f"{label} by stage, render and prompt")
    s = t[t["evaluator_id"] == V2_PRIMARY]
    fig = dot_grid(s, category_col="condition", series_col="prompt", value_col="direction_index", n_col="directional_n",
                   title="Measured direction by stage (not human-validated)",
                   subtitle="Agree rate on left-coded minus right-coded statements; reconstructed classifier; n = directional responses",
                   xlabel="direction index (−1 right-aligned … +1 left-aligned)", xlim=(-1, 1))
    save(fig, OUT / "olmo_v2_direction_index", source_artifact=src,
         description="Classifier-measured direction index by stage, render and prompt")

    src = T / "olmo_stages_v2_consistency.parquet"
    if not need(src):
        return
    k = pd.read_parquet(src)
    k = k[(k["evaluator_id"] == V2_PRIMARY) & (k["pair_subset"] == "A_faithful_only")].copy()
    k["condition"] = _cond(k)
    fig = dot_grid(k, category_col="condition", series_col="prompt", value_col="consistent_share",
                   n_col="n_items_both_directional",
                   title="Opposite stances on a statement and its faithful inversion",
                   subtitle="Audited faithful pairs (46) where both responses are directional; n = such pairs",
                   xlabel="share consistent (agree X & disagree not-X, or the reverse)")
    save(fig, OUT / "olmo_v2_consistency", source_artifact=src,
         description="Original/inverted consistency on faithful pairs, reconstructed classifier")


if __name__ == "__main__":
    for f in (amber_curves, classifier_validation, evaluator_agreement, precision, stance_first, backend, olmo_v2):
        f()
