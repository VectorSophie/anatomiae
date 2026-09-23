"""Formation layer, first pass: OLMo-2-13B stage comparison (analysis only).

Collects the Base / SFT / DPO (/ RLVR2) Transformers-BF16 greedy-200 records
for the 60 Faulborn prompts from the three caches that hold them, plus their
evaluations. Per stage x render arm x evaluator:

- position_rate: share labeled agreement / disagreement / mixed;
- directional_n: responses labeled agreement or disagreement;
- left_aligned_share: among directional responses, share whose direction
  matches the item's human left/right coding (agree with a left-coded item
  or disagree with a right-coded one). The 88-item bank is 59 right / 29
  left, so this share confounds direction with a general tendency to
  agree or disagree - hence also:
- acquiescence: mean of the agreement rates on left- and right-coded items
  (lean-balanced tendency to agree);
- direction_index: agreement rate on left-coded minus right-coded items
  (lean-balanced direction; +1 fully left-aligned, -1 fully right-aligned).
The coding is Faulborn's coding of the *item*, used only to put direction
on one axis; it says nothing about whether that axis suits a model.

Differences vs Base (raw) come with a 95% bootstrap CI resampling items
(both prompts of an item move together). Position rate and direction are
reported separately on purpose: Base rarely states a position, so a
pooled "stance score" would mostly measure "learned to answer".

Outputs: artifacts/tables/olmo_stages{,_diffs}.{csv,parquet,md,tex}
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from anatomiae.analysis.frame import build_analysis_frame, export_table
from anatomiae.datasets.faulborn import load_faulborn_items
from anatomiae.evaluators.taxonomy import POSITION_OUTCOMES
from anatomiae.provenance.evaluation_store import EvaluationStore
from anatomiae.provenance.generation_cache import GenerationCache

STAGE = {"allenai/OLMo-2-1124-13B": "0_base", "allenai/OLMo-2-1124-13B-SFT": "1_sft",
         "allenai/OLMo-2-1124-13B-DPO": "2_dpo", "allenai/OLMo-2-1124-13B-Instruct-RLVR2": "3_rlvr2"}
SOURCES = ["precision_sensitivity", "backend_equivalence", "olmo_stages"]
CHAT_MARK = "<|user|>"
DIRECTIONAL = {"agreement", "disagreement"}
N_BOOT = 2000


def load() -> pd.DataFrame:
    recs, evals = [], []
    for s in SOURCES:
        if Path(f"artifacts/cache/{s}.jsonl").exists():
            recs += GenerationCache(f"artifacts/cache/{s}.jsonl").read_all()
        if Path(f"artifacts/evaluations/{s}.jsonl").exists():
            evals += EvaluationStore(f"artifacts/evaluations/{s}.jsonl").read_all()
    links = {r.request.rendered_prompt_hash: (r.request.item_id, r.request.template_id)
             for r in recs if r.request.item_id is not None}
    recs = [r for r in recs if r.request.model_id in STAGE and r.request.backend == "transformers"
            and r.request.precision == "bf16" and r.request.decoding.max_new_tokens == 200
            and r.request.decoding.temperature == 0.0]
    recs = list({r.cache_key: r for r in recs}.values())
    keys = {r.cache_key for r in recs}
    render = {r.cache_key: "native" if CHAT_MARK in r.request.rendered_text else "raw" for r in recs}
    df = build_analysis_frame(recs, [e for e in evals if e.generation_cache_key in keys])
    df["stage"] = df["model_id"].map(STAGE)
    df["render"] = df["cache_key"].map(render)
    # precision-run records predate the link fields: recover them from the rendered-prompt hash
    for i, c in enumerate(("item_id", "template_id")):
        df[c] = df[c].fillna(df["rendered_prompt_hash"].map(lambda h, i=i: links.get(h, (None, None))[i]))
    assert df["item_id"].notna().all(), "unlinked records"
    ref = {i.item_id: i.reference_position for i in load_faulborn_items()}
    df["item_lean"] = df["item_id"].map(ref)
    df["is_position"] = df["outcome"].isin(POSITION_OUTCOMES)
    df["is_directional"] = df["outcome"].isin(DIRECTIONAL)
    df["left_aligned"] = ((df["outcome"] == "agreement") & (df["item_lean"] == "left")) | (
        (df["outcome"] == "disagreement") & (df["item_lean"] == "right"))
    return df


def _rate(num: float, den: float) -> float:
    return num / den if den else math.nan


def metrics(g: pd.DataFrame) -> dict:
    d = g[g["is_directional"]]
    agree = d["outcome"] == "agreement"
    left, right = d["item_lean"] == "left", d["item_lean"] == "right"
    a_left, a_right = _rate(agree[left].sum(), left.sum()), _rate(agree[right].sum(), right.sum())
    return {"n": len(g), "truncated_share": g["truncated"].mean(), "position_rate": g["is_position"].mean(),
            "directional_n": len(d), "directional_n_left_items": int(left.sum()),
            "directional_n_right_items": int(right.sum()),
            "agreement_share_of_directional": _rate(agree.sum(), len(d)),
            "left_aligned_share": _rate(d["left_aligned"].sum(), len(d)),
            "agree_rate_left_items": a_left, "agree_rate_right_items": a_right,
            "acquiescence": (a_left + a_right) / 2, "direction_index": a_left - a_right}


def _item_counts(g: pd.DataFrame, items: list[str]) -> dict[str, np.ndarray]:
    """Per-item counts, aligned to `items`, from which every bootstrapped metric is a ratio of sums."""
    d = g.assign(agree=g["outcome"] == "agreement", L=g["item_lean"] == "left")
    agg = d.groupby("item_id").agg(
        n=("is_position", "size"), pos=("is_position", "sum"), dirn=("is_directional", "sum"),
        la=("left_aligned", "sum"),
        agL=("agree", lambda s: (s & d.loc[s.index, "L"] & d.loc[s.index, "is_directional"]).sum()),
        dL=("is_directional", lambda s: (s & d.loc[s.index, "L"]).sum()),
        agR=("agree", lambda s: (s & ~d.loc[s.index, "L"] & d.loc[s.index, "is_directional"]).sum()),
        dR=("is_directional", lambda s: (s & ~d.loc[s.index, "L"]).sum()),
    ).reindex(items, fill_value=0)
    return {c: agg[c].to_numpy(float) for c in agg.columns}


def _boot_metric(c: dict[str, np.ndarray], idx: np.ndarray, col: str) -> np.ndarray:
    s = {k: v[idx].sum(1) for k, v in c.items()}
    aL, aR = s["agL"] / s["dL"], s["agR"] / s["dR"]
    return {"position_rate": s["pos"] / s["n"], "left_aligned_share": s["la"] / s["dirn"],
            "acquiescence": (aL + aR) / 2, "direction_index": aL - aR}[col]


def boot_diff(a: pd.DataFrame, b: pd.DataFrame, col: str) -> tuple[float, float]:
    """95% CI of metric(b) - metric(a), resampling items jointly (ratio of sums)."""
    items = sorted(set(a["item_id"]) | set(b["item_id"]))
    ca, cb = _item_counts(a, items), _item_counts(b, items)
    idx = np.random.default_rng(0).integers(0, len(items), size=(N_BOOT, len(items)))
    with np.errstate(invalid="ignore", divide="ignore"):
        v = _boot_metric(cb, idx, col) - _boot_metric(ca, idx, col)
    v = v[~np.isnan(v)]
    if len(v) < 0.9 * N_BOOT:  # metric undefined in too many resamples to report an interval
        return (math.nan, math.nan)
    return (float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975)))


def main() -> None:
    df = load()
    rows = [{"stage": s, "render": r, "evaluator_id": e, **metrics(g)}
            for (s, r, e), g in df.groupby(["stage", "render", "evaluator_id"])]
    table = pd.DataFrame(rows).round(4)
    export_table(table, Path("artifacts/tables/olmo_stages"), formats=("csv", "parquet", "md", "tex"))

    diffs = []
    for e, ge in df.groupby("evaluator_id"):
        base = ge[(ge["stage"] == "0_base") & (ge["render"] == "raw")]
        for (s, r), g in ge[ge["stage"] != "0_base"].groupby(["stage", "render"]):
            for col in ("position_rate", "left_aligned_share", "acquiescence", "direction_index"):
                lo, hi = boot_diff(base, g, col)
                diffs.append({"evaluator_id": e, "stage": s, "render": r, "metric": col,
                              "base_raw": metrics(base)[col], "stage_value": metrics(g)[col],
                              "diff": metrics(g)[col] - metrics(base)[col], "ci95_low": lo, "ci95_high": hi})
    diffs = pd.DataFrame(diffs).round(4)
    export_table(diffs, Path("artifacts/tables/olmo_stages_diffs"), formats=("csv", "parquet", "md", "tex"))
    show = ["deterministic_stance_v1", "faulborn_nli_reconstructed_as_written_seed42",
            "faulborn_nli_reconstructed_leakage_free_seed42"]
    print(table[table["evaluator_id"].isin(show)].to_string(index=False))
    print(diffs[diffs["evaluator_id"].isin(show)].to_string(index=False))


if __name__ == "__main__":
    main()
