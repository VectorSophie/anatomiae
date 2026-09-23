"""Analysis of the second-pass OLMo stage study (scripts/olmo_stages_v2.py).

Per stage x render x prompt x evaluator: position rate, lean-balanced
acquiescence and direction index (the orig+inv bank is 88 left / 88 right
by construction), empty and truncated shares. Differences with 95% item
bootstrap CIs (an item's orig and inv statements resample together):
vs Base (raw, same prompt) and vs the previous stage (same render, prompt).

Consistency: for each item, where a response to the statement and a
response to its GPT-generated opposite are both directional, are they
opposite (consistent) or the same (agrees/disagrees with both)? The
opposites are not strict negations, so "same" is soft incoherence;
reported over all pairs, over pairs the inversion audit
(artifacts/audits/faulborn_inversion_audit.csv) rates as opposites (A+B),
and over faithful opposites only (A).

Response mechanics (evaluator-independent, from the cache alone): empty
responses, immediate EOS (<=1 output token), completion within 600 and
within 200 tokens (greedy decoding makes the first 200 tokens of a 600-token
run identical to a 200-token run on the same backend, so the 200-token
view is exact, not a separate run), output-length quantiles.

Outputs: artifacts/tables/olmo_stages_v2{,_completion,_diffs,_consistency}.{csv,parquet,md,tex}
"""

from __future__ import annotations

import sys
from itertools import pairwise
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
from analyze_olmo_stages import boot_diff, metrics

from anatomiae.analysis.frame import build_analysis_frame, export_table
from anatomiae.datasets.faulborn import load_faulborn_items
from anatomiae.evaluators.taxonomy import POSITION_OUTCOMES
from anatomiae.provenance.evaluation_store import EvaluationStore
from anatomiae.provenance.generation_cache import GenerationCache

STAGE = {"allenai/OLMo-2-1124-13B": "0_base", "allenai/OLMo-2-1124-13B-SFT": "1_sft",
         "allenai/OLMo-2-1124-13B-DPO": "2_dpo", "allenai/OLMo-2-1124-13B-Instruct-RLVR2": "3_rlvr2"}
CHAT_MARK = "<|user|>"
FLIP = {"left": "right", "right": "left"}
OUT = Path("artifacts/tables/olmo_stages_v2")
REPORTED = ["position_rate", "acquiescence", "direction_index"]


def load() -> pd.DataFrame:
    recs = GenerationCache("artifacts/cache/olmo_stages_v2.jsonl").read_all()
    evals = EvaluationStore("artifacts/evaluations/olmo_stages_v2.jsonl").read_all()
    render = {r.cache_key: "native" if CHAT_MARK in r.request.rendered_text else "raw" for r in recs}
    empty = {r.cache_key: not r.raw_text.strip() for r in recs}
    df = build_analysis_frame(recs, evals)
    df["stage"] = df["model_id"].map(STAGE)
    df["render"] = df["cache_key"].map(render)
    df["empty"] = df["cache_key"].map(empty)
    df[["prompt", "polarity"]] = df["template_id"].str.split("__", expand=True)
    ref = {i.item_id: i.reference_position for i in load_faulborn_items()}
    df["item_lean"] = [ref[i] if p == "orig" else FLIP[ref[i]] for i, p in zip(df["item_id"], df["polarity"], strict=True)]
    df["is_position"] = df["outcome"].isin(POSITION_OUTCOMES)
    df["is_directional"] = df["outcome"].isin({"agreement", "disagreement"})
    df["left_aligned"] = ((df["outcome"] == "agreement") & (df["item_lean"] == "left")) | (
        (df["outcome"] == "disagreement") & (df["item_lean"] == "right"))
    return df


def mechanics() -> pd.DataFrame:
    rows = []
    for r in GenerationCache("artifacts/cache/olmo_stages_v2.jsonl").read_all():
        prompt, polarity = r.request.template_id.split("__")
        empty = not r.raw_text.strip()
        rows.append({"stage": STAGE[r.request.model_id],
                     "render": "native" if CHAT_MARK in r.request.rendered_text else "raw",
                     "prompt": prompt, "polarity": polarity, "empty": empty,
                     "immediate_eos": r.output_tokens <= 1 and r.finish_reason == "stop",
                     "completed_600": r.finish_reason == "stop" and not empty,
                     "completed_within_200": r.finish_reason == "stop" and not empty and r.output_tokens <= 200,
                     "truncated_600": r.finish_reason == "length", "output_tokens": r.output_tokens,
                     "error": r.error is not None})
    m = pd.DataFrame(rows)
    return m.groupby(["stage", "render", "prompt", "polarity"]).agg(
        n=("empty", "size"), errors=("error", "sum"), empty_rate=("empty", "mean"),
        immediate_eos_rate=("immediate_eos", "mean"), completed_within_200_rate=("completed_within_200", "mean"),
        completed_within_600_rate=("completed_600", "mean"), truncated_at_600_rate=("truncated_600", "mean"),
        output_tokens_p25=("output_tokens", lambda s: s.quantile(0.25)),
        output_tokens_median=("output_tokens", "median"),
        output_tokens_p75=("output_tokens", lambda s: s.quantile(0.75))).reset_index().round(4)


def main() -> None:
    completion = mechanics()
    export_table(completion, Path(f"{OUT}_completion"), formats=("csv", "parquet", "md", "tex"))
    df = load()
    keys = ["stage", "render", "prompt", "evaluator_id"]
    rows = [{**dict(zip(keys, k, strict=True)), **metrics(g), "empty_share": g["empty"].mean()}
            for k, g in df.groupby(keys)]
    table = pd.DataFrame(rows).round(4)
    export_table(table, OUT, formats=("csv", "parquet", "md", "tex"))

    stages = sorted(df["stage"].unique())
    diffs = []
    for (ev, prompt), g in df.groupby(["evaluator_id", "prompt"]):
        cells = {k: v for k, v in g.groupby(["stage", "render"])}
        comparisons = [(("0_base", "raw"), k, "vs_base_raw") for k in cells if k != ("0_base", "raw")]
        for render in ("native", "raw"):
            chain = [s for s in stages if (s, render) in cells]
            comparisons += [((a, render), (b, render), "vs_previous_stage") for a, b in pairwise(chain)]
        for ref_key, key, kind in comparisons:
            if ref_key not in cells:
                continue
            a, b = cells[ref_key], cells[key]
            for col in REPORTED:
                lo, hi = boot_diff(a, b, col)
                diffs.append({"evaluator_id": ev, "prompt": prompt, "comparison": kind,
                              "from": "/".join(ref_key), "to": "/".join(key), "metric": col,
                              "from_value": metrics(a)[col], "to_value": metrics(b)[col],
                              "diff": metrics(b)[col] - metrics(a)[col], "ci95_low": lo, "ci95_high": hi})
    diffs = pd.DataFrame(diffs).round(4)
    export_table(diffs, Path(f"{OUT}_diffs"), formats=("csv", "parquet", "md", "tex"))

    audit = pd.read_csv("artifacts/audits/faulborn_inversion_audit.csv").set_index("item_id")["inversion_class"]
    d = df[df["is_directional"]]
    w = d.pivot_table(index=["stage", "render", "prompt", "evaluator_id", "item_id"], columns="polarity",
                      values="outcome", aggfunc="first").dropna(subset=["orig", "inv"]).reset_index()
    w["consistent"] = w["orig"] != w["inv"]
    w["agrees_with_both"] = (w["orig"] == "agreement") & (w["inv"] == "agreement")
    w["disagrees_with_both"] = (w["orig"] == "disagreement") & (w["inv"] == "disagreement")
    w["inversion_class"] = w["item_id"].map(audit)
    parts = []
    for subset, keep in (("all_pairs", w), ("A_and_B", w[w["inversion_class"] != "C_not_an_opposite"]),
                         ("A_faithful_only", w[w["inversion_class"] == "A_faithful_opposite"])):
        c = keep.groupby(["stage", "render", "prompt", "evaluator_id"]).agg(
            n_items_both_directional=("consistent", "size"), consistent_share=("consistent", "mean"),
            agrees_with_both_share=("agrees_with_both", "mean"),
            disagrees_with_both_share=("disagrees_with_both", "mean")).reset_index()
        parts.append(c.assign(pair_subset=subset))
    cons = pd.concat(parts, ignore_index=True).round(4)
    export_table(cons, Path(f"{OUT}_consistency"), formats=("csv", "parquet", "md", "tex"))

    show = table["evaluator_id"].isin(["deterministic_stance_v1", "faulborn_nli_reconstructed_as_written_seed42"])
    print(table[show].to_string(index=False))
    print(cons[(cons["evaluator_id"] == "deterministic_stance_v1") & (cons["pair_subset"] == "A_faithful_only")].to_string(index=False))


if __name__ == "__main__":
    main()
