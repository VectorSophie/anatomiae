"""Re-read the whole OLMo v2 stage study with J-qy (post-hoc sensitivity).

Protocol: docs/experiments/olmo_v2_jqy_protocol.md. J-qy is
judge_qwen25_14b_pair_v1 from E_q_y v1, unchanged: same model revision,
prompt (JUDGE_QY), parser, vLLM/BF16 settings and greedy decoding, all
imported from scripts/eqy_experiment.py. No OLMo response is regenerated.

  export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1
  bash scripts/gpu_preflight.sh
  uv run python scripts/olmo_v2_jqy.py judge     # GPU: judge every non-empty v2 response
  uv run python scripts/olmo_v2_jqy.py score     # parse -> evaluation store
  uv run python scripts/olmo_v2_jqy.py analyze   # tables, figure
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from analyze_olmo_stages import N_BOOT, _boot_metric, _item_counts, boot_diff, metrics
from eqy_experiment import J_QY, JUDGE_DECODING, JUDGE_MODEL, JUDGE_QY, parse_judge

from anatomiae.evaluators.base import EvaluationResult
from anatomiae.inference.schema import GenerationRequest
from anatomiae.provenance.evaluation_store import EvaluationStore
from anatomiae.provenance.generation_cache import GenerationCache

V2_CACHE = "artifacts/cache/olmo_stages_v2.jsonl"
JUDGE_CACHE = "artifacts/cache/olmo_v2_jqy.jsonl"
EVAL_STORE = "artifacts/evaluations/olmo_v2_jqy.jsonl"
EQY_STORE = "artifacts/evaluations/eqy_v1.jsonl"
AUDIT = "artifacts/audits/faulborn_inversion_audit.csv"
OUT = "artifacts/tables/olmo_v2_jqy"
FIGURE = "figures/olmo_v2_jqy_direction_coverage"

OLD = ["deterministic_stance_v1",
       *[f"faulborn_nli_reconstructed_{v}_seed{s}" for v in ("as_written", "leakage_free") for s in (42, 43, 44)],
       "faulborn_nli_zeroshot"]
EVALUATORS = [J_QY, *OLD]
RELEASED = ("please_respond", "opinion")
DIRECTIONAL = {"agreement", "disagreement"}


def statement(item, polarity: str) -> str:
    """As make_labeling_sheet.py built it: original for orig, GPT opposite for inv."""
    return item.prompt_original if polarity == "orig" else item.metadata["pol_opposite_gpt"]


# ---------------------------------------------------------------- judge / score


def v2_records():
    from anatomiae.datasets.faulborn import load_faulborn_items

    items = {i.item_id: i for i in load_faulborn_items()}
    recs = GenerationCache(V2_CACHE).read_all()
    return recs, {r.cache_key: statement(items[r.request.item_id], r.request.template_id.split("__")[1])
                  for r in recs}


def cmd_judge(gpu_memory_utilization: float) -> None:
    from transformers import AutoTokenizer

    model_id, revision = JUDGE_MODEL
    tok = AutoTokenizer.from_pretrained(model_id, revision=revision, local_files_only=True)
    recs, q = v2_records()
    cache = GenerationCache(JUDGE_CACHE)
    requests = []
    for r in recs:
        if r.error is not None or not r.raw_text.strip():
            continue  # not judged: scored malformed/generation_error, like every other evaluator
        rendered = tok.apply_chat_template(
            [{"role": "user", "content": JUDGE_QY.format(q=q[r.cache_key], y=r.raw_text)}],
            add_generation_prompt=True, tokenize=False)
        req = GenerationRequest(
            rendered_prompt_hash=hashlib.sha256(rendered.encode("utf-8")).hexdigest(), rendered_text=rendered,
            model_id=model_id, model_revision=revision, tokenizer_revision=revision, backend="vllm",
            precision="bf16", decoding=JUDGE_DECODING, item_id=r.request.item_id,
            variant_id=f"judged::{r.cache_key}", template_id=J_QY)
        if not cache.has(req.cache_key()):
            requests.append(req)
    log = {"model_id": model_id, "revision": revision, "gpu_memory_utilization": gpu_memory_utilization,
           "n_v2_records": len(recs), "n_to_generate": len(requests)}
    if requests:
        from anatomiae.inference.backends import VLLMBackend

        backend = VLLMBackend(model_id, revision=revision, precision="bf16",
                              gpu_memory_utilization=gpu_memory_utilization)
        records = backend.generate_many(requests)
        for rec in records:
            cache.append(rec)
        log.update(n_errors=sum(rec.error is not None for rec in records),
                   n_malformed=sum(parse_judge(rec.raw_text)[0] == "malformed" for rec in records),
                   n_length=sum(rec.finish_reason == "length" for rec in records),
                   gpu_power_limit_w=backend._gpu_provenance.power_limit_w,
                   driver_version=backend._gpu_provenance.driver_version)
    log["ts"] = time.time()
    Path("artifacts/logs/olmo_v2_jqy_judge.json").write_text(json.dumps(log, indent=2))
    print(json.dumps(log, indent=2))


def cmd_score() -> None:
    recs, _ = v2_records()
    judged = {rec.request.variant_id: rec for rec in GenerationCache(JUDGE_CACHE).read_all()}
    out = []
    for r in recs:
        base = {"generation_cache_key": r.cache_key, "evaluator_id": J_QY, "evaluator_version": "1.0.0"}
        if r.error is not None:
            out.append(EvaluationResult(**base, outcome="generation_error", notes=r.error))
        elif not r.raw_text.strip():
            out.append(EvaluationResult(**base, outcome="malformed", notes="empty raw_text"))
        else:
            j = judged[f"judged::{r.cache_key}"]
            outcome, native = parse_judge(j.raw_text) if j.error is None else ("generation_error", None)
            out.append(EvaluationResult(**base, outcome=outcome, native_label=native,
                                        notes=json.dumps({"judge_cache_key": j.cache_key, "raw": j.raw_text})))
    store = EvaluationStore(EVAL_STORE)
    new = [e for e in out if not store.has(e.generation_cache_key, e.evaluator_id, e.evaluator_version)]
    print(f"appended {store.append_many(new)} of {len(out)} J-qy results to {EVAL_STORE}")


# ---------------------------------------------------------------- analyze


def coverage(g: pd.DataFrame, unclear: set[str]) -> dict:
    o = g["outcome"]
    return {"directional_rate": o.isin(DIRECTIONAL).mean(), "mixed_rate": (o == "mixed_or_conditional").mean(),
            "neutral_rate": (o == "neutral_or_no_position").mean(),
            "unclear_rate": g["cache_key"].isin(unclear).mean() if len(unclear) else 0.0,
            "other_rate": (~o.isin(DIRECTIONAL | {"mixed_or_conditional", "neutral_or_no_position"})).mean()}


def did_ci(cells: list[pd.DataFrame]) -> tuple[float, float, float]:
    """(b1-a1) - (b2-a2) on direction_index, all four cells resampled with the same items."""
    a1, b1, a2, b2 = cells
    point = ((metrics(b1)["direction_index"] - metrics(a1)["direction_index"])
             - (metrics(b2)["direction_index"] - metrics(a2)["direction_index"]))
    items = sorted(set().union(*(c["item_id"] for c in cells)))
    counts = [_item_counts(c, items) for c in cells]
    idx = np.random.default_rng(0).integers(0, len(items), size=(N_BOOT, len(items)))
    with np.errstate(invalid="ignore", divide="ignore"):
        m = [_boot_metric(c, idx, "direction_index") for c in counts]
        v = (m[1] - m[0]) - (m[3] - m[2])
    v = v[~np.isnan(v)]
    if len(v) < 0.9 * N_BOOT:
        return point, float("nan"), float("nan")
    return point, float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))


def question_rows(df: pd.DataFrame, unclear: set[str], subset: str) -> list[dict]:
    rows = []
    cell = {k: g for k, g in df.groupby(["evaluator_id", "stage", "render", "prompt"])}

    def add(question, ev, render, a_key, b_key, prompt_label):
        a, b = cell.get((ev, *a_key)), cell.get((ev, *b_key))
        if a is None or b is None:
            return
        ma, mb = metrics(a), metrics(b)
        lo, hi = boot_diff(a, b, "direction_index")
        ca, cb = coverage(a, unclear), coverage(b, unclear)
        rows.append({"question": question, "subset": subset, "evaluator_id": ev, "render": render,
                     "prompt": prompt_label, "from": "/".join(a_key), "to": "/".join(b_key),
                     "from_direction_index": ma["direction_index"], "to_direction_index": mb["direction_index"],
                     "diff": mb["direction_index"] - ma["direction_index"], "ci95_low": lo, "ci95_high": hi,
                     "from_directional_n": ma["directional_n"], "to_directional_n": mb["directional_n"],
                     "from_directional_rate": ca["directional_rate"], "to_directional_rate": cb["directional_rate"],
                     "from_mixed_rate": ca["mixed_rate"], "to_mixed_rate": cb["mixed_rate"],
                     "from_neutral_rate": ca["neutral_rate"], "to_neutral_rate": cb["neutral_rate"]})

    for ev in EVALUATORS:
        for render in ("native", "raw"):
            for prompt in (*RELEASED, "stance_first"):
                add("A_sft_to_dpo", ev, render, ("1_sft", render, prompt), ("2_dpo", render, prompt), prompt)
                add("C_dpo_to_rlvr2", ev, render, ("2_dpo", render, prompt), ("3_rlvr2", render, prompt), prompt)
            for stage in ("0_base", "1_sft", "2_dpo", "3_rlvr2"):
                for prompt in RELEASED:
                    add("B_released_minus_stance_first", ev, render, (stage, render, "stance_first"),
                        (stage, render, prompt), f"{prompt}_minus_stance_first@{stage}")
            for question, s0, s1 in (("A_did", "1_sft", "2_dpo"), ("C_did", "2_dpo", "3_rlvr2")):
                for prompt in RELEASED:
                    keys = [(ev, s0, render, prompt), (ev, s1, render, prompt),
                            (ev, s0, render, "stance_first"), (ev, s1, render, "stance_first")]
                    if all(k in cell for k in keys):
                        p, lo, hi = did_ci([cell[k] for k in keys])
                        rows.append({"question": question, "subset": subset, "evaluator_id": ev, "render": render,
                                     "prompt": f"{prompt}_minus_stance_first", "from": f"{s0}/{render}",
                                     "to": f"{s1}/{render}", "diff": p, "ci95_low": lo, "ci95_high": hi})
    return rows


def transitions(df: pd.DataFrame) -> pd.DataFrame:
    w = df.pivot_table(index=["cache_key", "stage", "render", "prompt"], columns="evaluator_id",
                       values="outcome", aggfunc="first").reset_index()
    rows = []
    for ev in OLD:
        j, o = w[J_QY], w[ev]
        kind = np.select(
            [j.isin(DIRECTIONAL) & (j == o), j.isin(DIRECTIONAL) & o.isin(DIRECTIONAL),
             j.isin(DIRECTIONAL) & ~o.isin(DIRECTIONAL), ~j.isin(DIRECTIONAL) & o.isin(DIRECTIONAL), j == o],
            ["same_direction", "polarity_change", "jqy_commits_old_not", "old_commits_jqy_not",
             "same_nondirectional"], "different_nondirectional")
        t = w.assign(kind=kind).groupby(["stage", "render", "prompt", "kind"]).size().unstack(fill_value=0)
        t = t.reindex(columns=["same_direction", "polarity_change", "jqy_commits_old_not", "old_commits_jqy_not",
                               "same_nondirectional", "different_nondirectional"], fill_value=0)
        t.insert(0, "n", t.sum(1))
        rows.append(t.reset_index().assign(old_evaluator=ev))
    return pd.concat(rows, ignore_index=True)


def consistency(df: pd.DataFrame) -> pd.DataFrame:
    audit = pd.read_csv(AUDIT).set_index("item_id")["inversion_class"]
    d = df[df["is_directional"]]
    w = d.pivot_table(index=["stage", "render", "prompt", "evaluator_id", "item_id"], columns="polarity",
                      values="outcome", aggfunc="first").dropna(subset=["orig", "inv"]).reset_index()
    w["consistent"] = w["orig"] != w["inv"]
    w["agrees_with_both"] = (w["orig"] == "agreement") & (w["inv"] == "agreement")
    w["inversion_class"] = w["item_id"].map(audit)
    parts = []
    for subset, keep in (("all_pairs", w), ("A_and_B", w[w["inversion_class"] != "C_not_an_opposite"]),
                         ("A_faithful_only", w[w["inversion_class"] == "A_faithful_opposite"])):
        c = keep.groupby(["stage", "render", "prompt", "evaluator_id"]).agg(
            n_items_both_directional=("consistent", "size"), consistent_share=("consistent", "mean"),
            agrees_with_both_share=("agrees_with_both", "mean")).reset_index()
        parts.append(c.assign(pair_subset=subset))
    return pd.concat(parts, ignore_index=True)


def overlap() -> pd.DataFrame:
    full = {e.generation_cache_key: e for e in EvaluationStore(EVAL_STORE).read_all()}
    val = [e for e in EvaluationStore(EQY_STORE).read_all() if e.evaluator_id == J_QY]
    same = [full[e.generation_cache_key].outcome == e.outcome for e in val]
    raw = [json.loads(full[e.generation_cache_key].notes)["raw"] == json.loads(e.notes)["raw"] for e in val]
    return pd.DataFrame([{"n_validation_rows": len(val), "same_outcome": sum(same), "same_raw_judge_text": sum(raw)}])


def figure(metrics_table: pd.DataFrame) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    shown = {J_QY: "J-qy (sees statement)", "deterministic_stance_v1": "rule-based",
             "faulborn_nli_reconstructed_as_written_seed42": "recon as-written s42",
             "faulborn_nli_reconstructed_leakage_free_seed42": "recon leakage-free s42",
             "faulborn_nli_zeroshot": "zero-shot"}
    stages = ["1_sft", "2_dpo", "3_rlvr2"]
    prompts = ["stance_first", "please_respond", "opinion"]
    m = metrics_table[(metrics_table.render == "native") & metrics_table.evaluator_id.isin(shown)]
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.5), sharex=True)
    for j, prompt in enumerate(prompts):
        for ev, label in shown.items():
            r = m[(m.prompt == prompt) & (m.evaluator_id == ev)].set_index("stage").reindex(stages)
            style = {"lw": 2.5, "color": "black"} if ev == J_QY else {"lw": 1, "alpha": 0.8}
            axes[0, j].plot(range(3), r["direction_index"], marker="o", label=label, **style)
            axes[1, j].plot(range(3), r["directional_rate"], marker="o", label=label, **style)
        axes[0, j].set_title(f"chat / {prompt}", fontsize=9)
        axes[0, j].set_ylim(-0.2, 1.0)
        axes[1, j].set_ylim(0, 1.05)
        axes[1, j].set_xticks(range(3))
        axes[1, j].set_xticklabels(["SFT", "DPO", "RLVR2"])
    axes[0, 0].set_ylabel("direction index\n(agree L − agree R, among directional)")
    axes[1, 0].set_ylabel("directional rate\n(share of all responses)")
    h, lab = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=5, fontsize=8, frameon=False)
    fig.suptitle("OLMo-2-13B v2, chat render: direction index with its commitment coverage, by evaluator",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    Path(FIGURE).parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"{FIGURE}.{ext}", dpi=200)
    plt.close(fig)


def cmd_analyze() -> None:
    from analyze_olmo_stages_v2 import load

    df = load(extra_evaluations=(EVAL_STORE,))
    df = df[df["evaluator_id"].isin(EVALUATORS)]
    unclear = {e.generation_cache_key for e in EvaluationStore(EVAL_STORE).read_all() if e.native_label == "unclear"}
    assert df.groupby("evaluator_id")["cache_key"].nunique().eq(3696).all(), "every evaluator must cover the cache"

    keys = ["stage", "render", "prompt", "evaluator_id"]
    mt = pd.DataFrame([{**dict(zip(keys, k, strict=True)), **metrics(g),
                        **coverage(g, unclear if k[3] == J_QY else set())} for k, g in df.groupby(keys)])

    audit = pd.read_csv(AUDIT).set_index("item_id")["inversion_class"]
    class_a = df[df["item_id"].map(audit) == "A_faithful_opposite"]
    questions = pd.DataFrame(question_rows(df, unclear, "all_items") + question_rows(class_a, unclear, "class_A_items"))

    Path(OUT).parent.mkdir(parents=True, exist_ok=True)
    mt.round(4).to_csv(f"{OUT}_metrics.csv", index=False)
    questions.round(4).to_csv(f"{OUT}_questions.csv", index=False)
    consistency(df).round(4).to_csv(f"{OUT}_consistency.csv", index=False)
    transitions(df).to_csv(f"{OUT}_transitions.csv", index=False)
    ov = overlap()
    ov.to_csv(f"{OUT}_overlap.csv", index=False)

    q = questions[(questions.render == "native") & (questions.subset == "all_items")]
    cols = ["question", "evaluator_id", "prompt", "from", "to", "diff", "ci95_low", "ci95_high",
            "from_directional_n", "to_directional_n", "from_directional_rate", "to_directional_rate"]
    Path(f"{OUT}_questions.md").write_text(
        "<!-- generated by scripts/olmo_v2_jqy.py analyze; do not edit. chat render, all items -->\n\n"
        + q[cols].round(3).to_markdown(index=False) + "\n", encoding="utf-8")
    figure(mt)
    print(ov.to_string(index=False))
    print(q[q.evaluator_id == J_QY][cols].round(3).to_string(index=False))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    j = sub.add_parser("judge")
    j.add_argument("--gpu-memory-utilization", type=float, default=0.5)
    sub.add_parser("score")
    sub.add_parser("analyze")
    args = ap.parse_args()
    if args.cmd == "judge":
        cmd_judge(args.gpu_memory_utilization)
    elif args.cmd == "score":
        cmd_score()
    else:
        cmd_analyze()


if __name__ == "__main__":
    main()
