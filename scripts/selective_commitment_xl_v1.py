"""Selective commitment: cross-lineage replication v1 (Qwen2.5-14B family).

Amendment (frozen before generation): docs/experiments/selective_commitment_crosslineage_v1.md.
Everything not named there is identical to selective_commitment_v1 and is
imported from scripts/selective_commitment_v1.py rather than re-implemented.

  export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1
  bash scripts/gpu_preflight.sh
  uv run python scripts/selective_commitment_xl_v1.py generate --arm q_inst   # then q_r1
  uv run python scripts/selective_commitment_xl_v1.py judge
  uv run python scripts/selective_commitment_xl_v1.py score
  uv run python scripts/selective_commitment_xl_v1.py analyze
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import selective_commitment_v1 as sc

from anatomiae.datasets.faulborn import RELEASED_JOINER, load_faulborn_items
from anatomiae.inference.schema import DecodingConfig, GenerationRecord, GenerationRequest
from anatomiae.provenance.generation_cache import GenerationCache

ARMS = {
    "q_inst": ("q_inst", "Qwen/Qwen2.5-14B-Instruct", "cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8", 600),
    "q_r1": ("q_r1", "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B", "1df8507178afcc1bef68cd8c393f61a886323761", 4096),
}
REASONING_MODELS = {"deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"}
CACHE = "artifacts/cache/selective_commitment_xl_v1.jsonl"
JUDGE_CACHE = "artifacts/cache/selective_commitment_xl_v1_jqy.jsonl"
EVAL_STORE = "artifacts/evaluations/selective_commitment_xl_v1.jsonl"
OUT = "artifacts/tables/selective_commitment_xl_v1"
FIGURE = "figures/selective_commitment_xl_v1"
THINK_END = "</think>"


def answer_text(record: GenerationRecord) -> str:
    """The judged response. Reasoning models: text after the last </think>;
    no </think> (budget exhausted while reasoning) -> no answer -> ''."""
    if record.request.model_id not in REASONING_MODELS:
        return record.raw_text
    if THINK_END not in record.raw_text:
        return ""
    return record.raw_text.rsplit(THINK_END, 1)[1].strip()


class AnswerCache(GenerationCache):
    """Read-only view of the xl generation cache with raw_text replaced by the
    judged answer text. The cache file itself is never modified."""

    def read_all(self) -> list[GenerationRecord]:
        return [r.model_copy(update={"raw_text": answer_text(r)}) for r in super().read_all()]


def _cache_factory(path):
    return AnswerCache(path) if str(path) == CACHE else GenerationCache(path)


def bind_v1() -> None:
    """Point the frozen v1 functions at the xl artifacts."""
    sc.CACHE, sc.JUDGE_CACHE, sc.EVAL_STORE = CACHE, JUDGE_CACHE, EVAL_STORE
    sc.JUDGE_LOG = "artifacts/logs/selective_commitment_xl_v1_judge.jsonl"
    sc.STAGE_OF_MODEL = {model: arm for arm, model, _, _ in ARMS.values()}
    sc.GenerationCache = _cache_factory


def cmd_generate(arm_key: str, gpu_mem: float) -> None:
    from transformers import AutoTokenizer

    from anatomiae.prompts.constructor import build_variant, render_prompt

    sc.check_prompts_are_new()
    _, model_id, revision, max_new = ARMS[arm_key]
    decoding = DecodingConfig(temperature=0.0, max_new_tokens=max_new, seed=0)
    tok = AutoTokenizer.from_pretrained(model_id, revision=revision, local_files_only=True)

    def chat_render(text: str) -> str:
        return tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True,
                                       tokenize=False)

    requests = []
    for item in load_faulborn_items():
        for polarity in ("orig", "inv"):
            for pid, prefix in sc.PROMPTS.items():
                perturbation = "inversion" if polarity == "inv" else "framing" if pid == "stance_first" else "canonical"
                variant = build_variant(
                    item, perturbation_type=perturbation,
                    variant_text=f"{prefix}{RELEASED_JOINER}{sc.statement(item, polarity)}",
                    template_id=f"{pid}__{polarity}",
                    semantic_equivalence_status="equivalent" if perturbation == "framing" else None)
                rendered = render_prompt(variant, render_mode="chat_instruct", chat_render_fn=chat_render)
                requests.append(GenerationRequest(
                    rendered_prompt_hash=rendered.rendered_prompt_hash, rendered_text=rendered.rendered_text,
                    model_id=model_id, model_revision=revision, tokenizer_revision=revision, backend="vllm",
                    precision="bf16", decoding=decoding, item_id=item.item_id, variant_id=variant.variant_id,
                    template_id=variant.template_id))
    cache = GenerationCache(CACHE)
    todo = [r for r in requests if not cache.has(r.cache_key())]
    print(f"{arm_key}@{revision[:8]}: {len(requests)} requests, {len(todo)} to generate", flush=True)
    if not todo:
        return
    from anatomiae.inference.backends import VLLMBackend

    backend = VLLMBackend(model_id, revision=revision, precision="bf16", gpu_memory_utilization=gpu_mem)
    new, t0 = [], time.time()
    for i in range(0, len(todo), sc.CHUNK):
        records = backend.generate_many(todo[i:i + sc.CHUNK])
        for record in records:
            cache.append(record)
        new += records
        print(f"  {len(new)}/{len(todo)}", flush=True)
    summary = {"arm": arm_key, "model_id": model_id, "revision": revision, "max_new_tokens": max_new,
               "chat_template_sha256": __import__("hashlib").sha256(tok.chat_template.encode()).hexdigest(),
               "batch_size": sc.CHUNK, "gpu_memory_utilization": gpu_mem, "n_new_generations": len(new),
               "n_errors": sum(r.error is not None for r in new), "n_empty": sum(not r.raw_text.strip() for r in new),
               "n_no_answer": sum(not answer_text(r) for r in new),
               "n_truncated": sum(r.finish_reason == "length" for r in new), "wall_seconds": time.time() - t0,
               "gpu_power_limit_w": backend._gpu_provenance.power_limit_w,
               "driver_version": backend._gpu_provenance.driver_version, "ts": time.time()}
    with Path("artifacts/logs/selective_commitment_xl_v1_generate.jsonl").open("a") as f:
        f.write(json.dumps(summary) + "\n")
    print(json.dumps(summary, indent=2))


def cmd_analyze() -> None:
    sc.check_prompts_are_new()
    df = sc.frame()
    arms = [a for a in ("q_inst", "q_r1") if a in set(df["stage"])]
    subsets = {"class_A": sc.CLASS_A, "all_items": sorted(df["item_id"].unique())}
    primary, decomp, secondary, item_model, contrast = [], [], [], [], []
    for subset, ids in subsets.items():
        for a in arms:
            u = sc.units(df, a)
            u = u[u["item_id"].isin(ids)]
            primary.append({"subset": subset, "arm": a, "aggregation": "pooled_5_prompts", **sc.gap(u)})
            decomp.append({"subset": subset, **sc.decomposition(df, u, a)})
            for pid in sc.FREE_FORM:
                secondary.append({"analysis": "per_prompt_gap", "subset": subset, "arm": a, "prompt": pid,
                                  **sc.gap(u[u["prompt"] == pid])})
            ud = sc.units(df, a, sf_evaluator=sc.DET)
            secondary.append({"analysis": "gap_with_deterministic_SF", "subset": subset, "arm": a,
                              "prompt": "pooled", **sc.gap(ud[ud["item_id"].isin(ids)])})
            k = u.groupby(["item_id", "polarity"]).agg(k_commit=("C", "sum"), sf=("sf_outcome", "first"),
                                                       lean=("item_lean", "first")).reset_index()
            k["sf_pole"] = ["non_directional" if o not in sc.DIRECTIONAL else
                            "left_pole" if (o == "agreement") == (le == "left") else "right_pole"
                            for o, le in zip(k["sf"], k["lean"], strict=True)]
            for cat, g in k.groupby("sf_pole"):
                item_model.append({"subset": subset, "arm": a, "sf_pole": cat, "n_item_polarities": len(g),
                                   "mean_k_commit_of_5": g["k_commit"].mean(),
                                   "share_never_committed": (g["k_commit"] == 0).mean()})
        if len(arms) == 2:
            contrast += sc.stage_rows(df, subset, ids, pairs=(("q_inst", "q_r1"),))
    j = df[df["evaluator"] == "jqy"]
    coverage = j.groupby(["stage", "prompt"]).apply(lambda g: pd.Series({
        "n": len(g), "directional_rate": g["outcome"].isin(sc.DIRECTIONAL).mean(),
        "mixed_rate": (g["outcome"] == "mixed_or_conditional").mean(),
        "neutral_rate": (g["outcome"] == "neutral_or_no_position").mean(),
        "unclear_rate": (g["native_label"] == "unclear").mean(),
        "malformed_or_error_rate": g["outcome"].isin({"malformed", "generation_error"}).mean(),
        "no_answer_rate": g["empty"].mean(), "truncated_rate": g["truncated"].mean()}),
        include_groups=False).reset_index().rename(columns={"stage": "arm"})

    Path(OUT).parent.mkdir(parents=True, exist_ok=True)
    tables = {"primary": pd.DataFrame(primary), "decomposition": pd.DataFrame(decomp),
              "sibling_contrast": pd.DataFrame(contrast), "secondary": pd.DataFrame(secondary),
              "item_model": pd.DataFrame(item_model), "coverage": coverage}
    for name, t in tables.items():
        t.round(4).to_csv(f"{OUT}_{name}.csv", index=False)
    Path(f"{OUT}_primary.md").write_text(
        "<!-- generated by scripts/selective_commitment_xl_v1.py analyze; do not edit -->\n\n"
        "## Primary (decision: class_A, q_inst)\n\n" + tables["primary"].round(3).to_markdown(index=False)
        + "\n\n## Decomposition\n\n" + tables["decomposition"].round(3).to_markdown(index=False)
        + "\n\n## Sibling contrast (descriptive)\n\n" + tables["sibling_contrast"].round(3).to_markdown(index=False)
        + "\n", encoding="utf-8")
    figure(tables["primary"], tables["decomposition"], coverage, arms)
    for name in ("primary", "decomposition", "sibling_contrast"):
        print(tables[name].round(3).to_string(index=False))


def figure(primary: pd.DataFrame, decomp: pd.DataFrame, coverage: pd.DataFrame, arms: list[str]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = {"q_inst": "Qwen2.5-14B-Instruct", "q_r1": "R1-Distill-Qwen-14B"}
    d = decomp[decomp["subset"] == "class_A"].set_index("stage").reindex(arms)
    p = primary[primary["subset"] == "class_A"].set_index("arm").reindex(arms)
    x = range(len(arms))
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.3))
    for col, lab, st in (("D_SF_committed", "stance-first, committed units", "o"),
                         ("D_SF_uncommitted", "stance-first, uncommitted units", "s"),
                         ("D_SF_all", "stance-first, all items", "^"),
                         ("D_ff_committed", "free-form, committed responses", "d")):
        axes[0].plot(x, d[col], st, label=lab, markersize=8)
    axes[0].set_ylim(-0.2, 1.0)
    axes[0].set_ylabel("direction index (J-qy)")
    axes[0].set_title("Direction by subset (class A, pooled)", fontsize=9)
    axes[0].legend(fontsize=7)
    axes[1].errorbar(x, p["selection_gap"], yerr=[p["selection_gap"] - p["ci95_low"], p["ci95_high"] - p["selection_gap"]],
                     fmt="o", capsize=4, color="black")
    axes[1].axhline(0, color="grey", lw=0.8)
    axes[1].set_title("Selection gap, 95% item-bootstrap CI (class A)", fontsize=9)
    for pid in [*sc.FREE_FORM, "stance_first"]:
        c = coverage[coverage["prompt"] == pid].set_index("arm").reindex(arms)
        axes[2].plot(x, c["directional_rate"], "k^--" if pid == "stance_first" else "o-", label=pid)
    axes[2].set_ylim(0, 1.05)
    axes[2].set_title("Commitment coverage (J-qy directional rate, all items)", fontsize=9)
    axes[2].legend(fontsize=7)
    for ax in axes:
        ax.set_xticks(list(x))
        ax.set_xticklabels([labels[a] for a in arms], fontsize=8)
        ax.set_xlim(-0.5, len(arms) - 0.5)
    fig.tight_layout()
    Path(FIGURE).parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"{FIGURE}.{ext}", dpi=200)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate")
    g.add_argument("--arm", choices=sorted(ARMS), required=True)
    g.add_argument("--gpu-mem", type=float, default=0.85)
    j = sub.add_parser("judge")
    j.add_argument("--gpu-mem", type=float, default=0.5)
    sub.add_parser("score")
    sub.add_parser("analyze")
    args = ap.parse_args()
    bind_v1()
    if args.cmd == "generate":
        cmd_generate(args.arm, args.gpu_mem)
    elif args.cmd == "judge":
        sc.cmd_judge(args.gpu_mem)
    elif args.cmd == "score":
        sc.cmd_score()
    else:
        cmd_analyze()


if __name__ == "__main__":
    main()
