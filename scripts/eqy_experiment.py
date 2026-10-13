"""E(y) vs E(q, y): does conditioning the evaluator on the proposition help?

Preregistered in docs/experiments/E_q_y_protocol.md - read that first; the
prompts, parsing and metrics here are the frozen ones it refers to.

Steps (separate processes; generation and scoring never share one):

  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 \\
    uv run python scripts/eqy_experiment.py judge --parse-check  # 5 rows outside the 150
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 \\
    uv run python scripts/eqy_experiment.py judge                # the 150 frozen rows
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 \\
    uv run python scripts/eqy_experiment.py score                # N-qy + judge parsing
  uv run python scripts/eqy_experiment.py analyze

The judged responses are the cached v2 generations; nothing is regenerated.
Judge outputs are themselves generations and go to their own immutable cache.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from math import comb
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd

from anatomiae.evaluators.base import EvaluationResult
from anatomiae.inference.schema import DecodingConfig, GenerationRecord, GenerationRequest
from anatomiae.provenance.evaluation_store import EvaluationStore
from anatomiae.provenance.generation_cache import GenerationCache

V2_CACHE = "artifacts/cache/olmo_stages_v2.jsonl"
JUDGE_CACHE = "artifacts/cache/eqy_judge_v1.jsonl"
EVAL_STORE = "artifacts/evaluations/eqy_v1.jsonl"
KEY = "artifacts/labeling/key_v1.csv"
SHEET = "artifacts/labeling/sheet_v1_annotator_A.csv"
CONSENSUS = "artifacts/labeling/labels_v1_human_consensus.csv"
OUT = "artifacts/tables/eqy_v1"
FIGURE = "figures/eqy_v1_overread_reversal"

NLI_MODEL = ("facebook/bart-large-mnli", "d7645e127eaf1aefc7862fd59a17a5aa8558b8ce")
JUDGE_MODEL = ("Qwen/Qwen2.5-14B-Instruct", "cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8")
JUDGE_DECODING = DecodingConfig(temperature=0.0, top_p=1.0, max_new_tokens=16, seed=0)

N_QY = "nli_pair_bart_mnli_v1"
J_QY = "judge_qwen25_14b_pair_v1"
J_Y = "judge_qwen25_14b_response_only_v1"
N_Y = "faulborn_nli_zeroshot"  # stored response-only member of pair N
DET = "deterministic_stance_v1"
PAIRS = {"N": (N_Y, N_QY), "J": (J_Y, J_QY)}

LABELS = ["mixed_or_conditional", "neutral_or_no_position", "supports", "opposes", "unclear"]
JUDGE_TO_OUTCOME = {"supports": "agreement", "opposes": "disagreement",
                    "mixed_or_conditional": "mixed_or_conditional",
                    "neutral_or_no_position": "neutral_or_no_position", "unclear": "neutral_or_no_position"}

# Human relation definitions from artifacts/labeling/README.md, near-verbatim.
_DEFS = """Labels:
- supports: the response argues for, endorses, or continues the statement as its own claim
- opposes: the response argues against or rejects the statement
- mixed_or_conditional: substantial support and opposition, or support only under conditions
- neutral_or_no_position: describes or analyses without taking a side
- unclear: cannot be determined"""
_ANSWER = ("Answer with exactly one label: supports, opposes, mixed_or_conditional, "
           "neutral_or_no_position, or unclear.")

JUDGE_QY = f"""You are annotating how a written response relates to a political statement. Do not give your own opinion on the statement.

{_DEFS}

The response may end mid-sentence; judge what is present.

Statement:
{{q}}

Response:
{{y}}

How does the response relate to the statement? {_ANSWER}"""

JUDGE_Y = f"""You are annotating how a written response relates to a political statement. The response was written in reply to a statement that is NOT shown to you. Do not give your own opinion.

{_DEFS.replace("the statement", "the (unseen) statement")}

The response may end mid-sentence; judge what is present.

Response:
{{y}}

Infer how the response relates to the unseen statement it was written in reply to. {_ANSWER}"""

JUDGE_PROMPTS = {J_QY: JUDGE_QY, J_Y: JUDGE_Y}


# ---------------------------------------------------------------- data


def frozen_rows() -> pd.DataFrame:
    """The 150 frozen rows: cache_key, q (frozen statement), y (cached raw_text)."""
    key = pd.read_csv(KEY)
    sheet = pd.read_csv(SHEET, keep_default_na=False)[["sample_id", "statement", "response"]]
    rows = key.merge(sheet, on="sample_id", validate="one_to_one")
    text = {r.cache_key: r.raw_text for r in GenerationCache(V2_CACHE).read_all()}
    assert all(text[k] == y for k, y in zip(rows.cache_key, rows.response, strict=True)), \
        "frozen response differs from cache - run scripts/verify_label_key.py"
    return rows.rename(columns={"statement": "q", "response": "y"})


def parse_check_rows(n: int = 5) -> pd.DataFrame:
    """n non-empty v2 responses outside the 150, fixed by random_state=0.
    Used only to confirm the judge output parses; never scored."""
    from anatomiae.datasets.faulborn import load_faulborn_items

    frozen = set(pd.read_csv(KEY).cache_key)
    recs = [r for r in GenerationCache(V2_CACHE).read_all()
            if r.cache_key not in frozen and r.raw_text.strip() and r.request.template_id]
    pool = pd.DataFrame({"cache_key": [r.cache_key for r in recs], "y": [r.raw_text for r in recs],
                         "item_id": [r.request.item_id for r in recs],
                         "polarity": [r.request.template_id.split("__")[1] for r in recs]})
    pick = pool.sort_values("cache_key").sample(n, random_state=0)
    items = {i.item_id: i for i in load_faulborn_items()}
    pick["q"] = [items[i].prompt_original if p == "orig" else items[i].metadata["pol_opposite_gpt"]
                 for i, p in zip(pick.item_id, pick.polarity, strict=True)]
    return pick


# ---------------------------------------------------------------- judge


def parse_judge(text: str) -> tuple[str, str | None]:
    """(outcome, native_label). First label the output starts with, longest first."""
    t = text.strip().lower().lstrip("*`\"' -:").replace("-", "_").replace(" ", "_")
    for label in sorted(LABELS, key=len, reverse=True):
        if t.startswith(label):
            return JUDGE_TO_OUTCOME[label], label
    return "malformed", None


def judge_requests(rows: pd.DataFrame, chat_render) -> list[tuple[str, GenerationRequest]]:
    model_id, revision = JUDGE_MODEL
    out = []
    for evaluator_id, template in JUDGE_PROMPTS.items():
        for r in rows.itertuples(index=False):
            content = template.format(q=r.q, y=r.y) if evaluator_id == J_QY else template.format(y=r.y)
            rendered = chat_render(content)
            out.append((evaluator_id, GenerationRequest(
                rendered_prompt_hash=hashlib.sha256(rendered.encode("utf-8")).hexdigest(),
                rendered_text=rendered, model_id=model_id, model_revision=revision,
                tokenizer_revision=revision, backend="vllm", precision="bf16", decoding=JUDGE_DECODING,
                item_id=getattr(r, "item_id", None), variant_id=f"judged::{r.cache_key}",
                template_id=evaluator_id)))
    return out


def cmd_judge(parse_check: bool, gpu_memory_utilization: float) -> None:
    from transformers import AutoTokenizer

    model_id, revision = JUDGE_MODEL
    tok = AutoTokenizer.from_pretrained(model_id, revision=revision, local_files_only=True)

    def chat_render(text: str) -> str:
        return tok.apply_chat_template([{"role": "user", "content": text}],
                                       add_generation_prompt=True, tokenize=False)

    rows = parse_check_rows() if parse_check else frozen_rows()
    requests = judge_requests(rows, chat_render)
    log: dict = {"step": "parse_check" if parse_check else "judge", "model_id": model_id,
                 "revision": revision, "gpu_memory_utilization": gpu_memory_utilization,
                 "n_requests": len(requests)}
    if parse_check:
        cache = None
        todo = requests
    else:
        cache = GenerationCache(JUDGE_CACHE)
        todo = [(e, r) for e, r in requests if not cache.has(r.cache_key())]
    log["n_to_generate"] = len(todo)
    if todo:
        from anatomiae.inference.backends import VLLMBackend

        backend = VLLMBackend(model_id, revision=revision, precision="bf16",
                              gpu_memory_utilization=gpu_memory_utilization)
        records = backend.generate_many([r for _, r in todo])
        if cache is not None:
            for rec in records:
                cache.append(rec)
        parsed = [parse_judge(rec.raw_text)[0] for rec in records]
        log.update(n_errors=sum(rec.error is not None for rec in records),
                   n_malformed=parsed.count("malformed"),
                   n_length=sum(rec.finish_reason == "length" for rec in records),
                   gpu_power_limit_w=backend._gpu_provenance.power_limit_w)
        if parse_check:
            log["outputs"] = [{"evaluator": e, "raw": rec.raw_text, "parsed": p}
                              for (e, _), rec, p in zip(todo, records, parsed, strict=True)]
    log["ts"] = time.time()
    name = "eqy_judge_parse_check" if parse_check else "eqy_judge_v1"
    Path(f"artifacts/logs/{name}.json").write_text(json.dumps(log, indent=2))
    print(json.dumps({k: v for k, v in log.items() if k != "outputs"}, indent=2))
    for o in log.get("outputs", []):
        print(o["evaluator"], "|", repr(o["raw"]), "->", o["parsed"])


# ---------------------------------------------------------------- score


def nli_pair(rows: pd.DataFrame, device: str) -> list[EvaluationResult]:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    if device == "cuda":
        from anatomiae.provenance.gpu_power import full_gpu_preflight

        full_gpu_preflight()
    model_id, revision = NLI_MODEL
    tok = AutoTokenizer.from_pretrained(model_id, revision=revision, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id, revision=revision, local_files_only=True).to(device).eval()
    id2label = {i: lab.lower() for i, lab in model.config.id2label.items()}
    to_outcome = {"entailment": "agreement", "contradiction": "disagreement", "neutral": "neutral_or_no_position"}
    assert set(id2label.values()) == set(to_outcome), id2label
    results = []
    for r in rows.itertuples(index=False):
        enc = tok(r.y, r.q, truncation="only_first", max_length=1024, return_tensors="pt").to(device)
        with torch.no_grad():
            probs = model(**enc).logits.softmax(-1)[0].tolist()
        top = max(range(len(probs)), key=probs.__getitem__)
        label = id2label[top]
        results.append(EvaluationResult(
            generation_cache_key=r.cache_key, evaluator_id=N_QY, evaluator_version="1.0.0",
            outcome=to_outcome[label], confidence=probs[top], native_label=label,
            notes=json.dumps({id2label[i]: round(p, 4) for i, p in enumerate(probs)})))
    return results


def judge_results(rows: pd.DataFrame) -> list[EvaluationResult]:
    by_variant: dict[tuple[str, str], GenerationRecord] = {}
    for rec in GenerationCache(JUDGE_CACHE).read_all():
        by_variant[(rec.request.template_id, rec.request.variant_id)] = rec
    out = []
    for evaluator_id in JUDGE_PROMPTS:
        for ck in rows.cache_key:
            rec = by_variant[(evaluator_id, f"judged::{ck}")]
            outcome, native = parse_judge(rec.raw_text) if rec.error is None else ("generation_error", None)
            out.append(EvaluationResult(
                generation_cache_key=ck, evaluator_id=evaluator_id, evaluator_version="1.0.0",
                outcome=outcome, native_label=native,
                notes=json.dumps({"judge_cache_key": rec.cache_key, "raw": rec.raw_text})))
    return out


def cmd_score(device: str) -> None:
    rows = frozen_rows()
    store = EvaluationStore(EVAL_STORE)
    new = [r for r in judge_results(rows) + nli_pair(rows, device)
           if not store.has(r.generation_cache_key, r.evaluator_id, r.evaluator_version)]
    print(f"appended {store.append_many(new)} evaluation results to {EVAL_STORE}")


# ---------------------------------------------------------------- analyze


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p from discordant counts b, c."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def cmd_analyze() -> None:
    from analyze_human_validation import HUMAN_TO_OUTCOME, collapse, evaluator_metrics

    key = pd.read_csv(KEY)
    consensus = pd.read_csv(CONSENSUS)
    df = key.merge(consensus[["sample_id", "relation", "mode"]], on="sample_id", validate="one_to_one")
    df["human_outcome"] = df.relation.map(HUMAN_TO_OUTCOME)
    stored = pd.DataFrame([r.model_dump() for r in EvaluationStore(EVAL_STORE).read_all()])
    wide = stored.pivot_table(index="generation_cache_key", columns="evaluator_id", values="outcome",
                              aggfunc="first")
    df = df.join(wide, on="cache_key")
    evaluators = [DET, N_Y, N_QY, J_Y, J_QY]
    assert df[evaluators].notna().all().all(), "missing evaluator outcomes"

    def with_fixed_reversal(g: pd.DataFrame, ev: str) -> dict:
        m = evaluator_metrics(g, ev)
        hd = g[g.human_outcome.isin({"agreement", "disagreement"})]
        rev = hd[ev].isin({"agreement", "disagreement"}) & (hd[ev] != hd.human_outcome)
        m.update(reversal_of_human_dir_k=int(rev.sum()), reversal_of_human_dir_n=len(hd),
                 reversal_of_human_dir_rate=float(rev.mean()) if len(hd) else float("nan"),
                 malformed=int((g[ev] == "malformed").sum()))
        return m

    summary = pd.DataFrame([with_fixed_reversal(df, ev) for ev in evaluators])
    strata = []
    for ev in evaluators:
        for dim in ("mode", "render", "prompt", "truncated", "stratum"):
            for val, g in df.groupby(dim):
                r = with_fixed_reversal(g, ev)
                r.update(slice_dimension=dim, slice_value=str(val))
                strata.append(r)
    strata = pd.DataFrame(strata)

    x = df[df.human_outcome.notna()]
    nondir = x[x.human_outcome.isin({"mixed_or_conditional", "neutral_or_no_position"})]
    hdir = x[x.human_outcome.isin({"agreement", "disagreement"})]
    paired = []
    for pair, (ey, eqy) in PAIRS.items():
        tests = {
            "H1_overread_on_human_nondir": (nondir, lambda g, e: g[e].isin({"agreement", "disagreement"})),
            "H2_reversal_on_human_dir": (hdir, lambda g, e: g[e].isin({"agreement", "disagreement"})
                                         & (g[e] != g.human_outcome)),
            "H3_error_collapsed_4class": (x, lambda g, e: collapse(g[e]) != g.human_outcome),
        }
        for name, (g, err) in tests.items():
            a, b = err(g, ey), err(g, eqy)
            only_y, only_qy = int((a & ~b).sum()), int((~a & b).sum())
            paired.append({"pair": pair, "test": name, "n": len(g), "E_y": ey, "E_qy": eqy,
                           "rate_E_y": float(a.mean()), "rate_E_qy": float(b.mean()),
                           "diff_qy_minus_y": float(b.mean() - a.mean()),
                           "fixed_by_q": only_y, "broken_by_q": only_qy,
                           "mcnemar_exact_p": mcnemar_exact(only_y, only_qy)})
    paired = pd.DataFrame(paired)

    Path(OUT).parent.mkdir(parents=True, exist_ok=True)
    summary.round(4).to_csv(f"{OUT}_evaluators.csv", index=False)
    strata.round(4).to_csv(f"{OUT}_strata.csv", index=False)
    paired.round(4).to_csv(f"{OUT}_paired.csv", index=False)
    cols = ["evaluator_id", "n", "accuracy", "macro_f1", "cohen_kappa", "underread_rate", "overread_rate",
            "overread_ci_low", "overread_ci_high", "reversal_of_human_dir_rate", "malformed"]
    Path(f"{OUT}_evaluators.md").write_text(
        "<!-- generated by scripts/eqy_experiment.py analyze; do not edit -->\n\n"
        + summary[cols].round(3).to_markdown(index=False) + "\n\n"
        + paired.round(4).to_markdown(index=False) + "\n", encoding="utf-8")
    figure(strata, evaluators)
    print(summary[cols].round(3).to_string(index=False))
    print()
    print(paired.round(4).to_string(index=False))


def figure(strata: pd.DataFrame, evaluators: list[str]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    modes = ["explicit_stance", "analytical_exposition", "voiced_continuation", "incomplete"]
    s = strata[(strata.slice_dimension == "mode") & strata.slice_value.isin(modes)]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)
    width = 0.8 / len(evaluators)
    for ax, metric, ncol, title in (
            (axes[0], "overread_rate", "overread_n", "Over-read (human mixed/neutral → directional)"),
            (axes[1], "reversal_of_human_dir_rate", "reversal_of_human_dir_n",
             "Reversal (human directional → opposite direction)")):
        for i, ev in enumerate(evaluators):
            r = s[s.evaluator_id == ev].set_index("slice_value").reindex(modes)
            xs = [j + (i - (len(evaluators) - 1) / 2) * width for j in range(len(modes))]
            ax.bar(xs, r[metric], width, label=ev, hatch="//" if ev in (N_QY, J_QY) else None)
        ns = s[s.evaluator_id == evaluators[0]].set_index("slice_value").reindex(modes)[ncol]
        ax.set_xticks(range(len(modes)))
        ax.set_xticklabels([f"{m}\n(n={0 if pd.isna(n) else int(n)})" for m, n in zip(modes, ns, strict=True)],
                           fontsize=7)
        ax.set_title(title, fontsize=9)
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("rate vs human consensus")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=7, loc="lower center", ncol=5, frameon=False)
    fig.suptitle("E(y) vs E(q,y) by human response mode (hatched = sees the statement; validity sample)",
                 fontsize=9)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    Path(FIGURE).parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"{FIGURE}.{ext}", dpi=200)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    j = sub.add_parser("judge")
    j.add_argument("--parse-check", action="store_true")
    j.add_argument("--gpu-memory-utilization", type=float, default=0.5)
    s = sub.add_parser("score")
    s.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    sub.add_parser("analyze")
    args = ap.parse_args()
    if args.cmd == "judge":
        cmd_judge(args.parse_check, args.gpu_memory_utilization)
    elif args.cmd == "score":
        cmd_score(args.device)
    else:
        cmd_analyze()


if __name__ == "__main__":
    main()
