"""Validate automatic political-response evaluators against human annotation.

The v1 150-response set is a stratified validity sample that intentionally
oversamples ambiguous / evaluator-disagreement cases. It is suitable for
evaluator validation, but NOT for an unweighted population estimate of OLMo
stage direction.

Inputs:
  artifacts/labeling/key_v1.csv  frozen provenance + evaluator labels
                                 (verified by scripts/verify_label_key.py)
  artifacts/labeling/labels_v1_human_consensus.csv  primary reference
  artifacts/labeling/labels_v1_annotator_{A,C}_human.csv

Outputs (all generated; docs/results/human_validation_v1.md is written by
hand from these and is not touched here):
  artifacts/tables/human_validation_v1_evaluators.{csv,md}
  artifacts/tables/human_validation_v1_strata.csv
  artifacts/tables/human_validation_v1_confusion.csv
  artifacts/tables/human_validation_v1_human_agreement.csv
  figures/human_validation_v1_overread_by_mode.{png,pdf}

Label spaces. Humans label `relation` in {supports, opposes,
mixed_or_conditional, neutral_or_no_position, unclear}; `unclear` (7 rows)
has no evaluator counterpart and is excluded from every evaluator metric, so
evaluator metrics use n = 143 unless a slice says otherwise. Evaluators can
also emit outcomes humans could not choose (`irrelevant`, `safety_refusal`,
`refusal`, `epistemic_uncertainty`, `malformed`). Two multiclass views are
reported:
  - raw: the evaluator's own outcome vs the human outcome (those extra
    outcomes always count as errors);
  - collapsed: every non-position evaluator outcome is mapped to
    `neutral_or_no_position` (none of them asserts a stance), giving the
    same 4-class space as the humans.
Directional metrics (over-read, under-read, reversal) do not depend on this
choice: they only ask whether the evaluator assigned agreement/disagreement.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd

HUMAN_TO_OUTCOME = {
    "supports": "agreement",
    "opposes": "disagreement",
    "mixed_or_conditional": "mixed_or_conditional",
    "neutral_or_no_position": "neutral_or_no_position",
    "unclear": None,
}
DIRECTIONAL = {"agreement", "disagreement"}
NONDIR = {"mixed_or_conditional", "neutral_or_no_position"}
COLLAPSE = {"agreement": "agreement", "disagreement": "disagreement",
            "mixed_or_conditional": "mixed_or_conditional"}  # everything else -> neutral_or_no_position
SLICES = ("mode", "stage", "render", "prompt", "truncated", "stratum")


def collapse(s: pd.Series) -> pd.Series:
    return s.map(lambda v: COLLAPSE.get(v, "neutral_or_no_position") if isinstance(v, str) else v)


def kappa(a: pd.Series, b: pd.Series) -> float:
    x = pd.DataFrame({"a": a, "b": b}).dropna()
    if not len(x):
        return float("nan")
    po = (x.a == x.b).mean()
    cats = sorted(set(x.a) | set(x.b))
    pa = x.a.value_counts(normalize=True)
    pb = x.b.value_counts(normalize=True)
    pe = sum(pa.get(c, 0.0) * pb.get(c, 0.0) for c in cats)
    return float((po - pe) / (1 - pe)) if pe < 1 else float("nan")


def macro_f1(y: pd.Series, p: pd.Series) -> float:
    """Macro-F1 over the reference (human) classes present in `y`; predicted
    classes humans cannot emit lower recall/precision but are not classes."""
    x = pd.DataFrame({"y": y, "p": p}).dropna()
    fs = []
    for c in sorted(set(x.y)):
        tp = ((x.y == c) & (x.p == c)).sum()
        fp = ((x.y != c) & (x.p == c)).sum()
        fn = ((x.y == c) & (x.p != c)).sum()
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        fs.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return float(sum(fs) / len(fs)) if fs else float("nan")


def exact(a: pd.Series, b: pd.Series) -> float:
    x = pd.DataFrame({"a": a, "b": b}).dropna()
    return float((x.a == x.b).mean()) if len(x) else float("nan")


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def rate(mask: pd.Series, prefix: str) -> dict:
    k, n = int(mask.sum()), len(mask)
    lo, hi = wilson(k, n)
    return {f"{prefix}_k": k, f"{prefix}_n": n, f"{prefix}_rate": k / n if n else float("nan"),
            f"{prefix}_ci_low": lo, f"{prefix}_ci_high": hi}


def evaluator_metrics(df: pd.DataFrame, ev: str) -> dict:
    x = df[df.human_outcome.notna()]
    pred, human = x[ev], x.human_outcome
    human_dir, human_nondir = human.isin(DIRECTIONAL), human.isin(NONDIR)
    pred_dir = pred.isin(DIRECTIONAL)
    both_dir = human_dir & pred_dir
    return {
        "evaluator_id": ev,
        "n": len(x),
        "accuracy_raw": exact(human, pred),
        "accuracy": exact(human, collapse(pred)),
        "macro_f1": macro_f1(human, collapse(pred)),
        "cohen_kappa": kappa(human, collapse(pred)),
        "directional_exact_accuracy": exact(human[human_dir], pred[human_dir]),
        **rate(~pred_dir[human_dir], "underread"),
        **rate(pred_dir[human_nondir], "overread"),
        # direct supports<->opposes flips among responses both call directional
        **rate(pred[both_dir] != human[both_dir], "reversal"),
    }


def human_agreement(a: pd.DataFrame, c: pd.DataFrame) -> pd.DataFrame:
    m = a.merge(c, on="sample_id", suffixes=("_A", "_C"), validate="one_to_one")
    rows = [{"dimension": d, "n": len(m), "exact_agreement": exact(m[f"{d}_A"], m[f"{d}_C"]),
             "cohen_kappa": kappa(m[f"{d}_A"], m[f"{d}_C"])} for d in ("relation", "mode")]
    both = m[m.relation_A.isin({"supports", "opposes"}) & m.relation_C.isin({"supports", "opposes"})]
    rows.append({"dimension": "direction_when_both_directional", "n": len(both),
                 "exact_agreement": exact(both.relation_A, both.relation_C),
                 "cohen_kappa": kappa(both.relation_A, both.relation_C)})
    return pd.DataFrame(rows)


def md_table(df: pd.DataFrame) -> str:
    try:
        return df.to_markdown(index=False)
    except ImportError:
        return df.to_csv(index=False)


def overread_figure(strata: pd.DataFrame, evaluators: list[str], out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    modes = ["explicit_stance", "analytical_exposition", "incomplete", "voiced_continuation"]
    s = strata[(strata.slice_dimension == "mode") & strata.slice_value.isin(modes)]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)
    for ax, metric, title in ((axes[0], "overread", "Over-read: human non-directional → evaluator directional"),
                              (axes[1], "underread", "Under-read: human directional → evaluator non-directional")):
        width = 0.8 / len(evaluators)
        for i, ev in enumerate(evaluators):
            r = s[s.evaluator_id == ev].set_index("slice_value").reindex(modes)
            xs = [j + (i - (len(evaluators) - 1) / 2) * width for j in range(len(modes))]
            ax.bar(xs, r[f"{metric}_rate"], width, label=ev.replace("faulborn_nli_reconstructed_", "recon_"))
        ax.set_xticks(range(len(modes)))
        ax.set_xticklabels([f"{m}\n(n_nondir={int(s[(s.slice_value == m)][f'{metric}_n'].iloc[0])})"
                            if len(s[s.slice_value == m]) else m for m in modes], fontsize=7)
        ax.set_title(title, fontsize=9)
        ax.set_ylim(0, 1)
    axes[1].set_xticklabels([f"{m}\n(n_dir={int(s[(s.slice_value == m)]['underread_n'].iloc[0])})"
                             if len(s[s.slice_value == m]) else m for m in modes], fontsize=7)
    axes[0].set_ylabel("rate (consensus reference)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=7, loc="lower center", ncol=4, frameon=False)
    fig.suptitle("Evaluator error by human-consensus response mode (stratified validity sample, not population)",
                 fontsize=9)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    out.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out.with_suffix(f".{ext}"), dpi=200)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", default="artifacts/labeling/key_v1.csv")
    ap.add_argument("--labels-a", default="artifacts/labeling/labels_v1_annotator_A_human.csv")
    ap.add_argument("--labels-c", default="artifacts/labeling/labels_v1_annotator_C_human.csv")
    ap.add_argument("--labels-consensus", default="artifacts/labeling/labels_v1_human_consensus.csv")
    ap.add_argument("--out-prefix", default="artifacts/tables/human_validation_v1")
    ap.add_argument("--figure", default="figures/human_validation_v1_overread_by_mode")
    args = ap.parse_args()

    if not Path(args.key).exists():
        raise SystemExit(f"Missing {args.key}; see artifacts/labeling/README.md")
    key = pd.read_csv(args.key)
    a = pd.read_csv(args.labels_a)
    c = pd.read_csv(args.labels_c)
    consensus = pd.read_csv(args.labels_consensus)
    for d in (key, a, c, consensus):
        assert d.sample_id.is_unique and set(d.sample_id) == set(key.sample_id), "sample ids differ"
    df = key.merge(consensus[["sample_id", "relation", "mode"]], on="sample_id", validate="one_to_one")
    df["human_outcome"] = df.relation.map(HUMAN_TO_OUTCOME)
    assert df.human_outcome.notna().sum() == len(df) - (df.relation == "unclear").sum()

    evaluators = sorted(col for col in key.columns if col.startswith(("faulborn_nli_", "deterministic")))
    summary = pd.DataFrame([evaluator_metrics(df, ev) for ev in evaluators])

    strata_rows = []
    for ev in evaluators:
        for dim in SLICES:
            for val, g in df.groupby(dim, dropna=False):
                r = evaluator_metrics(g, ev)
                r.update({"slice_dimension": dim, "slice_value": str(val)})
                strata_rows.append(r)
    strata = pd.DataFrame(strata_rows)

    confusion = (df[df.human_outcome.notna()]
                 .melt(id_vars=["sample_id", "human_outcome"], value_vars=evaluators,
                       var_name="evaluator_id", value_name="evaluator_outcome")
                 .groupby(["evaluator_id", "human_outcome", "evaluator_outcome"]).size()
                 .rename("n").reset_index())

    prefix = Path(args.out_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    summary.round(4).to_csv(f"{prefix}_evaluators.csv", index=False)
    strata.round(4).to_csv(f"{prefix}_strata.csv", index=False)
    confusion.to_csv(f"{prefix}_confusion.csv", index=False)
    agreement = human_agreement(a, c).round(4)
    agreement.to_csv(f"{prefix}_human_agreement.csv", index=False)

    cols = ["evaluator_id", "n", "accuracy_raw", "accuracy", "macro_f1", "cohen_kappa",
            "directional_exact_accuracy", "underread_rate", "underread_n", "overread_rate",
            "overread_ci_low", "overread_ci_high", "overread_n", "reversal_rate", "reversal_n"]
    Path(f"{prefix}_evaluators.md").write_text(
        "<!-- generated by scripts/analyze_human_validation.py; do not edit -->\n\n"
        + md_table(summary[cols].round(3)) + "\n", encoding="utf-8")
    overread_figure(strata, evaluators, Path(args.figure))

    print(summary[cols].round(3).to_string(index=False))
    print("\nHuman-human agreement:\n", agreement.to_string(index=False))


if __name__ == "__main__":
    main()
