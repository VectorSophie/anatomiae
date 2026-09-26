"""Validate automatic political-response evaluators against human annotation.

The v1 150-response set is a stratified validity sample that intentionally
oversamples ambiguous / evaluator-disagreement cases. It is suitable for
evaluator validation, but NOT for an unweighted population estimate of OLMo
stage direction.

Expected local-only input:
  artifacts/labeling/key_v1.csv

Outputs:
  artifacts/tables/human_validation_v1_evaluators.csv
  artifacts/tables/human_validation_v1_strata.csv
  artifacts/tables/human_validation_v1_human_agreement.csv  (if C exists)
  docs/results/human_validation_v1.md
"""
from __future__ import annotations

import argparse
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
    x = pd.DataFrame({"y": y, "p": p}).dropna()
    cats = sorted(set(x.y) | set(x.p))
    fs = []
    for c in cats:
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


def evaluator_metrics(df: pd.DataFrame, ev: str) -> dict:
    x = df[df.human_outcome.notna()].copy()
    pred = x[ev]
    human = x.human_outcome
    human_dir = human.isin(DIRECTIONAL)
    human_nondir = human.isin(NONDIR)
    pred_dir = pred.isin(DIRECTIONAL)
    return {
        "evaluator_id": ev,
        "n": len(x),
        "accuracy": exact(human, pred),
        "macro_f1": macro_f1(human, pred),
        "cohen_kappa": kappa(human, pred),
        "directional_n": int(human_dir.sum()),
        "directional_exact_accuracy": exact(human[human_dir], pred[human_dir]),
        "directional_underread_rate": float((~pred_dir[human_dir]).mean()) if human_dir.any() else float("nan"),
        "nondirectional_n": int(human_nondir.sum()),
        "directional_overread_rate": float(pred_dir[human_nondir].mean()) if human_nondir.any() else float("nan"),
    }


def human_agreement(a: pd.DataFrame, c: pd.DataFrame) -> pd.DataFrame:
    m = a.merge(c, on="sample_id", suffixes=("_A", "_C"), validate="one_to_one")
    return pd.DataFrame([
        {"dimension": "relation", "n": len(m), "exact_agreement": exact(m.relation_A, m.relation_C),
         "cohen_kappa": kappa(m.relation_A, m.relation_C)},
        {"dimension": "mode", "n": len(m), "exact_agreement": exact(m.mode_A, m.mode_C),
         "cohen_kappa": kappa(m.mode_A, m.mode_C)},
    ])


def md_table(df: pd.DataFrame) -> str:
    try:
        return df.to_markdown(index=False)
    except Exception:
        return df.to_csv(index=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", default="artifacts/labeling/key_v1.csv")
    ap.add_argument("--labels-a", default="artifacts/labeling/labels_v1_annotator_A_human.csv")
    ap.add_argument("--labels-c", default="artifacts/labeling/labels_v1_annotator_C_human.csv")
    ap.add_argument("--out-prefix", default="artifacts/tables/human_validation_v1")
    args = ap.parse_args()

    key_path = Path(args.key)
    if not key_path.exists():
        raise SystemExit(
            f"Missing {key_path}. This file was intentionally kept local-only during blind annotation. "
            "Run on the research workstation where key_v1.csv was generated, or restore the frozen key there."
        )

    key = pd.read_csv(key_path)
    a = pd.read_csv(args.labels_a)
    assert key.sample_id.is_unique and a.sample_id.is_unique
    assert set(a.sample_id) == set(key.sample_id), "A labels and frozen key differ"
    df = key.merge(
        a[["sample_id", "relation", "mode", "confidence_1to3", "notes"]],
        on="sample_id", validate="one_to_one"
    )
    df["human_outcome"] = df.relation.map(HUMAN_TO_OUTCOME)

    evaluators = [c for c in key.columns if c.startswith(("faulborn_nli_", "deterministic"))]
    summary = pd.DataFrame([evaluator_metrics(df, ev) for ev in evaluators]).round(4)

    strata_rows = []
    for ev in evaluators:
        for dim in ("mode", "stage", "render", "prompt", "truncated", "stratum"):
            if dim not in df.columns:
                continue
            for val, g in df.groupby(dim, dropna=False):
                if len(g) < 3:
                    continue
                r = evaluator_metrics(g, ev)
                r.update({"slice_dimension": dim, "slice_value": val})
                strata_rows.append(r)
    strata = pd.DataFrame(strata_rows).round(4)

    prefix = Path(args.out_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(str(prefix) + "_evaluators.csv", index=False)
    strata.to_csv(str(prefix) + "_strata.csv", index=False)

    agreement = None
    c_path = Path(args.labels_c)
    if c_path.exists():
        c = pd.read_csv(c_path)
        agreement = human_agreement(a, c).round(4)
        agreement.to_csv(str(prefix) + "_human_agreement.csv", index=False)

    report = [
        "# Human validation v1\n",
        "**Primary reference:** Annotator A, the human researcher.\n",
        "The frozen 150-response set was deliberately stratified to oversample ambiguous cases. "
        "Therefore the tables below validate evaluators on this sample but MUST NOT be treated as an unweighted population estimate of OLMo political direction.\n",
        "## Automatic evaluators vs human A\n",
        md_table(summary),
        "\n\nThe key diagnostic is `directional_overread_rate`: among responses the human marked mixed/conditional or neutral/no-position, how often the evaluator still forced agreement/disagreement.\n",
    ]
    if agreement is not None:
        report += ["\n## Human A vs Human C\n", md_table(agreement), "\n"]
    report += [
        "\n## Interpretation guardrail\n",
        "This validation set is quota-sampled from disagreement/ambiguity strata. For a human-grounded stage-level direction estimate, use either a separately drawn representative human sample or an explicitly justified sampling-weight estimator. Do not simply compute a stage direction index from these 150 rows.\n",
    ]
    out_doc = Path("docs/results/human_validation_v1.md")
    out_doc.parent.mkdir(parents=True, exist_ok=True)
    out_doc.write_text("\n".join(report), encoding="utf-8")
    print(summary.to_string(index=False))
    if agreement is not None:
        print("\nHuman-human agreement:\n", agreement.to_string(index=False))


if __name__ == "__main__":
    main()
