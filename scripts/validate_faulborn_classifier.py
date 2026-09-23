"""Validate Faulborn-procedure classifiers on their released test split.

For each available classifier variant, classify the 264 released test
responses with the exact released inference procedure
(anatomiae.evaluators.faulborn_nli) and compute macro-F1 at the same
confidence thresholds as their eval.py (subset with confidence >= c,
macro-F1 over that subset, N retained). Compare against their released
eval_df.csv, which reports both their fine-tuned classifier and the
zero-shot bart-large-mnli baseline.

The zero-shot row is an exact-reproduction check of their evaluation path
(public weights, same template/labels/procedure). Reconstructed rows are
checked against their fine-tuned numbers, with bootstrap 95% CIs
(percentile, 1,000 resamples).
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from anatomiae.analysis.frame import export_table
from anatomiae.evaluators.faulborn_nli import FaulbornNLIEvaluator
from anatomiae.metrics.agreement import macro_f1

FAULBORN_ROOT = Path("/data/jackb/anatomiae/external/faulborn")
RECON_ROOT = Path("/data/jackb/anatomiae/models/faulborn_classifier_reconstruction")
CONF_LEVELS = [round(x * 0.1, 1) for x in range(10)]
N_BOOT = 1000


def bootstrap_ci(y_true, y_pred, conf, level, n_boot=N_BOOT, seed=0):
    rng = random.Random(seed)
    idx = list(range(len(y_true)))
    stats = []
    for _ in range(n_boot):
        sample = [rng.choice(idx) for _ in idx]
        kept = [i for i in sample if conf[i] >= level]
        if not kept:
            continue
        stats.append(macro_f1([y_true[i] for i in kept], [y_pred[i] for i in kept]))
    stats.sort()
    return stats[int(0.025 * len(stats))], stats[int(0.975 * len(stats)) - 1]


def main() -> None:
    from datasets import load_from_disk

    test = load_from_disk(str(FAULBORN_ROOT / "stance_detector_eval" / "test" / "test_set")).to_pandas()
    theirs = pd.read_csv(FAULBORN_ROOT / "stance_detector_eval" / "eval_df.csv", index_col=0)
    y_true = list(test["label"])
    texts = list(test["text"])

    variants = {"zeroshot": ("facebook/bart-large-mnli", "their zeroshot_f1 / zeroshot_n")}
    for d in sorted(RECON_ROOT.glob("*_seed*")):
        if (d / "anatomiae_provenance.json").exists():
            variants[f"reconstructed_{d.name}"] = (str(d), "their f1 / n (fine-tuned)")

    rows = []
    predictions = {}
    for name, (model_path, compare_to) in variants.items():
        ev = FaulbornNLIEvaluator(model_path=model_path, evaluator_id=f"faulborn_nli_{name}")
        preds = ev.classify_texts(texts)
        y_pred = [p for p, _ in preds]
        conf = [c for _, c in preds]
        predictions[name] = preds
        for level in CONF_LEVELS:
            kept = [i for i in range(len(y_true)) if conf[i] >= level]
            f1 = macro_f1([y_true[i] for i in kept], [y_pred[i] for i in kept]) if kept else float("nan")
            lo, hi = bootstrap_ci(y_true, y_pred, conf, level) if level in (0.0, 0.9) else (None, None)
            their_row = theirs[theirs["conf"].round(1) == level].iloc[0]
            is_zero = name == "zeroshot"
            rows.append({
                "classifier": name,
                "conf_threshold": level,
                "n_retained": len(kept),
                "macro_f1": f1,
                "macro_f1_ci95_low": lo,
                "macro_f1_ci95_high": hi,
                "faulborn_reported_f1": their_row["zeroshot_f1" if is_zero else "f1"],
                "faulborn_reported_n": int(their_row["zeroshot_n" if is_zero else "n"]),
                "compared_against": compare_to,
            })
        del ev

    df = pd.DataFrame(rows)
    df["delta_vs_faulborn"] = df["macro_f1"] - df["faulborn_reported_f1"]
    written = export_table(df, Path("artifacts/tables/faulborn_classifier_validation"),
                           formats=("csv", "parquet", "md", "tex"))
    Path("artifacts/tables/faulborn_classifier_validation_predictions.json").write_text(
        json.dumps({k: [[p, round(c, 6)] for p, c in v] for k, v in predictions.items()}, indent=0)
    )
    print(df[df["conf_threshold"].isin([0.0, 0.5, 0.9])].to_string(index=False))
    print(f"\nWrote {sorted(str(p) for p in written.values())}")


if __name__ == "__main__":
    main()
