"""Aggregate the per-seed classifier validation into per-variant rows
(mean / min / max macro-F1 across training seeds, at conf >= 0.0 and 0.9),
next to Faulborn et al.'s reported values. Reads only
artifacts/tables/faulborn_classifier_validation.parquet.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from anatomiae.analysis.frame import export_table

SRC = Path("artifacts/tables/faulborn_classifier_validation.parquet")


def main() -> None:
    df = pd.read_parquet(SRC)
    df = df[df["conf_threshold"].isin([0.0, 0.9])].copy()
    df["variant"] = df["classifier"].map(lambda c: re.sub(r"_seed\d+$", "", c.replace("reconstructed_", "")))
    agg = (df.groupby(["variant", "conf_threshold"])
             .agg(n_seeds=("classifier", "nunique"), macro_f1_mean=("macro_f1", "mean"),
                  macro_f1_min=("macro_f1", "min"), macro_f1_max=("macro_f1", "max"),
                  n_retained_mean=("n_retained", "mean"),
                  faulborn_reported_f1=("faulborn_reported_f1", "first"),
                  faulborn_reported_n=("faulborn_reported_n", "first"))
             .reset_index())
    agg["mean_delta_vs_faulborn"] = agg["macro_f1_mean"] - agg["faulborn_reported_f1"]
    agg = agg.round(4)
    export_table(agg, Path("artifacts/tables/faulborn_classifier_validation_by_variant"),
                 formats=("csv", "parquet", "md", "tex"))
    print(agg.to_string(index=False))


if __name__ == "__main__":
    main()
