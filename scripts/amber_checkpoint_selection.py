"""Choose Amber's early/mid pilot checkpoints from its published per-checkpoint
evaluations, not by picking an index arbitrarily.

Each IFM/Amber ckpt_NNN branch carries its own eval_{arc,hellaswag,mmlu,
truthfulqa}.json. This fetches them for a subset of checkpoints (every one
of ckpt_000-040, then every 10th, plus ckpt_358/359), writes the curve
table, and applies a selection rule fixed *before* looking at the data:

  EARLY = the earliest checkpoint whose HellaSwag acc_norm has covered at
  least 50% of the gap between ckpt_000 and the last checkpoint, and which
  is not a local dip (its HellaSwag is not more than 2 points below the
  previous sampled checkpoint's). Rationale: past the initial warm-up,
  measurably a language model, not a transient trough.
  MID   = the sampled checkpoint closest to the middle of the run (ckpt_180).
  FINAL = main (the released final model).

HellaSwag is the criterion because MMLU for a 7B model of this generation
sits near chance throughout and cannot distinguish stages; ARC and
TruthfulQA are reported alongside, not used for the choice.

Outputs: artifacts/tables/amber_eval_curves.{csv,parquet,md},
artifacts/tables/amber_checkpoint_selection.json
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
from huggingface_hub import hf_hub_download

from anatomiae.analysis.frame import export_table

REPO = "IFM/Amber"
TASKS = {"arc": ("arc_challenge", "acc_norm"), "hellaswag": ("hellaswag", "acc_norm"),
         "truthfulqa": ("truthfulqa_mc", "mc2"), "mmlu": (None, "acc")}
BRANCHES = [f"ckpt_{i:03d}" for i in list(range(41)) + list(range(50, 360, 10)) + [358, 359]]


def metric(task: str, payload: dict) -> float | None:
    key, field = TASKS[task]
    results = payload.get("results", {})
    if task == "mmlu":  # mean over all hendrycksTest-* subjects
        vals = [v["acc"] for k, v in results.items() if k.startswith("hendrycksTest-")]
        return sum(vals) / len(vals) if vals else None
    for k, v in results.items():
        if k == key or k.startswith(key):
            return v.get(field)
    return None


def fetch() -> pd.DataFrame:
    rows = []
    for b in sorted(set(BRANCHES)):
        row = {"checkpoint": b, "index": int(b.split("_")[1])}
        for task in TASKS:
            try:
                p = hf_hub_download(REPO, f"eval_{task}.json", revision=b)
                row[task] = metric(task, json.loads(Path(p).read_text()))
            except Exception as e:  # noqa: BLE001 - a missing file is recorded, not fatal
                row[task] = None
                row[f"{task}_error"] = type(e).__name__
            time.sleep(0.2)  # be gentle with the Hub
        rows.append(row)
        print(b, {t: row[t] for t in TASKS})
    return pd.DataFrame(rows).sort_values("index").reset_index(drop=True)


def select(df: pd.DataFrame) -> dict:
    hs = df.dropna(subset=["hellaswag"]).reset_index(drop=True)
    start, end = hs["hellaswag"].iloc[0], hs["hellaswag"].iloc[-1]
    threshold = start + 0.5 * (end - start)
    early = None
    for i in range(1, len(hs)):
        cur, prev = hs.loc[i, "hellaswag"], hs.loc[i - 1, "hellaswag"]
        if cur >= threshold and cur >= prev - 0.02:
            early = hs.loc[i, "checkpoint"]
            break
    return {
        "rule": "earliest ckpt with HellaSwag acc_norm >= ckpt_000 + 50% of (last - ckpt_000), "
                "not a >2pt dip vs previous sampled ckpt; MID = ckpt_180; FINAL = main",
        "hellaswag_first": start, "hellaswag_last": end, "threshold": threshold,
        "early": early, "mid": "ckpt_180", "final": "main",
    }


def main() -> None:
    df = fetch()
    export_table(df, Path("artifacts/tables/amber_eval_curves"), formats=("csv", "parquet", "md"))
    sel = select(df)
    Path("artifacts/tables/amber_checkpoint_selection.json").write_text(json.dumps(sel, indent=2))
    print(json.dumps(sel, indent=2))


if __name__ == "__main__":
    main()
