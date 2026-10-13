"""Blind human-audit sample of the commitment indicator C (protocol: docs/experiments/commitment_audit_v1.md).

10 free-form responses drawn uniformly at random from each stage x free-form
prompt cell of the OLMo selective-commitment confirmatory run (150 total).
Writes, never overwrites:
  artifacts/labeling/commitment_audit_v1/sheet_A.csv, sheet_C.csv  blind; independent row orders
  artifacts/labeling/commitment_audit_v1/key.csv                   provenance + J-qy C + SF pole (gitignored)
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import selective_commitment_v1 as sc

OUT = Path("artifacts/labeling/commitment_audit_v1")
SEED = 20261014
PER_CELL = 10


def main() -> None:
    key_path = OUT / "key.csv"
    if key_path.exists():
        raise SystemExit(f"{key_path} exists; a frozen audit sample is never regenerated in place")
    df = sc.frame()
    text = {r.cache_key: r.raw_text for r in sc.GenerationCache(sc.CACHE).read_all()}
    statements = sc.records_with_statements()[1]
    parts = []
    for stage in ("1_sft", "2_dpo", "3_rlvr2"):
        u = sc.units(df, stage)
        u["sf_pole"] = ["non_directional" if o not in sc.DIRECTIONAL else
                        "left_pole" if (o == "agreement") == (le == "left") else "right_pole"
                        for o, le in zip(u["sf_outcome"], u["item_lean"], strict=True)]
        for prompt in sc.FREE_FORM:
            cell = u[u["prompt"] == prompt].sort_values("cache_key")
            parts.append(cell.sample(PER_CELL, random_state=SEED))
    sample = pd.concat(parts).sample(frac=1, random_state=SEED + 1)  # ids must not encode cell order
    sample["sample_id"] = [f"CA{n:03d}" for n in range(1, len(sample) + 1)]
    blind = pd.DataFrame({"sample_id": sample["sample_id"],
                          "statement": [statements[k] for k in sample["cache_key"]],
                          "response": [text[k] for k in sample["cache_key"]],
                          "relation": "", "mode": "", "confidence_1to3": "", "notes": ""})
    OUT.mkdir(parents=True, exist_ok=True)
    for annotator, seed in (("A", 11), ("C", 33)):
        blind.sample(frac=1, random_state=seed).to_csv(OUT / f"sheet_{annotator}.csv", index=False)
    key = sample[["sample_id", "cache_key", "stage", "prompt", "item_id", "polarity", "item_lean", "outcome",
                  "C", "sf_outcome", "sf_pole"]].rename(columns={"outcome": "jqy_outcome", "C": "C_jqy"})
    key.to_csv(key_path, index=False)
    print(key.groupby(["stage", "prompt"]).size().to_string())
    print("C_jqy rate in sample:", round(key["C_jqy"].mean(), 3))


if __name__ == "__main__":
    main()
