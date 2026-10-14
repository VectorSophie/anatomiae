"""Blind human-audit sample of commitment C on Qwen2.5-14B-Instruct outputs
(protocol: docs/experiments/commitment_audit_qwen_v1.md).

20 Qwen2.5-14B-Instruct free-form responses drawn uniformly at random per free-form
prompt (100 total). Reads the xl caches directly for the q_inst arm only, so it is
independent of whether other arms in the same cache are scored yet. Writes, never
overwrites:
  artifacts/labeling/commitment_audit_qwen_v1/sheet_A.csv, sheet_C.csv  blind
  artifacts/labeling/commitment_audit_qwen_v1/key.csv                   local-only (gitignored)
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import selective_commitment_v1 as sc
import selective_commitment_xl_v1 as xl
from eqy_experiment import J_QY

from anatomiae.datasets.faulborn import load_faulborn_items
from anatomiae.provenance.evaluation_store import EvaluationStore
from anatomiae.provenance.generation_cache import GenerationCache

OUT = Path("artifacts/labeling/commitment_audit_qwen_v1")
SEED = 7101
PER_PROMPT = 20
MODEL = xl.ARMS["q_inst"][1]


def main() -> None:
    key_path = OUT / "key.csv"
    if key_path.exists():
        raise SystemExit(f"{key_path} exists; a frozen audit sample is never regenerated in place")
    items = {i.item_id: i for i in load_faulborn_items()}
    flip = {"left": "right", "right": "left"}
    recs = [r for r in GenerationCache(xl.CACHE).read_all() if r.request.model_id == MODEL]
    jqy = {e.generation_cache_key: e.outcome for e in EvaluationStore(xl.EVAL_STORE).read_all()
           if e.evaluator_id == J_QY}
    rows = []
    for r in recs:
        prompt, polarity = r.request.template_id.split("__")
        lean = items[r.request.item_id].reference_position
        rows.append({"cache_key": r.cache_key, "prompt": prompt, "item_id": r.request.item_id,
                     "polarity": polarity, "item_lean": lean if polarity == "orig" else flip[lean],
                     "jqy_outcome": jqy[r.cache_key], "raw_text": r.raw_text,
                     "statement": sc.statement(items[r.request.item_id], polarity)})
    df = pd.DataFrame(rows)
    assert len(df) == 1056, len(df)
    sf = df[df["prompt"] == "stance_first"].set_index(["item_id", "polarity"])["jqy_outcome"].rename("sf_outcome")
    ff = df[df["prompt"].isin(sc.FREE_FORM)].join(sf, on=["item_id", "polarity"])
    ff["C_jqy"] = ff["jqy_outcome"].isin(sc.DIRECTIONAL)
    ff["sf_pole"] = ["non_directional" if o not in sc.DIRECTIONAL else
                     "left_pole" if (o == "agreement") == (le == "left") else "right_pole"
                     for o, le in zip(ff["sf_outcome"], ff["item_lean"], strict=True)]
    parts = [ff[ff["prompt"] == p].sort_values("cache_key").sample(PER_PROMPT, random_state=SEED)
             for p in sc.FREE_FORM]
    sample = pd.concat(parts).sample(frac=1, random_state=SEED + 1)
    sample["sample_id"] = [f"CQ{n:03d}" for n in range(1, len(sample) + 1)]
    blind = pd.DataFrame({"sample_id": sample["sample_id"], "statement": sample["statement"],
                          "response": sample["raw_text"], "relation": "", "mode": "",
                          "confidence_1to3": "", "notes": ""})
    OUT.mkdir(parents=True, exist_ok=True)
    for annotator, seed in (("A", 11), ("C", 33)):
        blind.sample(frac=1, random_state=seed).to_csv(OUT / f"sheet_{annotator}.csv", index=False)
    sample[["sample_id", "cache_key", "prompt", "item_id", "polarity", "item_lean", "jqy_outcome", "C_jqy",
            "sf_outcome", "sf_pole"]].to_csv(key_path, index=False)
    print(sample.groupby("prompt").size().to_string())
    print("C_jqy rate in sample:", round(sample["C_jqy"].mean(), 3), "population:", round(ff["C_jqy"].mean(), 3))


if __name__ == "__main__":
    main()
