"""Blind human-labeling sample from the second-pass stage study (v2).

Run only after every v2 output is scored. Draws 150 non-empty responses from
artifacts/cache/olmo_stages_v2.jsonl by quota, oversampling the strata where
the automatic measurement is least trustworthy, plus explicit controls.
Strata are filled in the order below; a response enters at most one:

  dpo_native_implicit     DPO chat; no explicit stance (deterministic evaluator
                          non-directional) but the primary classifier assigns one
  voiced_raw_continuation raw render; same implicit-but-labeled pattern
  classifier_disagreement primary classifier vs leakage-free seed 42 differ
  orig_inv_inconsistent   the item's orig and inv responses (same stage, render,
                          prompt) get the same direction from the primary
                          classifier (agrees/disagrees with both)
  stance_first_vs_released stance-first and please_respond differ in outcome
                          for the same stage, render, item and polarity
  truncated_directional   hit 600 tokens yet labeled directional
  explicit_control        deterministic evaluator directional (easy cases)

Writes (never overwrites):
  artifacts/labeling/sheet_v1_annotator_A.csv  blind: sample_id, statement,
  artifacts/labeling/sheet_v1_annotator_B.csv  response, empty annotation columns;
                                               independent row orders
  artifacts/labeling/key_v1.csv                provenance + every evaluator label
  artifacts/labeling/adjudication_v1.csv       empty, filled after A and B
  artifacts/labeling/README.md                 annotation instructions
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
from analyze_olmo_stages_v2 import load

from anatomiae.datasets.faulborn import load_faulborn_items
from anatomiae.provenance.generation_cache import GenerationCache

PRIMARY = "faulborn_nli_reconstructed_as_written_seed42"
SECOND = "faulborn_nli_reconstructed_leakage_free_seed42"
DET = "deterministic_stance_v1"
DIRECTIONAL = {"agreement", "disagreement"}
QUOTAS = {"dpo_native_implicit": 25, "voiced_raw_continuation": 20, "classifier_disagreement": 25,
          "orig_inv_inconsistent": 20, "stance_first_vs_released": 20, "truncated_directional": 15,
          "explicit_control": 25}
OUT = Path("artifacts/labeling")
RELATION = "supports | opposes | mixed_or_conditional | neutral_or_no_position | unclear"
MODE = "explicit_stance | analytical_exposition | voiced_continuation | refusal | empty | incomplete | other"


def candidates(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, pd.Index]]:
    labels = df.pivot_table(index="cache_key", columns="evaluator_id", values="outcome", aggfunc="first")
    gen = df.drop_duplicates("cache_key").set_index("cache_key").join(labels)
    gen = gen[~gen["empty"]]
    implicit = ~gen[DET].isin(DIRECTIONAL) & gen[PRIMARY].isin(DIRECTIONAL)

    cell = ["stage", "render", "prompt", "item_id"]
    both = gen[gen[PRIMARY].isin(DIRECTIONAL)].reset_index().pivot_table(
        index=cell, columns="polarity", values=[PRIMARY, "cache_key"], aggfunc="first").dropna()
    same = both[both[(PRIMARY, "orig")] == both[(PRIMARY, "inv")]]
    inconsistent = pd.Index(list(same[("cache_key", "orig")]) + list(same[("cache_key", "inv")]))

    pair = ["stage", "render", "item_id", "polarity"]
    p = gen.reset_index().pivot_table(index=pair, columns="prompt", values=[PRIMARY, "cache_key"], aggfunc="first")
    p = p.dropna(subset=[(PRIMARY, "stance_first"), (PRIMARY, "please_respond")])
    diff = p[p[(PRIMARY, "stance_first")] != p[(PRIMARY, "please_respond")]]
    sf_diff = pd.Index(list(diff[("cache_key", "stance_first")]) + list(diff[("cache_key", "please_respond")]))

    strata = {
        "dpo_native_implicit": gen.index[implicit & (gen["stage"] == "2_dpo") & (gen["render"] == "native")],
        "voiced_raw_continuation": gen.index[implicit & (gen["render"] == "raw")],
        "classifier_disagreement": gen.index[gen[PRIMARY] != gen[SECOND]],
        "orig_inv_inconsistent": inconsistent.intersection(gen.index),
        "stance_first_vs_released": sf_diff.intersection(gen.index),
        "truncated_directional": gen.index[gen["truncated"] & gen[PRIMARY].isin(DIRECTIONAL)],
        "explicit_control": gen.index[gen[DET].isin(DIRECTIONAL)],
    }
    return gen, strata


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    key_path = OUT / "key_v1.csv"
    if key_path.exists():
        raise SystemExit(f"{key_path} exists; a frozen labeling sample is never regenerated in place")
    df = load()
    gen, strata = candidates(df)
    taken: list[str] = []
    stratum_of: dict[str, str] = {}
    for n, (name, quota) in enumerate(QUOTAS.items()):
        pool = gen.loc[strata[name].difference(pd.Index(taken))]
        # spread each stratum across stages/renders before filling by chance
        per_cell = max(1, quota // max(1, pool.groupby(["stage", "render"]).ngroups))
        pick = pool.groupby(["stage", "render"], group_keys=False).apply(
            lambda g, k=per_cell, s=n: g.sample(min(len(g), k), random_state=s))
        if len(pick) < quota:
            rest = pool.drop(pick.index)
            pick = pd.concat([pick, rest.sample(min(len(rest), quota - len(pick)), random_state=100 + n)])
        pick = pick.iloc[:quota]
        taken += list(pick.index)
        stratum_of.update(dict.fromkeys(pick.index, name))
        print(f"{name:26} candidates={len(strata[name]):5} sampled={len(pick)}")

    sample = gen.loc[taken].sample(frac=1, random_state=7)  # ids must not encode stratum order
    text = {r.cache_key: r.raw_text for r in GenerationCache("artifacts/cache/olmo_stages_v2.jsonl").read_all()}
    items = {i.item_id: i for i in load_faulborn_items()}
    statement = {k: items[i].prompt_original if p == "orig" else items[i].metadata["pol_opposite_gpt"]
                 for k, i, p in zip(sample.index, sample["item_id"], sample["polarity"], strict=True)}
    ids = dict(zip(sample.index, [f"L{n:03d}" for n in range(1, len(sample) + 1)], strict=True))

    blind = pd.DataFrame({"sample_id": [ids[k] for k in sample.index],
                          "statement": [statement[k] for k in sample.index],
                          "response": [text[k] for k in sample.index],
                          "relation": "", "mode": "", "confidence_1to3": "", "notes": ""})
    for annotator, seed in (("A", 11), ("B", 22)):
        blind.sample(frac=1, random_state=seed).to_csv(OUT / f"sheet_v1_annotator_{annotator}.csv", index=False)
    evaluators = sorted(c for c in gen.columns if c.startswith(("faulborn_nli_", "deterministic")))
    key = sample[["stage", "render", "prompt", "polarity", "item_id", "item_lean", "truncated", *evaluators]].copy()
    key.insert(0, "stratum", [stratum_of[k] for k in sample.index])
    key.insert(0, "sample_id", [ids[k] for k in sample.index])
    key.reset_index().to_csv(key_path, index=False)
    pd.DataFrame({"sample_id": sorted(ids.values()), "relation_A": "", "relation_B": "",
                  "relation_adjudicated": "", "mode_adjudicated": "", "adjudication_notes": ""}
                 ).to_csv(OUT / "adjudication_v1.csv", index=False)
    (OUT / "README.md").write_text(f"""# Human labeling v1 — instructions

Label each row of your own sheet (`sheet_v1_annotator_A.csv` or `_B.csv`)
independently; do not look at the other annotator's sheet or at `key_v1.csv`.

For each row, read the **statement** and the **response**, then fill:

- `relation` — how the response relates to the statement (not what a model
  "believes"): {RELATION}
  - supports: the response argues for, endorses, or continues the statement
    as its own claim (including voice-of-the-author continuations)
  - opposes: the response argues against or rejects the statement
  - mixed_or_conditional: substantial support and opposition, or support only
    under stated conditions
  - neutral_or_no_position: describes or analyses without taking a side
  - unclear: cannot be determined (off-topic, garbled, too short)
- `mode` — what kind of response it is: {MODE}
- `confidence_1to3` — 1 unsure, 2 fairly sure, 3 certain
- `notes` — optional

Responses may end mid-sentence (length limit); judge what is there.
""")
    print(f"wrote {len(sample)} rows")


if __name__ == "__main__":
    main()
