"""Verify the frozen v1 labeling key against the immutable v2 artifacts.

`artifacts/labeling/key_v1.csv` was written once by `make_labeling_sheet.py`
and kept out of the public repo while annotation was blind. Before it is
versioned and used as the provenance for human validation, this checks that
every key row is exactly recoverable from the saved artifacts:

  - 150 unique sample ids, identical to both blind sheets;
  - each key `cache_key` exists exactly once in the v2 generation cache;
  - the sheet `response` equals the cached `raw_text` byte for byte;
  - the sheet `statement` equals the Faulborn item (orig) or its generated
    opposite (inv) for the key's `item_id` / `polarity`;
  - stage / render / prompt / polarity / item_id / truncated agree with the
    cached request;
  - every evaluator column equals the stored evaluation outcome.

Read-only: exits non-zero on the first class of mismatch and writes nothing.
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

LAB = Path("artifacts/labeling")
CACHE = "artifacts/cache/olmo_stages_v2.jsonl"
META = ["stage", "render", "prompt", "polarity", "item_id", "item_lean", "truncated"]


def main() -> int:
    key = pd.read_csv(LAB / "key_v1.csv", keep_default_na=False)
    sheets = [pd.read_csv(LAB / f"sheet_v1_annotator_{a}.csv", keep_default_na=False) for a in "AB"]
    problems: list[str] = []

    if len(key) != 150 or not key.sample_id.is_unique or not key.cache_key.is_unique:
        problems.append("key is not 150 unique sample ids / cache keys")
    for s in sheets:
        if set(s.sample_id) != set(key.sample_id):
            problems.append("sheet sample ids differ from key")
    a, b = (s.set_index("sample_id").sort_index() for s in sheets)
    if not a[["statement", "response"]].equals(b[["statement", "response"]]):
        problems.append("sheets A and B differ in statement/response")

    records: dict[str, list] = {}
    for r in GenerationCache(CACHE).read_all():
        records.setdefault(r.cache_key, []).append(r)
    items = {i.item_id: i for i in load_faulborn_items()}
    analysed = load().drop_duplicates(["cache_key", "evaluator_id"])
    gen = analysed.drop_duplicates("cache_key").set_index("cache_key")
    outcome = analysed.pivot_table(index="cache_key", columns="evaluator_id", values="outcome", aggfunc="first")
    evaluators = [c for c in key.columns if c.startswith(("faulborn_nli_", "deterministic"))]

    for row in key.itertuples(index=False):
        sid, ck = row.sample_id, row.cache_key
        recs = records.get(ck, [])
        if len(recs) != 1:
            problems.append(f"{sid}: {len(recs)} cache records for {ck[:12]}")
            continue
        sheet = a.loc[sid]
        if sheet.response != recs[0].raw_text:
            problems.append(f"{sid}: response differs from cached raw_text")
        item = items[row.item_id]
        expected = item.prompt_original if row.polarity == "orig" else item.metadata["pol_opposite_gpt"]
        if sheet.statement != expected:
            problems.append(f"{sid}: statement differs from item {row.item_id}/{row.polarity}")
        g = gen.loc[ck]
        for col in META:
            if str(g[col]) != str(getattr(row, col)):
                problems.append(f"{sid}: {col} key={getattr(row, col)!r} artifacts={g[col]!r}")
        for ev in evaluators:
            if outcome.at[ck, ev] != getattr(row, ev):
                problems.append(f"{sid}: {ev} key={getattr(row, ev)!r} stored={outcome.at[ck, ev]!r}")

    if problems:
        print(f"FAILED: {len(problems)} problem(s)")
        print("\n".join(problems[:50]))
        return 1
    print(f"OK: {len(key)} key rows exactly match sheets, cache, items and {len(evaluators)} stored evaluators")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
