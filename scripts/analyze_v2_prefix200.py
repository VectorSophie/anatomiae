"""200- vs 600-token view of the v2 stage study (Expression-layer sensitivity).

Greedy decoding: a 200-token run of the same request is the first 200 tokens
of the 600-token run (up to batch-numerics effects in vLLM, see Gate B). So
instead of regenerating, each v2 response is cut to its first 200 tokens
(re-tokenized with the model's own tokenizer, so the cut can differ from the
generated token boundary by a token or two) and re-scored in memory.

Nothing is written to the generation cache or evaluation store; results go
to artifacts/tables/olmo_stages_v2_prefix200.{csv,parquet,md,tex}.

    CUDA_VISIBLE_DEVICES=1 uv run python scripts/analyze_v2_prefix200.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
from analyze_olmo_stages import metrics
from analyze_olmo_stages_v2 import load

from anatomiae.analysis.frame import export_table
from anatomiae.evaluators.deterministic_stance import DeterministicStanceEvaluator
from anatomiae.evaluators.faulborn_nli import FAULBORN_TO_OUTCOME, FaulbornNLIEvaluator
from anatomiae.evaluators.taxonomy import POSITION_OUTCOMES
from anatomiae.provenance.generation_cache import GenerationCache

RECON = Path("/data/jackb/anatomiae/models/faulborn_classifier_reconstruction")
NLI = {"faulborn_nli_reconstructed_as_written_seed42": RECON / "as_written_seed42",
       "faulborn_nli_reconstructed_leakage_free_seed42": RECON / "leakage_free_seed42"}
DET = "deterministic_stance_v1"
CUT = 200


def main() -> None:
    from transformers import AutoTokenizer

    recs = GenerationCache("artifacts/cache/olmo_stages_v2.jsonl").read_all()
    toks = {}
    cut = []
    for r in recs:
        key = (r.request.model_id, r.request.model_revision)
        if key not in toks:
            toks[key] = AutoTokenizer.from_pretrained(key[0], revision=key[1])
        if r.output_tokens <= CUT:
            cut.append(r)
            continue
        ids = toks[key](r.raw_text, add_special_tokens=False).input_ids[:CUT]
        cut.append(r.model_copy(update={"raw_text": toks[key].decode(ids), "finish_reason": "length"}))

    labels = {DET: {e.generation_cache_key: e.outcome for e in DeterministicStanceEvaluator().evaluate_many(cut)}}
    nonempty = [r for r in cut if r.raw_text.strip() and r.error is None]
    for ev_id, path in NLI.items():
        ev = FaulbornNLIEvaluator(model_path=str(path), evaluator_id=ev_id)
        pred = ev.classify_texts([r.raw_text for r in nonempty])
        out = {r.cache_key: "malformed" for r in cut}
        out.update({r.cache_key: FAULBORN_TO_OUTCOME[lab] for r, (lab, _) in zip(nonempty, pred, strict=True)})
        labels[ev_id] = out
        del ev

    full = load()
    full = full[full["evaluator_id"].isin(labels)].copy()
    view200 = full.copy()
    view200["outcome"] = [labels[e][k] for e, k in zip(view200["evaluator_id"], view200["cache_key"], strict=True)]
    view200["is_position"] = view200["outcome"].isin(POSITION_OUTCOMES)
    view200["is_directional"] = view200["outcome"].isin({"agreement", "disagreement"})
    view200["left_aligned"] = ((view200["outcome"] == "agreement") & (view200["item_lean"] == "left")) | (
        (view200["outcome"] == "disagreement") & (view200["item_lean"] == "right"))
    view200["truncated"] = view200["truncated"] | (view200["output_tokens"] > CUT)

    keys = ["stage", "render", "prompt", "evaluator_id"]
    rows = []
    for k, g600 in full.groupby(keys):
        g200 = view200.loc[g600.index]
        m6, m2 = metrics(g600), metrics(g200)
        rows.append({**dict(zip(keys, k, strict=True)), "n": len(g600),
                     "truncated_share_200": m2["truncated_share"], "truncated_share_600": m6["truncated_share"],
                     "position_rate_200": m2["position_rate"], "position_rate_600": m6["position_rate"],
                     "direction_index_200": m2["direction_index"], "direction_index_600": m6["direction_index"],
                     "label_changed_share": (g200["outcome"] != g600["outcome"]).mean(),
                     "direction_flip_share": ((g200["is_directional"] & g600["is_directional"])
                                              & (g200["outcome"] != g600["outcome"])).mean()})
    table = pd.DataFrame(rows).round(4)
    export_table(table, Path("artifacts/tables/olmo_stages_v2_prefix200"), formats=("csv", "parquet", "md", "tex"))
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
