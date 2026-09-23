"""Gate C analysis rows: one per lineage, from cache + evaluation store only.

OLMo row: OLMo-2-13B-SFT Transformers records from the Gate B cache.
Other lineages: artifacts/cache/gate_c.jsonl. Same 60 prompts, decoding and
evaluators everywhere, so rows are comparable as a *mechanism* check. No
cross-lineage political comparison is drawn from 60 prompts.

Outputs: artifacts/tables/gate_c_lineages.{csv,parquet,md,tex}
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from anatomiae.analysis.frame import build_analysis_frame, export_table
from anatomiae.evaluators.taxonomy import POSITION_OUTCOMES
from anatomiae.provenance.evaluation_store import EvaluationStore
from anatomiae.provenance.generation_cache import GenerationCache

LINEAGE = {"allenai/OLMo-2-1124-13B-SFT": "OLMo 2", "IFM/Amber": "Amber",
           "Qwen/Qwen2.5-14B-Instruct": "Qwen2.5"}
EVALUATORS = {"deterministic_stance_v1": "det",
              "faulborn_nli_reconstructed_as_written_seed42": "recon_s42",
              "faulborn_nli_zeroshot": "zeroshot"}


def main() -> None:
    recs = [r for r in GenerationCache("artifacts/cache/backend_equivalence.jsonl").read_all()
            if r.request.model_id == "allenai/OLMo-2-1124-13B-SFT" and r.request.backend == "transformers"]
    evals = EvaluationStore("artifacts/evaluations/backend_equivalence.jsonl").read_all()
    if Path("artifacts/cache/gate_c.jsonl").exists():
        recs += GenerationCache("artifacts/cache/gate_c.jsonl").read_all()
        evals += EvaluationStore("artifacts/evaluations/gate_c.jsonl").read_all()
    df = build_analysis_frame(recs, evals)
    df = df[df["evaluator_id"].isin(EVALUATORS)]

    rows = []
    for (model_id, rev), g in df.groupby(["model_id", "model_revision"]):
        gen = g.drop_duplicates("cache_key")
        row = {"lineage": LINEAGE.get(model_id, model_id), "model_id": model_id, "revision": rev[:12],
               "n_generations": len(gen), "errors": int(gen["error"].notna().sum()),
               "empty": int((gen["output_tokens"] == 0).sum()),
               "truncated_share": round(gen["truncated"].mean(), 3)}
        for ev, short in EVALUATORS.items():
            e = g[g["evaluator_id"] == ev]
            row[f"{short}_n_scored"] = len(e)
            row[f"{short}_position_rate"] = round(e["outcome"].isin(POSITION_OUTCOMES).mean(), 3) if len(e) else None
            row[f"{short}_top_outcome"] = e["outcome"].mode().iat[0] if len(e) else None
        rows.append(row)
    table = pd.DataFrame(rows).sort_values("lineage")
    export_table(table, Path("artifacts/tables/gate_c_lineages"), formats=("csv", "parquet", "md", "tex"))
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
