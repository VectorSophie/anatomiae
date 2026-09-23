"""Score an existing generation cache with every available evaluator.

Never generates. Reads immutable GenerationRecords, runs each evaluator
only on records it has not already scored (per evaluator id + version),
and appends results to an EvaluationStore. Idempotent: re-running scores
nothing new unless an evaluator was added or its version bumped.

Evaluators:
- deterministic_stance_v1 (rule-based; CPU);
- faulborn_nli_zeroshot: Faulborn's inference procedure over untouched
  facebook/bart-large-mnli (their own zero-shot baseline);
- faulborn_nli_reconstructed_<variant>_seed<N>: Faulborn's procedure over
  each anatomiae reconstruction of their fine-tuned classifier found under
  RECON_ROOT. Reconstructions, not the authors' (missing) weights.

    CUDA_VISIBLE_DEVICES=1 uv run python scripts/score_cache.py \\
        --cache artifacts/cache/faulborn_reproduction.jsonl \\
        --store artifacts/evaluations/faulborn_reproduction.jsonl
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from anatomiae.evaluators.deterministic_stance import DeterministicStanceEvaluator
from anatomiae.provenance.evaluation_store import EvaluationStore, results_summary
from anatomiae.provenance.generation_cache import GenerationCache

RECON_ROOT = Path("/data/jackb/anatomiae/models/faulborn_classifier_reconstruction")


def nli_specs(skip_nli: bool) -> list[tuple[str, str]]:
    if skip_nli:
        return []
    specs = [("faulborn_nli_zeroshot", "facebook/bart-large-mnli")]
    for d in sorted(RECON_ROOT.glob("*_seed*")):
        if (d / "anatomiae_provenance.json").exists():
            specs.append((f"faulborn_nli_reconstructed_{d.name}", str(d)))
    return specs


def score(store: EvaluationStore, records, evaluator) -> int:
    todo = [r for r in records if not store.has(r.cache_key, evaluator.evaluator_id, evaluator.evaluator_version)]
    if not todo:
        return 0
    return store.append_many(evaluator.evaluate_many(todo))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--store", required=True)
    ap.add_argument("--skip-nli", action="store_true", help="deterministic evaluator only (no GPU)")
    args = ap.parse_args()

    records = GenerationCache(Path(args.cache)).read_all()
    store = EvaluationStore(Path(args.store))
    print(f"{len(records)} generation records from {args.cache} (no generation performed)")

    n = score(store, records, DeterministicStanceEvaluator())
    print(f"deterministic_stance_v1: {n} new")

    for evaluator_id, model_path in nli_specs(args.skip_nli):
        from anatomiae.evaluators.faulborn_nli import FaulbornNLIEvaluator

        ev = FaulbornNLIEvaluator(model_path=model_path, evaluator_id=evaluator_id)
        n = score(store, records, ev)
        print(f"{evaluator_id}: {n} new")
        del ev

    print("store now holds:", results_summary(args.store))


if __name__ == "__main__":
    main()
