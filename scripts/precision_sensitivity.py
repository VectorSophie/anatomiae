"""OLMo-2-13B Base: FP32 vs BF16 inference, same checkpoint, one dtype per process.

OLMo-2-1124-13B (Base) is stored in FP32 while its SFT/DPO/RLVR descendants
are BF16. Comparing Base-FP32 against SFT-BF16 would confound training
stage with inference precision, so this measures the precision effect on
its own, on the *same* Base checkpoint.

Run once per dtype, in separate processes (never both resident: ~52 GB
FP32 + ~27 GB BF16), e.g.:

    CUDA_VISIBLE_DEVICES=1 uv run python scripts/precision_sensitivity.py --precision fp32
    CUDA_VISIBLE_DEVICES=1 uv run python scripts/precision_sensitivity.py --precision bf16

Each run: full GPU preflight (via the backend) -> load at the requested
dtype -> validate the cast (every parameter at the requested dtype, zero
non-finite values) -> generate -> append to the shared immutable cache
(precision is part of the cache key, so the runs never collide) -> write a
per-run summary (load time, peak VRAM, throughput, applied power limit).

Items/prompts: the first N Faulborn items, released prompt strings verbatim
(hyphen joiner), rendered as raw base-model completions - Faulborn's own
base-model protocol, including the elicitation limitation they flag for
base models. Held identical across dtypes: revision, prompts, backend,
greedy decoding, seed.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from anatomiae.datasets.faulborn import (
    RELEASED_JOINER,
    faulborn_prompt,
    load_faulborn_items,
)
from anatomiae.inference.backends import TransformersBackend
from anatomiae.inference.schema import DecodingConfig, GenerationRequest
from anatomiae.prompts.constructor import build_variant, render_prompt
from anatomiae.provenance.generation_cache import GenerationCache

MODEL_ID = "allenai/OLMo-2-1124-13B"
# Pinned to the exact snapshot downloaded (artifacts/logs/bulk_download.jsonl),
# not "main" - a moving ref would silently change what "Base" means.
REVISION = "3fefddc1bf18a30e1d9b91000271630718f2aa8b"
PREFIX_NAMES = ["please_respond", "opinion"]
CACHE_PATH = Path("artifacts/cache/precision_sensitivity.jsonl")
DECODING = DecodingConfig(temperature=0.0, max_new_tokens=200, seed=0)


def validate_cast(model, precision: str) -> dict:
    import torch

    expected = {"fp32": torch.float32, "bf16": torch.bfloat16}[precision]
    wrong_dtype = [n for n, p in model.named_parameters() if p.dtype != expected]
    non_finite = sum(int((~torch.isfinite(p)).sum()) for p in model.parameters())
    n_params = sum(p.numel() for p in model.parameters())
    report = {
        "expected_dtype": str(expected),
        "n_parameters": n_params,
        "params_with_unexpected_dtype": len(wrong_dtype),
        "non_finite_values": non_finite,
    }
    if wrong_dtype or non_finite:
        raise RuntimeError(f"cast validation failed, refusing to generate: {report}")
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--precision", choices=["fp32", "bf16"], required=True)
    ap.add_argument("--n-items", type=int, default=30)
    args = ap.parse_args()

    import torch

    backend = TransformersBackend(MODEL_ID, revision=REVISION)
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    model = backend._model_for_precision(args.precision)
    load_seconds = time.time() - t0
    cast_report = validate_cast(model, args.precision)
    print(f"loaded {MODEL_ID}@{REVISION[:8]} as {args.precision} in {load_seconds:.1f}s; cast OK: {cast_report}")

    cache = GenerationCache(CACHE_PATH)
    items = load_faulborn_items(args.n_items)
    new_records = []
    for item in items:
        for prefix_name in PREFIX_NAMES:
            text = faulborn_prompt(prefix_name, item.prompt_original, joiner=RELEASED_JOINER)
            variant = build_variant(
                item, perturbation_type="canonical", variant_text=text,
                template_id=f"faulborn_released_{prefix_name}",
            )
            rendered = render_prompt(variant, render_mode="base_completion")
            request = GenerationRequest(
                rendered_prompt_hash=rendered.rendered_prompt_hash,
                rendered_text=rendered.rendered_text,
                model_id=MODEL_ID,
                model_revision=REVISION,
                tokenizer_revision=REVISION,
                backend="transformers",
                precision=args.precision,
                decoding=DECODING,
                item_id=item.item_id,
                variant_id=variant.variant_id,
                template_id=variant.template_id,
            )
            if cache.has(request.cache_key()):
                continue
            record = backend.generate(request)
            cache.append(record)
            new_records.append(record)

    gen_seconds = sum(r.latency_seconds for r in new_records)
    out_tokens = sum(r.output_tokens for r in new_records)
    summary = {
        "model_id": MODEL_ID,
        "revision": REVISION,
        "precision": args.precision,
        "load_seconds": load_seconds,
        "cast_validation": cast_report,
        "n_new_generations": len(new_records),
        "n_errors": sum(r.error is not None for r in new_records),
        "total_output_tokens": out_tokens,
        "total_generation_seconds": gen_seconds,
        "tokens_per_second": out_tokens / gen_seconds if gen_seconds else None,
        "peak_vram_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "peak_vram_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
        "gpu_physical_index": backend._gpu_provenance.physical_index,
        "gpu_power_limit_w": backend._gpu_provenance.power_limit_w,
        "note": "throughput measured with the GPU at its device-reported minimum power limit",
    }
    Path("artifacts/logs").mkdir(parents=True, exist_ok=True)
    Path(f"artifacts/logs/precision_run_{args.precision}.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
