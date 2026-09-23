"""Gate B: Transformers vs vLLM on identical requests, one backend per process.

Everything is held fixed except the backend: model + pinned revision,
rendered prompt text (rendered once here with the HF tokenizer's chat
template, then handed as a string to either backend), BF16, greedy
decoding, max_new_tokens, seed. Items/prompts: the first 30 Faulborn
items x the released `please_respond` / `opinion` strings (hyphen joiner),
as in the precision run.

    CUDA_VISIBLE_DEVICES=1 uv run python scripts/backend_equivalence.py --model olmo1b_instruct --backend vllm

The 13B Base Transformers-BF16 side is not regenerated: those exact
requests (same revision, prompts, decoding) already exist in
artifacts/cache/precision_sensitivity.jsonl and are read from there by
scripts/analyze_backend_equivalence.py.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from anatomiae.datasets.faulborn import RELEASED_JOINER, faulborn_prompt, load_faulborn_items
from anatomiae.inference.schema import DecodingConfig, GenerationRequest
from anatomiae.prompts.constructor import build_variant, render_prompt
from anatomiae.provenance.generation_cache import GenerationCache

MODELS = {
    "olmo1b_instruct": ("allenai/OLMo-2-0425-1B-Instruct", "48d788eca847d4d7548f375ad03d3c9312f6139e", "chat_instruct"),
    "olmo13b_sft": ("allenai/OLMo-2-1124-13B-SFT", "b7ec47ed94f3f1d244d9d3cc53eb5d269f8eceec", "chat_instruct"),
    "olmo13b_base": ("allenai/OLMo-2-1124-13B", "3fefddc1bf18a30e1d9b91000271630718f2aa8b", "base_completion"),
}
PREFIX_NAMES = ["please_respond", "opinion"]
CACHE = Path("artifacts/cache/backend_equivalence.jsonl")
DECODING = DecodingConfig(temperature=0.0, max_new_tokens=200, seed=0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=sorted(MODELS), required=True)
    ap.add_argument("--backend", choices=["transformers", "vllm"], required=True)
    ap.add_argument("--n-items", type=int, default=30)
    args = ap.parse_args()
    model_id, revision, render_mode = MODELS[args.model]
    if args.model == "olmo13b_base" and args.backend == "transformers":
        raise SystemExit("13B Base Transformers-BF16 records already exist in the precision cache; not regenerating")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)

    def chat_render(text: str) -> str:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": text}], add_generation_prompt=True, tokenize=False
        )

    if args.backend == "vllm":
        from anatomiae.inference.backends import VLLMBackend

        backend = VLLMBackend(model_id, revision=revision, precision="bf16", gpu_memory_utilization=0.6)
    else:
        from anatomiae.inference.backends import TransformersBackend

        backend = TransformersBackend(model_id, revision=revision)

    cache = GenerationCache(CACHE)
    new = []
    for item in load_faulborn_items(args.n_items):
        for prefix_name in PREFIX_NAMES:
            text = faulborn_prompt(prefix_name, item.prompt_original, joiner=RELEASED_JOINER)
            variant = build_variant(item, perturbation_type="canonical", variant_text=text,
                                    template_id=f"faulborn_released_{prefix_name}")
            rendered = (render_prompt(variant, render_mode="chat_instruct", chat_render_fn=chat_render)
                        if render_mode == "chat_instruct" else render_prompt(variant, render_mode="base_completion"))
            request = GenerationRequest(
                rendered_prompt_hash=rendered.rendered_prompt_hash, rendered_text=rendered.rendered_text,
                model_id=model_id, model_revision=revision, tokenizer_revision=revision,
                backend=args.backend, precision="bf16", decoding=DECODING,
                item_id=item.item_id, variant_id=variant.variant_id, template_id=variant.template_id,
            )
            if cache.has(request.cache_key()):
                continue
            record = backend.generate(request)
            cache.append(record)
            new.append(record)

    gen_s = sum(r.latency_seconds for r in new)
    toks = sum(r.output_tokens for r in new)
    summary = {
        "model": args.model, "model_id": model_id, "revision": revision, "backend": args.backend,
        "n_new_generations": len(new), "n_errors": sum(r.error is not None for r in new),
        "total_output_tokens": toks, "total_generation_seconds": gen_s,
        "tokens_per_second_sequential": toks / gen_s if gen_s else None,
        "gpu_power_limit_w": backend._gpu_provenance.power_limit_w,
        "note": "one request at a time on both backends (no batching), GPU at its minimum power limit",
        "ts": time.time(),
    }
    Path(f"artifacts/logs/backend_equivalence_{args.model}_{args.backend}.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
