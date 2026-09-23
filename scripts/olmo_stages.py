"""Formation layer, first pass: OLMo-2-13B Base -> SFT -> DPO (-> RLVR2).

All 88 usable Faulborn items x 2 released prefixes (the first 30 items are the
Gate B / precision prompts); Transformers, BF16, greedy, 200 tokens as Gate B
and the precision run, so those measured floors apply directly. Two
rendering arms per post-trained stage:

  native - the stage's own format (chat template for SFT/DPO/RLVR2)
  raw    - raw completion, identical rendered text to Base

`raw` holds the input string fixed across all stages, separating what
training changed from what the chat format changes. Base has only `raw`.

Requests already present in the precision cache (Base, BF16 Transformers)
or the Gate B cache (SFT native, Transformers) are not regenerated.

    CUDA_VISIBLE_DEVICES=1 uv run python scripts/olmo_stages.py --stage dpo
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

STAGES = {
    "base": ("allenai/OLMo-2-1124-13B", "3fefddc1bf18a30e1d9b91000271630718f2aa8b"),
    "sft": ("allenai/OLMo-2-1124-13B-SFT", "b7ec47ed94f3f1d244d9d3cc53eb5d269f8eceec"),
    "dpo": ("allenai/OLMo-2-1124-13B-DPO", "97a7f33fc8b419808db4ff92795244021e5cf5c2"),
}
PREFIX_NAMES = ["please_respond", "opinion"]
CACHE = Path("artifacts/cache/olmo_stages.jsonl")
REUSE = [Path("artifacts/cache/precision_sensitivity.jsonl"), Path("artifacts/cache/backend_equivalence.jsonl")]
DECODING = DecodingConfig(temperature=0.0, max_new_tokens=200, seed=0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=sorted(STAGES), required=True)
    ap.add_argument("--n-items", type=int, default=88)
    args = ap.parse_args()
    model_id, revision = STAGES[args.stage]
    renders = ["raw"] if args.stage == "base" else ["native", "raw"]

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)

    def chat_render(text: str) -> str:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": text}], add_generation_prompt=True, tokenize=False
        )

    requests = []
    for item in load_faulborn_items(args.n_items):
        for prefix_name in PREFIX_NAMES:
            text = faulborn_prompt(prefix_name, item.prompt_original, joiner=RELEASED_JOINER)
            variant = build_variant(item, perturbation_type="canonical", variant_text=text,
                                    template_id=f"faulborn_released_{prefix_name}")
            for render in renders:
                rendered = (render_prompt(variant, render_mode="chat_instruct", chat_render_fn=chat_render)
                            if render == "native" else render_prompt(variant, render_mode="base_completion"))
                requests.append(GenerationRequest(
                    rendered_prompt_hash=rendered.rendered_prompt_hash, rendered_text=rendered.rendered_text,
                    model_id=model_id, model_revision=revision, tokenizer_revision=revision,
                    backend="transformers", precision="bf16", decoding=DECODING,
                    item_id=item.item_id, variant_id=variant.variant_id, template_id=variant.template_id,
                ))

    existing = {r.cache_key for p in REUSE if p.exists() for r in GenerationCache(p).read_all()}
    cache = GenerationCache(CACHE)
    todo = [r for r in requests if r.cache_key() not in existing and not cache.has(r.cache_key())]
    print(f"{args.stage}: {len(requests)} requests, {len(requests) - len(todo)} already cached, {len(todo)} to generate")
    if not todo:
        return

    from anatomiae.inference.backends import TransformersBackend

    backend = TransformersBackend(model_id, revision=revision)
    new = []
    for request in todo:
        record = backend.generate(request)
        cache.append(record)
        new.append(record)
    gen_s = sum(r.latency_seconds for r in new)
    summary = {"stage": args.stage, "model_id": model_id, "revision": revision,
               "n_new_generations": len(new), "n_errors": sum(r.error is not None for r in new),
               "tokens_per_second": sum(r.output_tokens for r in new) / gen_s if gen_s else None,
               "gpu_power_limit_w": backend._gpu_provenance.power_limit_w, "ts": time.time()}
    Path(f"artifacts/logs/olmo_stages_{args.stage}.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
