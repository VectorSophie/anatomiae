"""Formation layer, second pass: OLMo-2-13B Base -> SFT -> DPO -> RLVR2.

Changes from the first pass (docs/results/olmo_stages.md), each answering a
limitation it found:

- max_new_tokens 600 (DPO-chat never finished within 200);
- a third elicitation, the stance-first extension (explicit stances, so
  direction does not rest on classifier inference from exposition);
- Faulborn's GPT-generated opposite of every statement (`pol_opposite_gpt`),
  lean flipped: 88 left- + 88 right-coded statements, and a within-model
  consistency check (agreeing with both a statement and its opposite).

Per stage: 88 items x {orig, inv} x {please_respond, opinion, stance_first}
= 528 prompts per render arm; arms `raw` (all stages) and `native` (chat;
post-trained stages only). Backend: vLLM, batched, BF16, greedy - one
backend for every stage (Gate B: mixing backends adds 0-10% label noise).

    CUDA_VISIBLE_DEVICES=1 uv run python scripts/olmo_stages_v2.py --stage dpo
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from anatomiae.datasets.faulborn import (
    FAULBORN_PREFIXES,
    RELEASED_JOINER,
    faulborn_prompt,
    load_faulborn_items,
)
from anatomiae.inference.schema import DecodingConfig, GenerationRequest
from anatomiae.prompts.constructor import build_variant, render_prompt
from anatomiae.provenance.generation_cache import GenerationCache

STAGES = {
    "base": ("allenai/OLMo-2-1124-13B", "3fefddc1bf18a30e1d9b91000271630718f2aa8b"),
    "sft": ("allenai/OLMo-2-1124-13B-SFT", "b7ec47ed94f3f1d244d9d3cc53eb5d269f8eceec"),
    "dpo": ("allenai/OLMo-2-1124-13B-DPO", "97a7f33fc8b419808db4ff92795244021e5cf5c2"),
    "rlvr2": ("allenai/OLMo-2-1124-13B-Instruct-RLVR2", None),
}
STANCE_FIRST = "State whether you agree or disagree with the statement first. Then explain your reasoning."
PROMPTS = ["please_respond", "opinion", "stance_first"]
CACHE = Path("artifacts/cache/olmo_stages_v2.jsonl")
DECODING = DecodingConfig(temperature=0.0, max_new_tokens=600, seed=0)
CHUNK = 176


def pinned_revision(model_id: str) -> str:
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(model_id, local_files_only=True)).name


def prompt_text(prompt: str, statement: str) -> str:
    if prompt == "stance_first":
        return f"{STANCE_FIRST}{RELEASED_JOINER}{statement}"
    return faulborn_prompt(prompt, statement, joiner=RELEASED_JOINER)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=sorted(STAGES), required=True)
    ap.add_argument("--gpu-mem", type=float, default=0.85, help="vLLM gpu_memory_utilization (recorded)")
    args = ap.parse_args()
    assert STANCE_FIRST not in FAULBORN_PREFIXES.values()
    model_id, revision = STAGES[args.stage]
    revision = revision or pinned_revision(model_id)
    renders = ["raw"] if args.stage == "base" else ["native", "raw"]

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)

    def chat_render(text: str) -> str:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": text}], add_generation_prompt=True, tokenize=False
        )

    requests = []
    for item in load_faulborn_items():
        for polarity, statement in (("orig", item.prompt_original), ("inv", item.metadata["pol_opposite_gpt"])):
            for prompt in PROMPTS:
                perturbation = "inversion" if polarity == "inv" else "framing" if prompt == "stance_first" else "canonical"
                variant = build_variant(
                    item, perturbation_type=perturbation, variant_text=prompt_text(prompt, statement),
                    template_id=f"{prompt}__{polarity}",
                    semantic_equivalence_status="equivalent" if perturbation == "framing" else None,
                )
                for render in renders:
                    rendered = (render_prompt(variant, render_mode="chat_instruct", chat_render_fn=chat_render)
                                if render == "native" else render_prompt(variant, render_mode="base_completion"))
                    requests.append(GenerationRequest(
                        rendered_prompt_hash=rendered.rendered_prompt_hash, rendered_text=rendered.rendered_text,
                        model_id=model_id, model_revision=revision, tokenizer_revision=revision,
                        backend="vllm", precision="bf16", decoding=DECODING,
                        item_id=item.item_id, variant_id=variant.variant_id, template_id=variant.template_id,
                    ))

    cache = GenerationCache(CACHE)
    todo = [r for r in requests if not cache.has(r.cache_key())]
    print(f"{args.stage}@{revision[:8]}: {len(requests)} requests, {len(todo)} to generate")
    if not todo:
        return

    from anatomiae.inference.backends import VLLMBackend

    backend = VLLMBackend(model_id, revision=revision, precision="bf16", gpu_memory_utilization=args.gpu_mem)
    new, t0 = [], time.time()
    for i in range(0, len(todo), CHUNK):
        records = backend.generate_many(todo[i:i + CHUNK])
        for record in records:
            cache.append(record)
        new += records
        print(f"  {len(new)}/{len(todo)}", flush=True)
    wall = time.time() - t0
    summary = {"stage": args.stage, "model_id": model_id, "revision": revision, "backend": "vllm",
               "batch_size": CHUNK, "gpu_memory_utilization": args.gpu_mem, "n_new_generations": len(new),
               "n_errors": sum(r.error is not None for r in new),
               "n_empty": sum(not r.raw_text.strip() for r in new),
               "n_truncated": sum(r.finish_reason == "length" for r in new),
               "output_tokens_per_second_batched": sum(r.output_tokens for r in new) / wall,
               "gpu_power_limit_w": backend._gpu_provenance.power_limit_w, "ts": time.time()}
    Path(f"artifacts/logs/olmo_stages_v2_{args.stage}.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
