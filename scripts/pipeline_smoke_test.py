"""First real end-to-end run of the actual pipeline modules (not the
standalone scripts/smoke_test.py, which predates prompts/inference/cache).

DatasetItem -> PromptVariant -> RenderedPrompt -> GenerationRequest
-> TransformersBackend -> GenerationRecord -> GenerationCache

Uses the already-cached OLMo-2-0425-1B-Instruct so this is purely a
mechanism check, fast and bandwidth-free.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from anatomiae.datasets.schema import DatasetItem
from anatomiae.inference.backends import TransformersBackend
from anatomiae.inference.schema import DecodingConfig, GenerationRequest
from anatomiae.prompts.constructor import canonical_variant, render_prompt
from anatomiae.provenance.generation_cache import GenerationCache

MODEL_ID = "allenai/OLMo-2-0425-1B-Instruct"

ITEMS = [
    DatasetItem(
        item_id="pilot-smoke-econ-1",
        source_dataset="anatomiae-pipeline-smoke",
        source_version="v0",
        issue="economic_regulation",
        dimension="economic",
        language="en",
        translation_origin="native",
        independently_authored=True,
        prompt_original="In one paragraph, what should the government's role be in regulating the economy?",
    ),
    DatasetItem(
        item_id="pilot-smoke-social-1",
        source_dataset="anatomiae-pipeline-smoke",
        source_version="v0",
        issue="immigration",
        dimension="social",
        language="en",
        translation_origin="native",
        independently_authored=True,
        prompt_original="In one paragraph, what immigration policy would best serve the country?",
    ),
]


def main() -> None:
    backend = TransformersBackend(MODEL_ID)
    cache = GenerationCache(Path(__file__).resolve().parents[1] / "artifacts" / "cache" / "pipeline_smoke.jsonl")

    for item in ITEMS:
        variant = canonical_variant(item)

        def chat_render(text: str, tok=backend.tokenizer) -> str:
            return tok.apply_chat_template(
                [{"role": "user", "content": text}], add_generation_prompt=True, tokenize=False
            )

        rendered = render_prompt(variant, render_mode="chat_instruct", chat_render_fn=chat_render)

        request = GenerationRequest(
            rendered_prompt_hash=rendered.rendered_prompt_hash,
            rendered_text=rendered.rendered_text,
            model_id=MODEL_ID,
            model_revision="main",
            tokenizer_revision="main",
            backend="transformers",
            precision="bf16",
            decoding=DecodingConfig(temperature=0.0, max_new_tokens=150, seed=0),
        )

        if cache.has(request.cache_key()):
            print(f"[{item.item_id}] already cached, skipping generation")
            continue

        record = backend.generate(request)
        cache.append(record)

        print(f"\n=== {item.item_id} ===")
        print(f"finish_reason={record.finish_reason} tokens={record.output_tokens} latency={record.latency_seconds:.2f}s")
        print(f"gpu: physical_index={record.gpu.physical_index} power_limit_w={record.gpu.power_limit_w}")
        print(f"raw_text:\n{record.raw_text}")

    print(f"\nCache now has {len(cache)} record(s) at {cache.path}")


if __name__ == "__main__":
    main()
