"""Gate C: one checkpoint per independent lineage, end to end.

verify (pinned snapshot on disk, config identity, declared vs loaded
dtype) -> smoke (one generation must be non-empty and error-free, or
abort) -> GPU preflight (inside the backend) -> generation of the same
60 Faulborn prompts used in Gates B / precision -> immutable cache ->
scoring (scripts/score_cache.py, separate process) -> analysis row
(scripts/analyze_gate_c.py).

The OLMo lineage's row reuses the OLMo-2-13B-SFT Transformers records from
Gate B (identical prompts/decoding); this script covers the others.

    CUDA_VISIBLE_DEVICES=1 uv run python scripts/gate_c_lineage.py --model amber_final
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

# (model_id, pinned revision, render mode). Amber has no chat template: it is
# a base model, run with Faulborn's base-model completion protocol.
MODELS = {
    "amber_final": ("IFM/Amber", "2d35937a32b99aec0d59dd772b54e1bb08719a7f", "base_completion"),
    "qwen25_14b_instruct": ("Qwen/Qwen2.5-14B-Instruct", None, "chat_instruct"),
}
PREFIX_NAMES = ["please_respond", "opinion"]
CACHE = Path("artifacts/cache/gate_c.jsonl")
DECODING = DecodingConfig(temperature=0.0, max_new_tokens=200, seed=0)


def pinned_revision(model_id: str) -> str:
    from huggingface_hub import snapshot_download

    path = Path(snapshot_download(model_id, local_files_only=True))
    return path.name


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=sorted(MODELS), required=True)
    ap.add_argument("--n-items", type=int, default=30)
    args = ap.parse_args()
    model_id, revision, render_mode = MODELS[args.model]
    revision = revision or pinned_revision(model_id)
    log: dict = {"model": args.model, "model_id": model_id, "revision": revision, "render_mode": render_mode}

    # verify
    from transformers import AutoConfig, AutoTokenizer

    config = AutoConfig.from_pretrained(model_id, revision=revision, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, local_files_only=True)
    log["verify"] = {"architectures": config.architectures, "declared_torch_dtype": str(config.torch_dtype),
                     "has_chat_template": bool(tokenizer.chat_template)}
    if (render_mode == "chat_instruct") != bool(tokenizer.chat_template):
        raise SystemExit(f"render mode {render_mode} inconsistent with chat template presence: {log['verify']}")

    def chat_render(text: str) -> str:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": text}], add_generation_prompt=True, tokenize=False
        )

    from anatomiae.inference.backends import TransformersBackend

    backend = TransformersBackend(model_id, revision=revision)
    model = backend._model_for_precision("bf16")
    log["verify"]["loaded_param_dtypes"] = sorted({str(p.dtype) for p in model.parameters()})

    def request_for(item, prefix_name):
        text = faulborn_prompt(prefix_name, item.prompt_original, joiner=RELEASED_JOINER)
        variant = build_variant(item, perturbation_type="canonical", variant_text=text,
                                template_id=f"faulborn_released_{prefix_name}")
        rendered = (render_prompt(variant, render_mode="chat_instruct", chat_render_fn=chat_render)
                    if render_mode == "chat_instruct" else render_prompt(variant, render_mode="base_completion"))
        return GenerationRequest(
            rendered_prompt_hash=rendered.rendered_prompt_hash, rendered_text=rendered.rendered_text,
            model_id=model_id, model_revision=revision, tokenizer_revision=revision,
            backend="transformers", precision="bf16", decoding=DECODING,
            item_id=item.item_id, variant_id=variant.variant_id, template_id=variant.template_id,
        )

    items = load_faulborn_items(args.n_items)
    cache = GenerationCache(CACHE)

    # smoke: the first request doubles as the smoke test and is cached like any other
    first = request_for(items[0], PREFIX_NAMES[0])
    if not cache.has(first.cache_key()):
        rec = backend.generate(first)
        if rec.error or not rec.raw_text.strip():
            raise SystemExit(f"smoke generation failed, aborting: error={rec.error!r} text={rec.raw_text[:80]!r}")
        cache.append(rec)
    log["smoke"] = "ok"

    new = []
    for item in items:
        for prefix_name in PREFIX_NAMES:
            request = request_for(item, prefix_name)
            if cache.has(request.cache_key()):
                continue
            record = backend.generate(request)
            cache.append(record)
            new.append(record)
    gen_s = sum(r.latency_seconds for r in new)
    toks = sum(r.output_tokens for r in new)
    log.update(n_new_generations=len(new), n_errors=sum(r.error is not None for r in new),
               tokens_per_second=toks / gen_s if gen_s else None,
               gpu_power_limit_w=backend._gpu_provenance.power_limit_w, ts=time.time())
    Path(f"artifacts/logs/gate_c_{args.model}.json").write_text(json.dumps(log, indent=2))
    print(json.dumps(log, indent=2))


if __name__ == "__main__":
    main()
