"""Backend-divergence smoke test: same model, same prompt, deterministic
decoding, through both vLLM and Transformers (§60 - "for each family, test
a small golden set through both, compare rendered input, tokenization,
outputs, templates, termination behavior").

Reuses the already-cached OLMo-2-0425-1B-Instruct from scripts/smoke_test.py
so this doesn't consume any more of the constrained download bandwidth -
purely GPU-compute-bound, safe to run while a larger download is in flight.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from anatomiae.provenance.gpu_guard import GPUIsolationError, verify_gpu_isolation  # noqa: E402

MODEL_ID = "allenai/OLMo-2-0425-1B-Instruct"
PROMPT = "In one paragraph, what should the government's role be in regulating the economy?"

OUT_PATH = Path(__file__).resolve().parents[1] / "artifacts" / "logs" / "backend_divergence_result.json"


def main() -> None:
    try:
        prov = verify_gpu_isolation()
    except GPUIsolationError as e:
        print(f"ABORT: GPU isolation check failed: {e}", file=sys.stderr)
        raise SystemExit(1) from e
    print(f"GPU isolation OK: physical_index={prov.physical_index} uuid={prov.uuid}")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    messages = [{"role": "user", "content": PROMPT}]
    rendered_prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True)

    print(f"Rendered prompt ({len(input_ids)} tokens):\n{rendered_prompt!r}\n")

    from vllm import LLM, SamplingParams

    t0 = time.time()
    llm = LLM(model=MODEL_ID, dtype="bfloat16", gpu_memory_utilization=0.3, enforce_eager=True)
    load_s = time.time() - t0

    sampling_params = SamplingParams(temperature=0.0, max_tokens=200)

    t0 = time.time()
    outputs = llm.generate([rendered_prompt], sampling_params)
    gen_s = time.time() - t0

    vllm_completion = outputs[0].outputs[0].text
    vllm_finish_reason = outputs[0].outputs[0].finish_reason
    vllm_output_tokens = len(outputs[0].outputs[0].token_ids)

    transformers_result_path = OUT_PATH.parent / "smoke_test_result.json"
    transformers_completion = None
    if transformers_result_path.exists():
        transformers_completion = json.loads(transformers_result_path.read_text())["raw_completion"]

    record = {
        "model_id": MODEL_ID,
        "backend": "vllm",
        "gpu_provenance": {
            "physical_index": prov.physical_index,
            "uuid": prov.uuid,
        },
        "rendered_prompt": rendered_prompt,
        "prompt_tokens": len(input_ids),
        "vllm_completion": vllm_completion,
        "vllm_finish_reason": vllm_finish_reason,
        "vllm_output_tokens": vllm_output_tokens,
        "transformers_completion_for_comparison": transformers_completion,
        "completions_identical": transformers_completion == vllm_completion
        if transformers_completion
        else None,
        "load_seconds": load_s,
        "generation_seconds": gen_s,
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(record, indent=2))

    print(f"\nLoad: {load_s:.1f}s  Generate: {gen_s:.1f}s  Tokens: {vllm_output_tokens}")
    print(f"\n--- vLLM completion ---\n{vllm_completion}\n-----------------------")
    if transformers_completion:
        print(f"\nIdentical to Transformers (greedy) output: {record['completions_identical']}")
        if not record["completions_identical"]:
            print("--- Transformers completion (for diff) ---")
            print(transformers_completion)
    print(f"\nWrote {OUT_PATH}")


if __name__ == "__main__":
    main()
