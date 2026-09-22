"""Tier-1 smoke test: real download, real generation, real GPU-isolation check.

Not part of the final pipeline architecture (that comes out of
docs/FRAMEWORK_DECISION.md) - this is a standalone script whose only job is
to prove the chain works end to end: GPU guard -> model load on physical
GPU 1 -> generation -> raw output captured with provenance, before any
larger downloads are committed to.
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

OUT_PATH = Path(__file__).resolve().parents[1] / "artifacts" / "logs" / "smoke_test_result.json"


def main() -> None:
    try:
        prov = verify_gpu_isolation()
    except GPUIsolationError as e:
        print(f"ABORT: GPU isolation check failed: {e}", file=sys.stderr)
        raise SystemExit(1) from e

    print(f"GPU isolation OK: physical_index={prov.physical_index} uuid={prov.uuid}")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16).to("cuda")
    load_s = time.time() - t0

    messages = [{"role": "user", "content": PROMPT}]
    input_ids = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    ).to("cuda")

    t0 = time.time()
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=200,
            do_sample=False,
            temperature=None,
            top_p=None,
        )
    gen_s = time.time() - t0

    completion = tokenizer.decode(
        output_ids[0][input_ids.shape[1] :], skip_special_tokens=True
    )

    record = {
        "model_id": MODEL_ID,
        "backend": "transformers",
        "gpu_provenance": {
            "physical_index": prov.physical_index,
            "uuid": prov.uuid,
            "pci_bus_id": prov.pci_bus_id,
            "driver_version": prov.driver_version,
            "cuda_runtime_version": prov.cuda_runtime_version,
        },
        "prompt": PROMPT,
        "raw_completion": completion,
        "load_seconds": load_s,
        "generation_seconds": gen_s,
        "output_tokens": int(output_ids.shape[1] - input_ids.shape[1]),
        "torch_version": torch.__version__,
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(record, indent=2))

    print(f"\nLoad: {load_s:.1f}s  Generate: {gen_s:.1f}s  Tokens: {record['output_tokens']}")
    print(f"\n--- raw completion ---\n{completion}\n----------------------")
    print(f"\nWrote {OUT_PATH}")


if __name__ == "__main__":
    main()
