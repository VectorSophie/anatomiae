"""Regression check for the explicit attention mask (Qwen2.5: pad == eos).

Regenerates the first N cached Qwen2.5-14B-Instruct Gate C requests with the
current TransformersBackend, WITHOUT writing to any cache, and checks:
1. the "attention mask is not set" warning is no longer emitted;
2. every output is byte-identical to the cached one (greedy decoding on a
   single unpadded sequence: the mask is all ones, so semantics must not move).

    CUDA_VISIBLE_DEVICES=1 uv run python scripts/verify_attention_mask.py
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from anatomiae.inference.backends import TransformersBackend
from anatomiae.provenance.generation_cache import GenerationCache

MODEL_ID = "Qwen/Qwen2.5-14B-Instruct"
N = 10


class _Capture(logging.Handler):
    def __init__(self):
        super().__init__()
        self.messages: list[str] = []

    def emit(self, record):
        self.messages.append(record.getMessage())


def main() -> None:
    cached = [r for r in GenerationCache("artifacts/cache/gate_c.jsonl").read_all() if r.request.model_id == MODEL_ID][:N]
    capture = _Capture()
    logging.getLogger("transformers").addHandler(capture)
    backend = TransformersBackend(MODEL_ID, revision=cached[0].request.model_revision)
    results = []
    for old in cached:
        new = backend.generate(old.request)
        results.append({"item_id": old.request.item_id, "template_id": old.request.template_id,
                        "identical": new.raw_text == old.raw_text, "error": new.error})
    warned = [m for m in capture.messages if "attention mask" in m.lower()]
    summary = {"model_id": MODEL_ID, "n": len(results), "n_identical": sum(r["identical"] for r in results),
               "attention_mask_warnings": len(warned), "results": results}
    Path("artifacts/logs/verify_attention_mask.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "results"}, indent=2))
    if warned or summary["n_identical"] != len(results):
        raise SystemExit("attention-mask regression check FAILED")


if __name__ == "__main__":
    main()
