"""Real-hardware verification that GenerationRequest.decoding.seed actually
controls stochastic Transformers generation (not just recorded decoratively).

This is the check that caught a real bug: a first implementation attempt
passed `generator=torch.Generator(...)` to `model.generate()`, which
transformers 4.57.6 rejects outright (`ValueError: The following
model_kwargs are not used by the model: ['generator']`) rather than
silently ignoring it - see the comment in
src/anatomiae/inference/backends.py for the fix (global seeding
immediately before the call, which this transformers version actually
supports).

Invariant checked: same (model, prompt, decoding-config-including-seed)
-> byte-identical output across repeated calls; a different seed ->
(almost certainly) different output.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from anatomiae.inference.backends import TransformersBackend
from anatomiae.inference.schema import DecodingConfig, GenerationRequest

MODEL_ID = "allenai/OLMo-2-0425-1B-Instruct"


def main() -> None:
    backend = TransformersBackend(MODEL_ID)
    prompt = backend.tokenizer.apply_chat_template(
        [{"role": "user", "content": "Name one animal."}],
        add_generation_prompt=True,
        tokenize=False,
    )

    base_decoding = {"temperature": 0.9, "top_p": 0.9, "max_new_tokens": 30}
    req_seed_a = GenerationRequest(
        rendered_prompt_hash="seed-check-a",
        rendered_text=prompt,
        model_id=MODEL_ID,
        model_revision="main",
        tokenizer_revision="main",
        backend="transformers",
        precision="bf16",
        decoding=DecodingConfig(**base_decoding, seed=1234),
    )
    req_seed_b = req_seed_a.model_copy(
        update={"decoding": DecodingConfig(**base_decoding, seed=9999)}
    )

    run1 = backend.generate(req_seed_a)
    run2 = backend.generate(req_seed_a)
    run3 = backend.generate(req_seed_b)

    print(f"run1 (seed=1234): {run1.raw_text!r}")
    print(f"run2 (seed=1234): {run2.raw_text!r}")
    print(f"run3 (seed=9999): {run3.raw_text!r}")

    same_seed_reproducible = run1.raw_text == run2.raw_text
    diff_seed_differs = run1.raw_text != run3.raw_text

    print(f"\nSame seed -> identical output: {same_seed_reproducible}")
    print(f"Different seed -> different output: {diff_seed_differs}")

    assert same_seed_reproducible, "seed did not control generation - regression!"
    assert diff_seed_differs, "different seeds produced identical output - suspicious"
    print("\nPASS: seed semantics verified on real hardware.")


if __name__ == "__main__":
    main()
