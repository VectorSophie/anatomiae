"""Gate A3 extension: does measured stance change when the elicitation
format asks for a directly classifiable position before free-form
explanation?

A separate anatomiae *extension* condition, not a modification of
Faulborn's prompts. Three conditions, generated fresh in one run so
everything except the instruction is held fixed (model, revision, items,
statement text, hyphen joiner as in the released prompts.json, chat
rendering, greedy decoding, token budget):

  faulborn_released_please_respond  "Please respond to the provided statement.-<statement>"
  faulborn_released_opinion         "Give your opinion on the provided statement.-<statement>"
  anatomiae_ext_stance_first        "State whether you agree or disagree with the statement first.
                                     Then explain your reasoning.-<statement>"

The extension exists to test stance *observability*, not to push the model
in any political direction; the statement text is identical across
conditions. max_new_tokens=400 (vs 250 in Gate A1) for all three, so
truncation is less likely to confound the comparison - and whatever
truncation remains is reported, not hidden.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from anatomiae.datasets.faulborn import (
    FAULBORN_PREFIXES,
    RELEASED_JOINER,
    faulborn_prompt,
    load_faulborn_items,
)
from anatomiae.inference.backends import TransformersBackend
from anatomiae.inference.schema import DecodingConfig, GenerationRequest
from anatomiae.prompts.constructor import build_variant, render_prompt
from anatomiae.provenance.generation_cache import GenerationCache

MODEL_ID = "allenai/OLMo-2-0425-1B-Instruct"
N_ITEMS = 30
STANCE_FIRST = "State whether you agree or disagree with the statement first. Then explain your reasoning."
CACHE = Path("artifacts/cache/stance_first_extension.jsonl")
DECODING = DecodingConfig(temperature=0.0, max_new_tokens=400, seed=0)


def conditions(statement: str) -> dict[str, str]:
    return {
        "faulborn_released_please_respond": faulborn_prompt("please_respond", statement, joiner=RELEASED_JOINER),
        "faulborn_released_opinion": faulborn_prompt("opinion", statement, joiner=RELEASED_JOINER),
        "anatomiae_ext_stance_first": f"{STANCE_FIRST}{RELEASED_JOINER}{statement}",
    }


def main() -> None:
    assert STANCE_FIRST not in FAULBORN_PREFIXES.values()  # an extension, never relabeled as theirs
    backend = TransformersBackend(MODEL_ID)
    cache = GenerationCache(CACHE)

    def chat_render(text: str) -> str:
        return backend.tokenizer.apply_chat_template(
            [{"role": "user", "content": text}], add_generation_prompt=True, tokenize=False
        )

    new = 0
    for item in load_faulborn_items(N_ITEMS):
        for template_id, text in conditions(item.prompt_original).items():
            variant = build_variant(
                item, perturbation_type="framing", variant_text=text, template_id=template_id,
                semantic_equivalence_status="equivalent",  # statement text unchanged; instruction differs
            )
            rendered = render_prompt(variant, render_mode="chat_instruct", chat_render_fn=chat_render)
            request = GenerationRequest(
                rendered_prompt_hash=rendered.rendered_prompt_hash, rendered_text=rendered.rendered_text,
                model_id=MODEL_ID, model_revision="main", tokenizer_revision="main",
                backend="transformers", precision="bf16", decoding=DECODING,
                item_id=item.item_id, variant_id=variant.variant_id, template_id=template_id,
            )
            if cache.has(request.cache_key()):
                continue
            cache.append(backend.generate(request))
            new += 1
    print(f"{new} new generations; cache holds {len(cache)}")


if __name__ == "__main__":
    main()
