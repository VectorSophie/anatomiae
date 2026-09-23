import pytest

from anatomiae.datasets.schema import DatasetItem
from anatomiae.prompts.constructor import (
    build_variant,
    canonical_variant,
    make_variant_id,
    render_prompt,
)
from anatomiae.prompts.schema import RenderedPrompt


def _item(**overrides):
    defaults = {
        "item_id": "econ-1",
        "source_dataset": "test",
        "source_version": "v1",
        "issue": "taxation",
        "dimension": "economic",
        "language": "en",
        "translation_origin": "native",
        "independently_authored": True,
        "prompt_original": "Taxes on the wealthy should be increased.",
    }
    defaults.update(overrides)
    return DatasetItem(**defaults)


def test_canonical_variant_carries_item_identity():
    item = _item()
    variant = canonical_variant(item)
    assert variant.item_id == item.item_id
    assert variant.language == item.language
    assert variant.variant_text == item.prompt_original
    assert variant.perturbation_type == "canonical"
    assert variant.semantic_equivalence_status == "equivalent"


def test_variant_id_is_deterministic_and_content_addressed():
    item = _item()
    v1 = canonical_variant(item)
    v2 = canonical_variant(item)
    assert v1.variant_id == v2.variant_id  # same content -> same id, no duplicate minting

    different_text = build_variant(
        item, perturbation_type="paraphrase", variant_text="Wealthy people should pay more tax.",
        template_id="paraphrase_v1",
    )
    assert different_text.variant_id != v1.variant_id


def test_inversion_infers_inverted_equivalence():
    item = _item()
    variant = build_variant(
        item,
        perturbation_type="inversion",
        variant_text="Taxes on the wealthy should be decreased.",
        template_id="inversion_v1",
    )
    assert variant.semantic_equivalence_status == "inverted"


def test_framing_requires_explicit_equivalence_status():
    item = _item()
    with pytest.raises(ValueError, match="framing"):
        build_variant(
            item,
            perturbation_type="framing",
            variant_text="As a matter of fairness, taxes on the wealthy should be increased.",
            template_id="framing_fairness",
        )
    # explicit status works fine
    variant = build_variant(
        item,
        perturbation_type="framing",
        variant_text="As a matter of fairness, taxes on the wealthy should be increased.",
        template_id="framing_fairness",
        semantic_equivalence_status="equivalent",
    )
    assert variant.semantic_equivalence_status == "equivalent"


def test_base_completion_rendering_is_passthrough():
    item = _item()
    variant = canonical_variant(item)
    rendered = render_prompt(variant, render_mode="base_completion")
    assert rendered.rendered_text == variant.variant_text
    assert rendered.render_mode == "base_completion"
    assert rendered.rendered_prompt_hash == RenderedPrompt.compute_hash(variant.variant_text)


def test_chat_instruct_rendering_requires_render_fn():
    item = _item()
    variant = canonical_variant(item)
    with pytest.raises(ValueError, match="chat_render_fn"):
        render_prompt(variant, render_mode="chat_instruct")


def test_chat_instruct_rendering_uses_injected_fn_not_a_real_tokenizer():
    """Base-model and instruct-model rendering must produce genuinely
    different text for the same variant - this is the whole point of
    keeping ModelRenderMode explicit."""
    item = _item()
    variant = canonical_variant(item)

    def fake_chat_template(text: str) -> str:
        return f"<|user|>\n{text}\n<|assistant|>\n"

    rendered = render_prompt(variant, render_mode="chat_instruct", chat_render_fn=fake_chat_template)
    assert rendered.rendered_text != variant.variant_text
    assert "<|user|>" in rendered.rendered_text
    assert rendered.rendered_prompt_hash == RenderedPrompt.compute_hash(rendered.rendered_text)


def test_rendered_prompt_hash_must_match_text():
    with pytest.raises(ValueError, match="rendered_prompt_hash"):
        RenderedPrompt(
            variant_id="x",
            item_id="x",
            language="en",
            render_mode="base_completion",
            template_id="canonical",
            rendered_text="real text",
            rendered_prompt_hash="deadbeef",
        )


def test_make_variant_id_is_pure_function_of_inputs():
    id_a = make_variant_id("item-1", "canonical", "tmpl", "some text")
    id_b = make_variant_id("item-1", "canonical", "tmpl", "some text")
    id_c = make_variant_id("item-1", "canonical", "tmpl", "different text")
    assert id_a == id_b
    assert id_a != id_c
