"""Build PromptVariant / RenderedPrompt records from a DatasetItem.

Paraphrase/framing/inversion *text* is supplied by the caller (either
pre-curated alongside the source dataset, as Faulborn's released dataset
does, or produced by a separate, explicitly-provenanced generation step
later) - this module does not itself invent perturbed wording, since
silently generating "a paraphrase" without recording how it was produced
would be exactly the kind of missing provenance this project exists to
avoid.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable

from anatomiae.datasets.schema import DatasetItem
from anatomiae.prompts.schema import (
    INFERRED_EQUIVALENCE,
    ModelRenderMode,
    PerturbationType,
    PromptVariant,
    RenderedPrompt,
    SemanticEquivalenceStatus,
)


def make_variant_id(item_id: str, perturbation_type: str, template_id: str, variant_text: str) -> str:
    """Deterministic, content-addressed variant id - the same
    (item, perturbation, template, text) always yields the same id, so
    reruns don't mint duplicate variants for identical content."""
    digest = hashlib.sha256(
        f"{item_id}|{perturbation_type}|{template_id}|{variant_text}".encode()
    ).hexdigest()[:16]
    return f"{item_id}::{perturbation_type}::{digest}"


def build_variant(
    item: DatasetItem,
    *,
    perturbation_type: PerturbationType,
    variant_text: str,
    template_id: str,
    semantic_equivalence_status: SemanticEquivalenceStatus | None = None,
) -> PromptVariant:
    """Construct a PromptVariant for one perturbation condition of `item`.

    `semantic_equivalence_status` is inferred from `perturbation_type` for
    canonical/paraphrase/inversion (unambiguous by construction); it is
    mandatory for "framing" since framing may or may not preserve meaning
    depending on the specific framing applied.
    """
    if semantic_equivalence_status is None:
        if perturbation_type not in INFERRED_EQUIVALENCE:
            raise ValueError(
                f"semantic_equivalence_status must be given explicitly for "
                f"perturbation_type={perturbation_type!r} - it cannot be inferred."
            )
        semantic_equivalence_status = INFERRED_EQUIVALENCE[perturbation_type]

    return PromptVariant(
        variant_id=make_variant_id(item.item_id, perturbation_type, template_id, variant_text),
        item_id=item.item_id,
        language=item.language,
        perturbation_type=perturbation_type,
        semantic_equivalence_status=semantic_equivalence_status,
        template_id=template_id,
        variant_text=variant_text,
    )


def canonical_variant(item: DatasetItem, *, template_id: str = "canonical") -> PromptVariant:
    """The item's own original prompt text, unperturbed."""
    return build_variant(
        item,
        perturbation_type="canonical",
        variant_text=item.prompt_original,
        template_id=template_id,
    )


def render_prompt(
    variant: PromptVariant,
    *,
    render_mode: ModelRenderMode,
    chat_render_fn: Callable[[str], str] | None = None,
) -> RenderedPrompt:
    """Render a PromptVariant into the exact text a model receives.

    For `render_mode="chat_instruct"`, `chat_render_fn` must be supplied
    (normally a thin wrapper around `tokenizer.apply_chat_template`) - this
    module deliberately does not import a tokenizer itself, so unit tests
    can inject a fake renderer without loading a real model.
    """
    if render_mode == "chat_instruct":
        if chat_render_fn is None:
            raise ValueError("chat_render_fn is required for render_mode='chat_instruct'")
        text = chat_render_fn(variant.variant_text)
    else:
        text = variant.variant_text

    return RenderedPrompt(
        variant_id=variant.variant_id,
        item_id=variant.item_id,
        language=variant.language,
        render_mode=render_mode,
        template_id=variant.template_id,
        rendered_text=text,
        rendered_prompt_hash=RenderedPrompt.compute_hash(text),
    )
