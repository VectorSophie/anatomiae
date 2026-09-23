"""Prompt-construction schema: DatasetItem -> PromptVariant -> RenderedPrompt.

Kept separate from anatomiae.datasets.schema.DatasetItem: perturbations
are a transform layer applied on top of an item, not baked into it
(HELM-scenario-inspired split - see docs/FRAMEWORK_DECISION.md).

Base-model and instruction-model rendering are kept structurally distinct
(`ModelRenderMode`) because a base model reads a raw completion-style
string while an instruct model needs a chat template applied - collapsing
these silently would misrepresent what the model actually saw.
"""

from __future__ import annotations

import hashlib
from typing import Literal

from pydantic import BaseModel, ConfigDict, model_validator

PerturbationType = Literal["canonical", "paraphrase", "framing", "inversion"]
SemanticEquivalenceStatus = Literal["equivalent", "inverted", "unknown"]
ModelRenderMode = Literal["base_completion", "chat_instruct"]

# Perturbation types whose semantic-equivalence status is unambiguous from
# the type alone; "framing" is deliberately absent - framing changes can be
# equivalence-preserving or not depending on the specific framing, so it
# must be stated explicitly per variant rather than inferred.
INFERRED_EQUIVALENCE: dict[PerturbationType, SemanticEquivalenceStatus] = {
    "canonical": "equivalent",
    "paraphrase": "equivalent",
    "inversion": "inverted",
}


class PromptVariant(BaseModel):
    """One perturbation condition applied to a DatasetItem, still
    model-agnostic (no chat template / base-completion formatting yet)."""

    model_config = ConfigDict(frozen=True)

    variant_id: str
    item_id: str
    language: str
    perturbation_type: PerturbationType
    semantic_equivalence_status: SemanticEquivalenceStatus
    template_id: str
    variant_text: str


class RenderedPrompt(BaseModel):
    """A PromptVariant rendered into the exact text a specific model
    receives. `rendered_prompt_hash` is the generation-cache key component
    (see anatomiae.provenance.generation_cache) - it must be recomputed
    from `rendered_text`, never trusted as freestanding metadata."""

    model_config = ConfigDict(frozen=True)

    variant_id: str
    item_id: str
    language: str
    render_mode: ModelRenderMode
    template_id: str
    rendered_text: str
    rendered_prompt_hash: str

    @staticmethod
    def compute_hash(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    @model_validator(mode="after")
    def _hash_matches_text(self) -> RenderedPrompt:
        expected = self.compute_hash(self.rendered_text)
        if self.rendered_prompt_hash != expected:
            raise ValueError(
                "rendered_prompt_hash does not match rendered_text - construct "
                "RenderedPrompt via render_prompt(), not directly."
            )
        return self
