"""Canonical dataset item schema (project spec §42).

Every item entering the pipeline - regardless of source benchmark - is
normalized to this shape before prompt construction. Perturbations
(paraphrase, framing, translation) are applied as a separate transform
layer downstream (see docs/FRAMEWORK_DECISION.md's HELM-scenario-inspired
split) and are tracked via PerturbationRecord, not baked into this item.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict


class DatasetItem(BaseModel):
    model_config = ConfigDict(frozen=True)

    item_id: str
    source_dataset: str
    source_version: str
    issue: str
    dimension: str
    country_context: str | None = None
    language: str
    prompt_original: str
    reference_position: str | None = None
    reference_distribution: dict[str, float] | None = None
    metadata: dict = {}


class PerturbationRecord(BaseModel):
    """Provenance for a transform applied to a DatasetItem's prompt."""

    model_config = ConfigDict(frozen=True)

    base_item_id: str
    perturbation_type: Literal["canonical", "paraphrase", "framing", "inversion", "translation"]
    semantic_equivalence_status: Literal["equivalent", "inverted", "unknown"]
    translation_provenance: str | None = None
    rendered_prompt: str
