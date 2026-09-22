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

TranslationOrigin = Literal[
    "native",  # written directly in `language`, no translation involved
    "human_translation",
    "machine_translation",
    "independently_authored",  # separately sourced/written per language, not a translation pair
    "unknown",
]

TranslatorType = Literal["human_expert", "human_nonexpert", "machine", "paper_author", "unknown"]


class DatasetItem(BaseModel):
    """Canonical dataset item.

    Bilingual/multilingual provenance is mandatory, not optional metadata -
    added after discovering that a locked reference (Lim & Roettger) pairs
    languages via human translation from a single-language source media,
    not independent per-language authorship. Downstream analysis must be
    able to tell "two cultures independently said different things" apart
    from "one thing was translated and compared to itself" - collapsing
    that distinction would misattribute translation artifacts to genuine
    cross-lingual/cross-cultural variation. See docs/DATASET_AUDIT.md.
    """

    model_config = ConfigDict(frozen=True)

    item_id: str
    source_dataset: str
    source_version: str
    issue: str
    dimension: str
    country_context: str | None = None

    language: str
    source_language: str | None = None
    translation_origin: TranslationOrigin
    translation_method: str | None = None
    translator_type: TranslatorType | None = None
    independently_authored: bool
    parallel_item_id: str | None = None
    translation_notes: str | None = None

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
