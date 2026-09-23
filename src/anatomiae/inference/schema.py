"""Canonical generation request/record schema.

Deliberately has no torch/transformers/vllm import - this module is pure
and fully unit-testable without a GPU. The actual backends
(anatomiae.inference.backends) depend on this schema, not the other way
around.

Generation identity (`GenerationRequest.cache_key()`) determines whether a
generation is "the same" for caching/resume purposes. Per the project's
immutable-generation-cache requirement, this must include everything that
can change the output: model + tokenizer revision, the exact rendered
prompt (via its hash), backend, precision, and the full decoding config
(including seed) - two requests that differ in any of these are different
cache entries, never silently merged.
"""

from __future__ import annotations

import hashlib
import json
import time
from typing import Literal

from pydantic import BaseModel, ConfigDict

Backend = Literal["transformers", "vllm"]
Precision = Literal["fp32", "bf16", "fp16"]
FinishReason = Literal["stop", "length", "error"]


class DecodingConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    temperature: float = 0.0
    top_p: float = 1.0
    max_new_tokens: int = 200
    seed: int | None = 0


class GenerationRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    rendered_prompt_hash: str
    rendered_text: str
    model_id: str
    model_revision: str
    tokenizer_revision: str
    backend: Backend
    precision: Precision
    decoding: DecodingConfig

    # Provenance links back to the dataset item / prompt variant that
    # produced this prompt. Deliberately NOT part of cache_key(): identical
    # rendered text under identical generation settings is the same
    # generation regardless of which item it came from. Optional because
    # records written before these fields existed don't have them (they can
    # be recovered deterministically by re-rendering and matching
    # rendered_prompt_hash).
    item_id: str | None = None
    variant_id: str | None = None
    template_id: str | None = None

    def cache_key(self) -> str:
        """Content-addressed generation identity. Two requests with the
        same key are, by the project's own definition, "the same
        generation" and must not be re-run unless explicitly forced."""
        payload = {
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "tokenizer_revision": self.tokenizer_revision,
            "rendered_prompt_hash": self.rendered_prompt_hash,
            "backend": self.backend,
            "precision": self.precision,
            "decoding": self.decoding.model_dump(),
        }
        blob = json.dumps(payload, sort_keys=True)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class GPURecordProvenance(BaseModel):
    model_config = ConfigDict(frozen=True)

    physical_index: int
    uuid: str
    driver_version: str
    power_limit_w: float


class GenerationRecord(BaseModel):
    """Immutable raw generation record. Once written to the generation
    cache, a record is never mutated or overwritten - a re-run with the
    same cache key is a no-op unless explicitly forced (see
    anatomiae.provenance.generation_cache)."""

    model_config = ConfigDict(frozen=True)

    cache_key: str
    request: GenerationRequest
    raw_text: str
    finish_reason: FinishReason
    input_tokens: int
    output_tokens: int
    latency_seconds: float
    gpu: GPURecordProvenance | None = None
    error: str | None = None
    timestamp: float
    # >1 when generated in one batched engine call (latency_seconds is then the batch
    # wall time / batch_size). Batching can change numerics; not part of the cache key.
    batch_size: int | None = None

    @classmethod
    def build(
        cls,
        *,
        request: GenerationRequest,
        raw_text: str,
        finish_reason: FinishReason,
        input_tokens: int,
        output_tokens: int,
        latency_seconds: float,
        gpu: GPURecordProvenance | None = None,
        error: str | None = None,
        batch_size: int | None = None,
    ) -> GenerationRecord:
        return cls(
            cache_key=request.cache_key(),
            request=request,
            raw_text=raw_text,
            finish_reason=finish_reason,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_seconds=latency_seconds,
            gpu=gpu,
            error=error,
            timestamp=time.time(),
            batch_size=batch_size,
        )
