"""Evaluator interface.

`Evaluator.evaluate()` consumes an already-generated, cached
GenerationRecord and nothing else - it must never trigger regeneration,
must never see generation-time state (backend internals, sampler state),
and multiple evaluators must be able to score the same cached record
independently (project spec §39/§57: no single evaluator, including an
LLM judge, is ground truth).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from pydantic import BaseModel, ConfigDict

from anatomiae.evaluators.taxonomy import Outcome
from anatomiae.inference.schema import GenerationRecord


class EvaluationResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    generation_cache_key: str
    evaluator_id: str
    evaluator_version: str
    outcome: Outcome
    confidence: float | None = None
    notes: str | None = None


class Evaluator(ABC):
    evaluator_id: str
    evaluator_version: str

    @abstractmethod
    def evaluate(self, record: GenerationRecord) -> EvaluationResult:
        """Score one cached GenerationRecord. Must not regenerate, must not
        mutate `record`."""
        ...

    def evaluate_many(self, records: list[GenerationRecord]) -> list[EvaluationResult]:
        return [self.evaluate(r) for r in records]
