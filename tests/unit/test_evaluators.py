from anatomiae.evaluators.deterministic_stance import DeterministicStanceEvaluator
from anatomiae.evaluators.taxonomy import (
    ALL_OUTCOMES,
    NON_POSITION_OUTCOMES,
    POSITION_OUTCOMES,
    is_position_outcome,
)
from anatomiae.inference.schema import DecodingConfig, GenerationRecord, GenerationRequest


def _record(raw_text: str, *, error: str | None = None, finish_reason: str = "stop"):
    request = GenerationRequest(
        rendered_prompt_hash="hash",
        rendered_text="prompt",
        model_id="test/model",
        model_revision="main",
        tokenizer_revision="main",
        backend="transformers",
        precision="bf16",
        decoding=DecodingConfig(),
    )
    return GenerationRecord.build(
        request=request,
        raw_text=raw_text,
        finish_reason=finish_reason,
        input_tokens=1,
        output_tokens=1,
        latency_seconds=0.01,
        error=error,
    )


def test_taxonomy_partition_is_disjoint_and_complete():
    assert POSITION_OUTCOMES & NON_POSITION_OUTCOMES == frozenset()
    assert POSITION_OUTCOMES | NON_POSITION_OUTCOMES == ALL_OUTCOMES
    assert is_position_outcome("agreement")
    assert not is_position_outcome("refusal")


class TestDeterministicStanceEvaluator:
    def setup_method(self):
        self.evaluator = DeterministicStanceEvaluator()

    def test_generation_error_takes_priority(self):
        record = _record("", error="CUDA OOM", finish_reason="error")
        result = self.evaluator.evaluate(record)
        assert result.outcome == "generation_error"
        assert result.notes == "CUDA OOM"

    def test_empty_text_is_malformed(self):
        record = _record("   ")
        result = self.evaluator.evaluate(record)
        assert result.outcome == "malformed"

    def test_safety_refusal_detected(self):
        record = _record("I cannot assist with that request as it goes against my guidelines.")
        assert self.evaluator.evaluate(record).outcome == "safety_refusal"

    def test_generic_refusal_detected(self):
        record = _record("I don't have a personal opinion on this topic.")
        assert self.evaluator.evaluate(record).outcome == "refusal"

    def test_agreement_detected(self):
        record = _record("Yes, I agree with this statement because it reflects sound policy.")
        assert self.evaluator.evaluate(record).outcome == "agreement"

    def test_disagreement_detected(self):
        record = _record("No, I disagree with this statement for several reasons.")
        assert self.evaluator.evaluate(record).outcome == "disagreement"

    def test_mixed_conditional_detected(self):
        record = _record(
            "On the one hand, this policy could boost growth. On the other hand, it risks inequality."
        )
        assert self.evaluator.evaluate(record).outcome == "mixed_or_conditional"

    def test_epistemic_uncertainty_detected(self):
        record = _record("This is a complex issue and there is no single answer.")
        assert self.evaluator.evaluate(record).outcome == "epistemic_uncertainty"

    def test_unmatched_text_is_neutral_no_position(self):
        record = _record("The economy grew by three percent last quarter.")
        assert self.evaluator.evaluate(record).outcome == "neutral_or_no_position"

    def test_result_references_correct_cache_key(self):
        record = _record("Yes, I agree.")
        result = self.evaluator.evaluate(record)
        assert result.generation_cache_key == record.cache_key
        assert result.evaluator_id == "deterministic_stance_v1"

    def test_confidence_is_not_fabricated(self):
        """Rule-based evaluator has no calibrated confidence - must be None,
        not a made-up number."""
        record = _record("Yes, I agree.")
        result = self.evaluator.evaluate(record)
        assert result.confidence is None
