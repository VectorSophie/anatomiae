"""CPU-safe tests of FaulbornNLIEvaluator's routing and label mapping.

The real __init__ loads a BART model (and runs the GPU preflight for
device="cuda"); these tests build a bare instance via object.__new__ and
stub `classify_texts`, so they check routing/mapping logic without any
model weights or hardware.
"""

from __future__ import annotations

from anatomiae.evaluators.faulborn_nli import (
    FAULBORN_LABELS,
    FAULBORN_TO_OUTCOME,
    FaulbornNLIEvaluator,
)
from anatomiae.evaluators.taxonomy import ALL_OUTCOMES
from anatomiae.inference.schema import DecodingConfig, GenerationRecord, GenerationRequest


def _record(raw_text: str, prompt_hash: str, *, finish_reason="stop", error=None):
    request = GenerationRequest(
        rendered_prompt_hash=prompt_hash,
        rendered_text="prompt",
        model_id="m",
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
        latency_seconds=0.1,
        error=error,
    )


def _bare_evaluator(fake_predictions):
    ev = object.__new__(FaulbornNLIEvaluator)
    ev.evaluator_id = "faulborn_nli_test"
    calls = []

    def fake_classify(texts):
        calls.append(list(texts))
        return [fake_predictions[t] for t in texts]

    ev.classify_texts = fake_classify
    return ev, calls


def test_every_faulborn_label_maps_into_shared_taxonomy():
    assert set(FAULBORN_TO_OUTCOME) == set(FAULBORN_LABELS)
    assert set(FAULBORN_TO_OUTCOME.values()) <= ALL_OUTCOMES
    # refusal is never produced by this classifier's label set - it cannot
    # be silently mapped onto a political position here
    assert "refusal" not in FAULBORN_TO_OUTCOME.values()


def test_native_label_and_confidence_preserved():
    ev, _ = _bare_evaluator({"I agree.": ("agree", 0.97)})
    (result,) = ev.evaluate_many([_record("I agree.", "h1")])
    assert result.outcome == "agreement"
    assert result.native_label == "agree"
    assert result.confidence == 0.97


def test_errors_and_empty_text_never_reach_the_classifier():
    ev, calls = _bare_evaluator({"Fine text.": ("neutral", 0.6)})
    records = [
        _record("", "h1", finish_reason="error", error="OOM"),
        _record("   ", "h2"),
        _record("Fine text.", "h3"),
    ]
    results = ev.evaluate_many(records)
    assert [r.outcome for r in results] == ["generation_error", "malformed", "neutral_or_no_position"]
    assert calls == [["Fine text."]]  # only the valid text was classified


def test_results_align_with_input_order():
    preds = {"a": ("agree", 0.9), "b": ("disagree", 0.8), "c": ("unrelated", 0.7)}
    ev, _ = _bare_evaluator(preds)
    records = [_record(t, f"h-{t}") for t in ["a", "b", "c"]]
    results = ev.evaluate_many(records)
    assert [r.generation_cache_key for r in results] == [r.cache_key for r in records]
    assert [r.native_label for r in results] == ["agree", "disagree", "unrelated"]
