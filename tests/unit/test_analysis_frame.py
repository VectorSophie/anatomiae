import pytest

pytest.importorskip("pandas")  # analysis/frame.py needs the `ml` extra; CI runs `dev` only

from anatomiae.analysis.frame import build_analysis_frame, export_table, outcome_rate
from anatomiae.evaluators.base import EvaluationResult
from anatomiae.inference.schema import (
    DecodingConfig,
    GenerationRecord,
    GenerationRequest,
)


def _record(prompt_hash: str, model_id: str = "test/model", backend: str = "transformers"):
    request = GenerationRequest(
        rendered_prompt_hash=prompt_hash,
        rendered_text="prompt",
        model_id=model_id,
        model_revision="main",
        tokenizer_revision="main",
        backend=backend,
        precision="bf16",
        decoding=DecodingConfig(seed=0),
    )
    return GenerationRecord.build(
        request=request,
        raw_text="response",
        finish_reason="stop",
        input_tokens=5,
        output_tokens=3,
        latency_seconds=0.5,
    )


def _eval(record: GenerationRecord, outcome: str, evaluator_id: str = "eval-a"):
    return EvaluationResult(
        generation_cache_key=record.cache_key,
        evaluator_id=evaluator_id,
        evaluator_version="1.0",
        outcome=outcome,
    )


def test_build_analysis_frame_is_long_format_one_row_per_eval():
    r1, r2 = _record("aaa"), _record("bbb")
    evals = [_eval(r1, "agreement"), _eval(r2, "refusal"), _eval(r1, "disagreement", "eval-b")]
    df = build_analysis_frame([r1, r2], evals)
    assert len(df) == 3  # two evaluators on r1 -> two rows, not two columns
    assert set(df["outcome"]) == {"agreement", "refusal", "disagreement"}


def test_build_analysis_frame_carries_gpu_provenance():
    from anatomiae.inference.schema import GPURecordProvenance

    request = GenerationRequest(
        rendered_prompt_hash="ccc",
        rendered_text="prompt",
        model_id="test/model",
        model_revision="main",
        tokenizer_revision="main",
        backend="transformers",
        precision="bf16",
        decoding=DecodingConfig(),
    )
    record = GenerationRecord.build(
        request=request,
        raw_text="response",
        finish_reason="stop",
        input_tokens=1,
        output_tokens=1,
        latency_seconds=0.1,
        gpu=GPURecordProvenance(physical_index=1, uuid="GPU-x", driver_version="570", power_limit_w=250.0),
    )
    df = build_analysis_frame([record], [_eval(record, "agreement")])
    assert df.iloc[0]["gpu_physical_index"] == 1
    assert df.iloc[0]["gpu_power_limit_w"] == 250.0


def test_dangling_evaluation_reference_raises():
    r1 = _record("aaa")
    fake_eval = EvaluationResult(
        generation_cache_key="does-not-exist",
        evaluator_id="eval-a",
        evaluator_version="1.0",
        outcome="agreement",
    )
    with pytest.raises(ValueError, match="dangling"):
        build_analysis_frame([r1], [fake_eval])


def test_outcome_rate_denominator_is_full_group():
    r1, r2, r3 = _record("a"), _record("b"), _record("c")
    evals = [_eval(r1, "agreement"), _eval(r2, "refusal"), _eval(r3, "refusal")]
    df = build_analysis_frame([r1, r2, r3], evals)
    rates = outcome_rate(df)
    refusal_row = rates[rates["outcome"] == "refusal"].iloc[0]
    assert refusal_row["n"] == 2
    assert refusal_row["total"] == 3
    assert refusal_row["rate"] == pytest.approx(2 / 3)


def test_outcome_rate_grouped_by_model():
    r1 = _record("a", model_id="model-x")
    r2 = _record("b", model_id="model-y")
    evals = [_eval(r1, "agreement"), _eval(r2, "agreement")]
    df = build_analysis_frame([r1, r2], evals)
    rates = outcome_rate(df, group_by=["model_id"])
    assert len(rates) == 2
    assert set(rates["total"]) == {1}


def test_export_table_writes_all_requested_formats(tmp_path):
    r1 = _record("a")
    df = build_analysis_frame([r1], [_eval(r1, "agreement")])
    written = export_table(df, tmp_path / "results", formats=("csv", "parquet", "md"))
    assert written["csv"].exists()
    assert written["parquet"].exists()
    assert written["md"].exists()
    assert "agreement" in written["md"].read_text()

    import pandas as pd

    reloaded = pd.read_parquet(written["parquet"])
    assert len(reloaded) == 1
