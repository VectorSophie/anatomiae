import pytest

pytest.importorskip("pandas")  # analysis/frame.py needs the `ml` extra; CI runs `dev` only

from anatomiae.analysis.frame import (
    build_analysis_frame,
    export_table,
    outcome_rate,
    stratify_by_completeness,
)
from anatomiae.evaluators.base import EvaluationResult
from anatomiae.inference.schema import (
    DecodingConfig,
    GenerationRecord,
    GenerationRequest,
)


def _record(
    prompt_hash: str,
    model_id: str = "test/model",
    backend: str = "transformers",
    finish_reason: str = "stop",
):
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
        finish_reason=finish_reason,
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


def test_truncated_and_complete_flags_derived_from_finish_reason():
    r_complete = _record("a", finish_reason="stop")
    r_truncated = _record("b", finish_reason="length")
    r_errored = _record("c", finish_reason="error")
    df = build_analysis_frame(
        [r_complete, r_truncated, r_errored],
        [
            _eval(r_complete, "agreement"),
            _eval(r_truncated, "neutral_or_no_position"),
            _eval(r_errored, "generation_error"),
        ],
    )
    row = lambda key: df[df["cache_key"] == key].iloc[0]
    assert row(r_complete.cache_key)["response_complete"] == True
    assert row(r_complete.cache_key)["truncated"] == False
    assert row(r_truncated.cache_key)["truncated"] == True
    assert row(r_truncated.cache_key)["response_complete"] == False
    assert row(r_errored.cache_key)["truncated"] == False
    assert row(r_errored.cache_key)["response_complete"] == False


def test_truncation_never_silently_becomes_neutral():
    """A truncated response classified as neutral_or_no_position must be
    distinguishable from a complete response classified the same way -
    the whole point of this column existing (see docs/results/
    faulborn_reproduction.md's real finding)."""
    r_truncated_neutral = _record("a", finish_reason="length")
    r_complete_neutral = _record("b", finish_reason="stop")
    df = build_analysis_frame(
        [r_truncated_neutral, r_complete_neutral],
        [
            _eval(r_truncated_neutral, "neutral_or_no_position"),
            _eval(r_complete_neutral, "neutral_or_no_position"),
        ],
    )
    truncated_row = df[df["cache_key"] == r_truncated_neutral.cache_key].iloc[0]
    complete_row = df[df["cache_key"] == r_complete_neutral.cache_key].iloc[0]
    assert truncated_row["truncated"] == True
    assert complete_row["truncated"] == False
    # both have outcome=neutral_or_no_position, but only one carries the
    # truncation caveat - the raw label alone would conflate them
    assert truncated_row["outcome"] == complete_row["outcome"] == "neutral_or_no_position"


def test_stance_observed_before_truncation_flag():
    r_truncated_with_stance = _record("a", finish_reason="length")
    r_truncated_no_stance = _record("b", finish_reason="length")
    r_complete = _record("c", finish_reason="stop")
    df = build_analysis_frame(
        [r_truncated_with_stance, r_truncated_no_stance, r_complete],
        [
            _eval(r_truncated_with_stance, "agreement"),  # position outcome, but truncated
            _eval(r_truncated_no_stance, "neutral_or_no_position"),
            _eval(r_complete, "agreement"),
        ],
    )
    row = lambda key: df[df["cache_key"] == key].iloc[0]
    assert row(r_truncated_with_stance.cache_key)["stance_observed_before_truncation"] == True
    assert row(r_truncated_no_stance.cache_key)["stance_observed_before_truncation"] == False
    # not truncated - the flag is meaningless here, must be None not False
    assert row(r_complete.cache_key)["stance_observed_before_truncation"] is None


def test_stratify_by_completeness_never_drops_truncated_rows():
    r_complete = _record("a", finish_reason="stop")
    r_truncated = _record("b", finish_reason="length")
    df = build_analysis_frame(
        [r_complete, r_truncated],
        [_eval(r_complete, "agreement"), _eval(r_truncated, "neutral_or_no_position")],
    )
    strata = stratify_by_completeness(df)
    assert set(strata) == {"all", "completed_only", "truncated_only"}
    assert strata["all"]["n"].sum() == 2
    assert strata["completed_only"]["n"].sum() == 1
    assert strata["truncated_only"]["n"].sum() == 1
    assert strata["truncated_only"]["outcome"].iloc[0] == "neutral_or_no_position"


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
