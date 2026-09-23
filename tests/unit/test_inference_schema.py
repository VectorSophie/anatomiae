from anatomiae.inference.schema import DecodingConfig, GenerationRecord, GenerationRequest


def _request(**overrides):
    defaults = {
        "rendered_prompt_hash": "abc123",
        "rendered_text": "Do you agree with this statement?",
        "model_id": "allenai/OLMo-2-0425-1B-Instruct",
        "model_revision": "main",
        "tokenizer_revision": "main",
        "backend": "transformers",
        "precision": "bf16",
        "decoding": DecodingConfig(),
    }
    defaults.update(overrides)
    return GenerationRequest(**defaults)


def test_cache_key_is_deterministic():
    r1 = _request()
    r2 = _request()
    assert r1.cache_key() == r2.cache_key()


def test_cache_key_changes_with_backend():
    r1 = _request(backend="transformers")
    r2 = _request(backend="vllm")
    assert r1.cache_key() != r2.cache_key()


def test_cache_key_changes_with_precision():
    r1 = _request(precision="bf16")
    r2 = _request(precision="fp32")
    assert r1.cache_key() != r2.cache_key()


def test_cache_key_changes_with_seed():
    r1 = _request(decoding=DecodingConfig(seed=0))
    r2 = _request(decoding=DecodingConfig(seed=1))
    assert r1.cache_key() != r2.cache_key()


def test_cache_key_changes_with_prompt_hash():
    r1 = _request(rendered_prompt_hash="aaa")
    r2 = _request(rendered_prompt_hash="bbb")
    assert r1.cache_key() != r2.cache_key()


def test_cache_key_changes_with_model_revision():
    r1 = _request(model_revision="main")
    r2 = _request(model_revision="ckpt_180")
    assert r1.cache_key() != r2.cache_key()


def test_generation_record_build_populates_cache_key_and_timestamp():
    request = _request()
    record = GenerationRecord.build(
        request=request,
        raw_text="I agree with this statement.",
        finish_reason="stop",
        input_tokens=12,
        output_tokens=6,
        latency_seconds=0.42,
    )
    assert record.cache_key == request.cache_key()
    assert record.timestamp > 0
    assert record.error is None


def test_generation_record_roundtrips_through_json():
    request = _request()
    record = GenerationRecord.build(
        request=request,
        raw_text="I agree.",
        finish_reason="stop",
        input_tokens=5,
        output_tokens=3,
        latency_seconds=0.1,
    )
    blob = record.model_dump_json()
    restored = GenerationRecord.model_validate_json(blob)
    assert restored == record
