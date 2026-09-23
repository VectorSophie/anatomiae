import pytest

from anatomiae.inference.schema import DecodingConfig, GenerationRecord, GenerationRequest
from anatomiae.provenance.generation_cache import DuplicateGenerationError, GenerationCache


def _record(prompt_hash: str = "abc"):
    request = GenerationRequest(
        rendered_prompt_hash=prompt_hash,
        rendered_text="text",
        model_id="test/model",
        model_revision="main",
        tokenizer_revision="main",
        backend="transformers",
        precision="bf16",
        decoding=DecodingConfig(),
    )
    return GenerationRecord.build(
        request=request,
        raw_text="response",
        finish_reason="stop",
        input_tokens=1,
        output_tokens=1,
        latency_seconds=0.01,
    )


def test_empty_cache_has_nothing(tmp_path):
    cache = GenerationCache(tmp_path / "raw.jsonl")
    assert not cache.has("anything")
    assert len(cache) == 0
    assert cache.read_all() == []


def test_append_then_has(tmp_path):
    cache = GenerationCache(tmp_path / "raw.jsonl")
    record = _record()
    cache.append(record)
    assert cache.has(record.cache_key)
    assert len(cache) == 1


def test_duplicate_append_without_force_raises(tmp_path):
    cache = GenerationCache(tmp_path / "raw.jsonl")
    record = _record()
    cache.append(record)
    with pytest.raises(DuplicateGenerationError):
        cache.append(record)
    # the file must still only have one line - the rejected write never happened
    assert len(cache.path.read_text().strip().splitlines()) == 1


def test_duplicate_append_with_force_appends_a_second_line(tmp_path):
    cache = GenerationCache(tmp_path / "raw.jsonl")
    record = _record()
    cache.append(record)
    cache.append(record, force=True)
    assert len(cache.path.read_text().strip().splitlines()) == 2


def test_resume_across_cache_instances_sees_prior_writes(tmp_path):
    """A fresh GenerationCache object pointed at the same file (simulating
    a process restart / resumed run) must see what a prior instance wrote,
    not just its own in-memory state."""
    path = tmp_path / "raw.jsonl"
    cache1 = GenerationCache(path)
    record = _record()
    cache1.append(record)

    cache2 = GenerationCache(path)
    assert cache2.has(record.cache_key)


def test_read_all_returns_equivalent_records(tmp_path):
    cache = GenerationCache(tmp_path / "raw.jsonl")
    r1, r2 = _record("aaa"), _record("bbb")
    cache.append(r1)
    cache.append(r2)
    restored = cache.read_all()
    assert {r.cache_key for r in restored} == {r1.cache_key, r2.cache_key}
