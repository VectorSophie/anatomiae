from anatomiae.evaluators.base import EvaluationResult
from anatomiae.provenance.evaluation_store import EvaluationStore


def _result(key="g1", evaluator="det", version="1.0", outcome="agreement"):
    return EvaluationResult(
        generation_cache_key=key, evaluator_id=evaluator, evaluator_version=version, outcome=outcome
    )


def test_append_is_idempotent(tmp_path):
    store = EvaluationStore(tmp_path / "e.jsonl")
    assert store.append_many([_result()]) == 1
    assert store.append_many([_result()]) == 0  # same (generation, evaluator, version) -> skipped
    assert len(store.read_all()) == 1


def test_new_evaluator_version_stored_alongside_not_over(tmp_path):
    store = EvaluationStore(tmp_path / "e.jsonl")
    store.append_many([_result(version="1.0", outcome="agreement")])
    store.append_many([_result(version="1.1", outcome="disagreement")])
    stored = {(r.evaluator_version, r.outcome) for r in store.read_all()}
    assert stored == {("1.0", "agreement"), ("1.1", "disagreement")}


def test_fresh_instance_sees_prior_writes(tmp_path):
    path = tmp_path / "e.jsonl"
    EvaluationStore(path).append_many([_result(evaluator="nli")])
    assert EvaluationStore(path).has("g1", "nli", "1.0")


def test_read_for_filters_by_evaluator(tmp_path):
    store = EvaluationStore(tmp_path / "e.jsonl")
    store.append_many([_result(evaluator="a"), _result(evaluator="b")])
    assert [r.evaluator_id for r in store.read_for({"b"})] == ["b"]
