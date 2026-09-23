"""Append-only store of evaluator results, keyed by
(generation cache_key, evaluator_id, evaluator_version).

Generation is done once and cached immutably; scoring is repeated with many
evaluators. Persisting scores too means every analysis reads the same
evaluator outputs (no silent re-scoring drift between scripts), adding an
evaluator only scores what it hasn't scored yet, and a changed evaluator
must bump its version to be stored alongside - never over - the old one.
"""

from __future__ import annotations

from pathlib import Path

from anatomiae.evaluators.base import EvaluationResult


def _key(r: EvaluationResult) -> tuple[str, str, str]:
    return (r.generation_cache_key, r.evaluator_id, r.evaluator_version)


class EvaluationStore:
    def __init__(self, path: Path | str):
        self.path = Path(path)
        self._index: set[tuple[str, str, str]] | None = None

    def _load_index(self) -> set[tuple[str, str, str]]:
        if self._index is None:
            self._index = {_key(r) for r in self.read_all()}
        return self._index

    def has(self, cache_key: str, evaluator_id: str, evaluator_version: str) -> bool:
        return (cache_key, evaluator_id, evaluator_version) in self._load_index()

    def append_many(self, results: list[EvaluationResult]) -> int:
        """Append results not already stored; returns how many were new.
        Re-appending an existing (generation, evaluator, version) is skipped,
        so re-running a scoring script is idempotent."""
        index = self._load_index()
        new = [r for r in results if _key(r) not in index]
        if new:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a") as f:
                for r in new:
                    f.write(r.model_dump_json() + "\n")
                    index.add(_key(r))
        return len(new)

    def read_all(self) -> list[EvaluationResult]:
        if not self.path.exists():
            return []
        with self.path.open() as f:
            return [EvaluationResult.model_validate_json(line) for line in f if line.strip()]

    def read_for(self, evaluator_ids: set[str] | None = None) -> list[EvaluationResult]:
        return [r for r in self.read_all() if evaluator_ids is None or r.evaluator_id in evaluator_ids]


def results_summary(path: Path | str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for r in EvaluationStore(path).read_all():
        counts[r.evaluator_id] = counts.get(r.evaluator_id, 0) + 1
    return counts

