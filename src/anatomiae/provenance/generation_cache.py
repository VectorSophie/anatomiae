"""Append-only, resumable immutable generation cache.

Raw generations are the project's primary experimental evidence: never
overwritten, never regenerated merely because a result is inconvenient or
an evaluator changed (see docs/architecture.md's pipeline separation). This
module is the concrete enforcement of that rule for the JSONL raw-record
store: `has(key)` lets a runner skip completed work on resume, and
`append()` refuses to write a second record under a key that already
exists unless `force=True` is passed explicitly.
"""

from __future__ import annotations

import json
from pathlib import Path

from anatomiae.inference.schema import GenerationRecord


class DuplicateGenerationError(RuntimeError):
    """Raised when appending a cache_key that already exists without force=True."""


class GenerationCache:
    def __init__(self, path: Path | str):
        self.path = Path(path)
        self._index: set[str] | None = None

    def _load_index(self) -> set[str]:
        if self._index is not None:
            return self._index
        keys: set[str] = set()
        if self.path.exists():
            with self.path.open() as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    keys.add(json.loads(line)["cache_key"])
        self._index = keys
        return keys

    def has(self, cache_key: str) -> bool:
        return cache_key in self._load_index()

    def append(self, record: GenerationRecord, *, force: bool = False) -> None:
        keys = self._load_index()
        if record.cache_key in keys and not force:
            raise DuplicateGenerationError(
                f"cache_key {record.cache_key} already exists in {self.path} - "
                "refusing to duplicate a completed generation. Pass force=True "
                "only if you specifically intend to record a second attempt "
                "(e.g. after a documented backend/precision change)."
            )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as f:
            f.write(record.model_dump_json() + "\n")
        keys.add(record.cache_key)

    def read_all(self) -> list[GenerationRecord]:
        if not self.path.exists():
            return []
        records = []
        with self.path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                records.append(GenerationRecord.model_validate_json(line))
        return records

    def __len__(self) -> int:
        return len(self._load_index())
