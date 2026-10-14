"""Check a completed CSV (from either labeling UI) against its frozen blind sheet.

  uv run python scripts/labeling_ui/verify_export.py <olmo|qwen> <A|C> path/to/completed.csv

Read-only. Checks exact columns, row count, sample_ids and order, statement and
response text byte-for-byte, completeness and taxonomy values. Prints the cp
command to submit it; it never copies or writes anything itself.
"""

from __future__ import annotations

import sys
from pathlib import Path

import server as ui


def verify(audit: str, annotator: str, path: Path) -> list[str]:
    source = ui.read_sheet(ui.SHEETS[(audit, annotator)])
    try:
        done = ui.read_sheet(path)
    except ValueError as e:
        return [str(e)]
    problems = []
    if len(done) != len(source):
        problems.append(f"row count {len(done)} != {len(source)}")
    for a, b in zip(source, done, strict=False):
        for col in ("sample_id", "statement", "response"):
            if a[col] != b[col]:
                problems.append(f"row {a['sample_id']}: {col} differs from the frozen sheet")
    labels = {r["sample_id"]: {c: r[c] for c in ui.LABEL_COLUMNS} for r in done}
    if len(labels) != len(done):
        problems.append("duplicate sample_ids")
    p, _ = ui.validate_export(source, labels)
    return problems + p


def main() -> None:
    if len(sys.argv) != 4 or (sys.argv[1], sys.argv[2]) not in ui.SHEETS:
        raise SystemExit(__doc__)
    audit, annotator, path = sys.argv[1], sys.argv[2], Path(sys.argv[3])
    problems = verify(audit, annotator, path)
    if problems:
        print(f"INVALID: {len(problems)} problem(s)")
        print("\n".join(problems[:50]))
        raise SystemExit(1)
    print(f"VALID. To submit (explicit, manual): cp {path} {ui.display(ui.SHEETS[(audit, annotator)])}")


if __name__ == "__main__":
    main()
