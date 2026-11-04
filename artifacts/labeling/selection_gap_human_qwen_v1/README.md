# Follow-up labeling (Audit 3): instructions

Protocol: `docs/experiments/selective_commitment_human_reestimate_qwen_v1.md`.

There are 142 new rows, `QG001`–`QG142`. The same rows appear in `sheet_A.csv` and `sheet_C.csv`, in
independent orders. Label **only your own** sheet, independently. Do not look at the other
sheet, at any key or at earlier audits. The taxonomy and instructions are exactly those of
the two commitment audits (`artifacts/labeling/commitment_audit_qwen_v1/README.md`).

Easiest: open your own offline page, which needs no internet and no server.
- Annotator A: `artifacts/labeling/standalone/selection_gap_qwen_annotator_A.html`.
- Annotator C: `artifacts/labeling/standalone/selection_gap_qwen_annotator_C.html`. It is in
  French and shows each statement in French translation, with the English original
  collapsible. The responses stay in English.

Export the completed CSV from the page (`qwen_gap_sheet_<X>.completed.csv`) and send it to
the researcher. The researcher then runs:

```bash
uv run python scripts/labeling_ui/verify_export.py qwen_gap A path/to/qwen_gap_sheet_A.completed.csv
uv run python scripts/analyze_commitment_audit.py import-sheet --audit qwen_gap --annotator A path/to/qwen_gap_sheet_A.completed.csv
# same for C; then:
uv run python scripts/analyze_commitment_audit.py check --audit qwen_gap        # no key access; writes adjudication_queue.csv
uv run python scripts/analyze_commitment_audit.py import-adjudication --audit qwen_gap path/to/completed_queue.csv
uv run python scripts/analyze_commitment_audit.py consensus --audit qwen_gap
uv run python scripts/selection_gap_human_qwen_v1.py analyze                    # needs the Qwen audit consensus as well
```

Files:
- `manifest.json`: public counts and sheet hashes only.
- `statements_fr.csv`: the display-only French translations.
  - 14/33 are reused verbatim from the earlier human-C French rerun
    (`reused_from_human_C_retry150_FR_completed`; source
    `prior_C_translation_reuse_reference.csv`, SHA-256 `71e646a4…`).
  - 19/33 were drafted, then reviewed and corrected by the researcher on 2026-11-04
    (`researcher_reviewed_2026-11-04`). The file was saved with a UTF-8 BOM; readers use
    `utf-8-sig`.

  Translation reuse is not annotation reuse: no label from the earlier run is used for any
  follow-up row.
- `key.csv`: local-only and gitignored.
