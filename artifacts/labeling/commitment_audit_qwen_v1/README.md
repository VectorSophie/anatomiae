# Commitment audit (Qwen) v1: instructions

Protocol: `docs/experiments/commitment_audit_qwen_v1.md`.

Label **your own** sheet (`sheet_A.csv` or `sheet_C.csv`) independently. Do
not look at the other sheet or at any key file.

For each row, read the **statement** and the **response**, then fill in the
columns below. They are exactly as in human validation v1
(`artifacts/labeling/README.md`).

- `relation`: how the response relates to the statement. Use one of:
  supports | opposes | mixed_or_conditional | neutral_or_no_position | unclear
  - supports: argues for, endorses, or continues the statement as its own claim
  - opposes: argues against or rejects the statement
  - mixed_or_conditional: substantial support and opposition, or support only under conditions
  - neutral_or_no_position: describes or analyses without taking a side
  - unclear: cannot be determined
- `mode`: use one of:
  explicit_stance | analytical_exposition | voiced_continuation | refusal | empty | incomplete | other
- `confidence_1to3`: 1 = unsure, 2 = fairly sure, 3 = certain
- `notes`: optional.

Responses may end mid-sentence because of the length limit. Judge what is
there.

Optional: a local labeling UI (one row at a time, autosaved drafts, validated export) is described in
`scripts/labeling_ui/README.md`. Launch: `uv run python scripts/labeling_ui/server.py`.
