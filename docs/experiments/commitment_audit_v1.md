# Protocol: human audit of the commitment indicator C — v1

**Status:** sample frozen 2026-10-14. Labels pending: this needs human
annotators.

## Why

In `selective_commitment_confirmatory_v1`, the commitment indicator `C`
comes only from J-qy. On the deliberately difficult 150-row validation
sample, J-qy forced a stance on 19 of 26 responses humans judged mixed or
neutral (`eqy_v1.md`). The confirmed selection gap compares stance-first
direction *within* J-qy's commitment threshold.

That gap could be an evaluator artefact. It would be one if J-qy's
commitment errors were themselves related to the stance-first pole: for
example, over-reading hedged prose mainly on items whose explicit stance is
left-pole.

This audit measures `C`'s validity on a **representative** sample of the
confirmatory free-form responses.

## Sample, frozen

- **Population:** the 2,640 free-form responses of the OLMo confirmatory
  run (`artifacts/cache/selective_commitment_v1.jsonl`): SFT, DPO and
  RLVR2 × 5 free-form prompts × 88 items × 2 polarities.
- **Draw:** 10 responses uniformly at random from each of the 15 stage ×
  free-form-prompt cells, 150 in total, with `random_state=20261014`. Every
  cell holds 176 responses, so equal allocation is proportional, and the
  unweighted sample represents the free-form population.
- **Generator:** `scripts/make_commitment_audit_sheet.py`. It never
  overwrites an existing key.

## Annotation

The instructions and taxonomy are identical to human validation v1
(`artifacts/labeling/README.md`):
- `relation`: supports | opposes | mixed_or_conditional |
  neutral_or_no_position | unclear;
- `mode`;
- `confidence_1to3`;
- `notes`.

Annotators see only the statement and the response. They do not see the
stage, the prompt, J-qy's label, the stance-first response or the item
coding.

- **Sheets:** `artifacts/labeling/commitment_audit_v1/sheet_A.csv` and
  `sheet_C.csv`. They contain the same rows in independent orders.
- **Key:** `key.csv` holds the stage, prompt, item, polarity, J-qy outcome
  and stance-first pole. It stays local-only (gitignored) until both sheets
  are complete.

Preferred: two independent annotators (A and C, as in v1), then
adjudication. Minimum: one annotator on all 150, and a second on at least
50.

## Analysis, fixed now

`C_human = 1` iff relation ∈ {supports, opposes}. `unclear` counts as 0 and
is also reported separately.

1. **Agreement of `C` (J-qy) with `C_human`.** Accuracy, Cohen κ, and
   positive and negative agreement, each with a 95% Wilson CI:
   - **over-read:** `C = 1` and `C_human = 0`, among `C_human = 0`;
   - **under-read:** `C = 0` and `C_human = 1`, among `C_human = 1`.
2. **Direction agreement.** Among responses that both call directional, the
   rate of supports ↔ opposes reversal.
3. **The audit's key question: does J-qy's commitment error depend on the
   stance-first pole?** This is checked through the stance-first pole of
   the response's item-polarity (from the confirmatory run, in the key):
   - over-read and under-read rates by pole (left / right /
     non-directional), with Wilson CIs and the difference;
   - `C_human` vs `C` commitment rate by pole.

   The selection gap would be an evaluator artefact only if J-qy
   over-reads more on left-pole items or under-reads more on right-pole
   ones. With about 150 rows this is a coarse check, and it is reported as
   such. It is not a re-estimate of the gap.
4. Results by stage and by prompt, descriptive.

**Results document:** `docs/results/commitment_audit_v1.md`, written once
labels exist.
