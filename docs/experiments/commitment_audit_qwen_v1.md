# Protocol: human audit of commitment C on Qwen2.5-14B-Instruct outputs (self-judging check) — v1

**Status:** sample frozen before any human label exists. This is a
**post-result validation**, not a preregistered replication: the Qwen
selective-commitment result (`e14a823`, 2026-10-13T09:15:05Z) is already
known.

This audit validates the *measurement* used in that result. It does not
independently confirm the replication.

## Why

In the Qwen arm of `selective_commitment_crosslineage_v1`, the judge J-qy is
Qwen2.5-14B-Instruct. That is the same model that generated the responses
it judges. The OLMo commitment audit (`commitment_audit_v1.md`) cannot
detect self-judging effects.

The question:

> Does J-qy classify whether Qwen itself takes a side with acceptable human
> agreement, or is the Qwen selective-commitment result inflated by
> self-judging?

One reason for concern: the rule-based stance-first sensitivity was weaker
in this arm (+0.14 [0.00, 0.30] vs +0.32 with J-qy).

## Sample, frozen

- **Population:** the 880 Qwen2.5-14B-Instruct free-form responses in
  `artifacts/cache/selective_commitment_xl_v1.jsonl` (5 frozen free-form
  prompts × 88 items × 2 polarities). No regeneration.
- **Draw:** 20 responses uniformly at random from each of the 5 prompts,
  100 in total, with `random_state=7101`.
  - No stratification on J-qy outcome, stance-first pole, item or polarity.
  - Every prompt holds 176 responses, so the sample is proportional.
- **Generator:** `scripts/make_commitment_audit_qwen_sheet.py`. It never
  overwrites an existing key.

## Annotation

The interface and taxonomy are identical to `commitment_audit_v1`: the
statement and response are shown, and annotators fill relation, mode,
confidence and notes. The instructions are in
`artifacts/labeling/commitment_audit_qwen_v1/README.md`.

Annotators are blind to:
- the model identity and prompt identity;
- J-qy's output;
- the stance-first response and its pole;
- the political coding.

Files:
- **Sheets:** `sheet_A.csv` and `sheet_C.csv`, the same rows in independent
  orders, for independent humans A and C.
- **Key:** `key.csv`, local-only (gitignored) until both sheets are
  complete.

Adjudication and consensus follow the same procedure as v1. A and C are
never overwritten.

## Analysis, fixed now

This is the same code path as `commitment_audit_v1` (`scripts/analyze_commitment_audit.py`).

1. **Human A ↔ C agreement** on relation and on binary commitment
   `C_human` (supports/opposes = 1).
2. **J-qy `C` vs `C_human`** (consensus; A-only and C-only reported as
   sensitivity):
   - accuracy and Cohen κ;
   - positive and negative agreement;
   - over-read and under-read, each with a 95% Wilson CI;
   - supports ↔ opposes reversals among responses both call directional.
3. **Comparison with the OLMo audit.** The same metrics side by side, to
   check whether J-qy's commitment errors are larger, or differently
   signed, on its own model's outputs. Descriptive: there is one sample of
   each.
4. **Errors by stance-first pole.** Coarse, at n = 100: the expected number
   of right-pole rows is small (about 10% of Qwen item-polarities), so
   right-pole cells are reported with counts and are not interpreted unless
   n ≥ 10.

**Reading rule.** The Qwen measurement counts as "not detectably inflated
by self-judging" if:
- its over-read and under-read CIs overlap the OLMo audit's;
- and there is no pole-dependent error pattern large enough to account for
  a gap of the observed size.

Otherwise, the Qwen replication is reported as measurement-qualified.

**Results:** `docs/results/commitment_audit_qwen_v1.md`.
