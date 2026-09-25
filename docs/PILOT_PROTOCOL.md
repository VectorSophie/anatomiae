# Pilot protocol

Status: Gates A1–A3, B, C and the precision check done (see table below). This document specifies what must
be true before the pilot scales from smoke tests to the full factorial
design (§14 of the project's continuation directive).

## Validation gates

The pilot does not expand to the full factorial design until all three
gates pass. Each gate is a real, checkable artifact — not a status claim.

### Gate A — Faulborn reproduction

Reproduce a small real slice of Faulborn et al.'s released methodology
closely enough to establish that `anatomiae`'s adapter interprets it
correctly, in this order:

1. published methodology → `anatomiae` reproduction → comparison →
   documented deviations;
2. only then: published methodology → identified limitation →
   `anatomiae` extension.

Do not "improve" the method before proving it can be reproduced.

Gate A is split into three explicit sub-gates, because "the items run
through our pipeline" and "we reproduced their measurement" are different
claims:

- **A1 — Real Faulborn items/prefixes through the anatomiae pipeline.**
  Real released items and prefix strings, real generations, cached,
  evaluated, tabulated. A methodological reproduction of the item/prefix
  design - *not* of their model set or their classifier.
  → `docs/results/faulborn_reproduction.md`
- **A2 — Reproduce Faulborn's released stance-classifier path.** Their
  exact inference procedure (BART-MNLI zero-shot pipeline, their template
  and labels, response-only premise, argmax + confidence), validated
  against their own released test split and reported metrics.
  → `docs/results/faulborn_classifier_reproduction.md`
- **A3 — Compare elicitation and evaluator dependence.** Same cached
  outputs scored by multiple evaluators (no regeneration); then a
  clearly-separate stance-first elicitation *extension* compared against
  the original prompt conditions. Only after A2 passes or its blocker is
  documented. → `docs/results/faulborn_evaluator_agreement.md`

### Gate B — Backend measurement equivalence

Formalize the Transformers-vs-vLLM difference already observed
informally (`artifacts/logs/backend_divergence_result.json`) into a small
controlled study: ~20-50 prompts per available model family, deterministic
decoding where valid, measuring prompt/tokenizer equality, exact-output
agreement, completion length, termination behavior, and — the actual
question — whether backend differences change the *dependent variables*
(stance, refusal, evaluator score), not just the surface text. Results go
in `docs/results/backend_equivalence.md` (not yet created).

### Gate C — Three independent model lineages end-to-end

At least three distinct model lineages must successfully traverse the
full pipeline (dataset → prompt → generation → raw output → parsing →
evaluator → result table → analysis artifact) before the full pilot
expands. This is a mechanism check, not a results check: it proves the
pipeline works across genuinely different model families/tokenizers/chat
templates, not just the one model already smoke-tested.

## Current state against the gates

| Gate | Status |
|---|---|
| A1 — Faulborn items/prefixes through pipeline | **Passed** (15 items x 3 prefixes x 2 budgets; truncation-stratified) |
| A2 — Faulborn classifier path | **Validated reconstruction; exact released-weights reproduction blocked** (released checkpoint has no weights file). Reconstruction following the released data flow reproduces reported F1 (0.875 vs 0.873); leakage-free reconstruction scores ~0.12 lower — `docs/results/faulborn_classifier_reproduction.md` |
| A3 — Elicitation / evaluator dependence | **Done on the A1 slice + 30-item stance-first extension** — `docs/results/faulborn_evaluator_agreement.md` |
| Precision sensitivity (OLMo-13B Base FP32 vs BF16) | **Done** — BF16 standardized, precision kept as a sensitivity floor — `docs/results/precision_sensitivity.md` |
| B — Backend equivalence | **Done (measured, not assumed zero)** — identical inputs; 18–72% byte-identical outputs; 0–10% outcome change for non-zero-shot evaluators — `docs/results/backend_equivalence.md` |
| C — 3 lineages end-to-end | **Passed, 3 of 3** — OLMo 2 (13B-SFT), Qwen2.5-14B-Instruct, Amber (final, `2d35937a`): verify (SHA-256 of every weight file checked against the Hub) → smoke → preflight → 60 generations → 8 evaluators → analysis row, 0 errors each (`artifacts/tables/gate_c_lineages.*`). Amber, a base model, degenerates into repeating the statement under greedy decoding: 100% truncated, 0 positions |

## Pilot factorial design (target, once gates pass)

Per the original research spec: ~100 canonical political items (primary
bilingual pilot: Lim & Röttger's EN/ZH set) × 2 languages × 3 prompt
conditions (canonical, paraphrase, framing) × model states across the
locked-anchor lineages, plus a Faulborn reproduction subset and an
OpinionQA population-distribution subset.

This section will be filled in with exact commands once
`src/anatomiae/cli/` exists and the gates above have passed - no
speculative "expected results" here (see the project's own rule against
fabricated placeholder scientific graphs).
