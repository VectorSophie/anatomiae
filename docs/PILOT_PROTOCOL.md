# Pilot protocol

Status: gates defined, none yet passed. This document specifies what must
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

Do not "improve" the method before proving it can be reproduced. Results
and discrepancies go in `docs/results/faulborn_reproduction.md` (not yet
created).

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
| A — Faulborn reproduction | Not started |
| B — Backend equivalence | Not started (one informal smoke comparison exists) |
| C — 3 lineages end-to-end | Not started (1 model, Transformers+vLLM, smoke-tested only) |

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
