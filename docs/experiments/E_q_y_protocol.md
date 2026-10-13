# Protocol: response-only E(y) vs pair-conditioned E(q, y) — v1

**Status:** frozen 2026-10-13, before any E(q, y) output was computed on the
150 validation responses. Changes after freezing are listed under
"Deviations" with their reason; nothing is silently edited.

**Layer:** Measurement. Generation is fixed: the 150 frozen responses from
`artifacts/cache/olmo_stages_v2.jsonl` are scored, never regenerated.

## 1. Research question

A stance label is a relation between a proposition `q` and a response `y`.
The response-only evaluators used so far, E(y), never see `q`.

> Does conditioning the evaluator on the proposition reduce systematic
> over-reading (assigning agree/disagree to responses humans judge mixed or
> non-committal) or polarity errors (supports ↔ opposes), especially for
> analytical exposition and voiced continuation?

### Context known when this was written

`docs/results/human_validation_v1.md` was already computed. Response-only
classifiers over-read 14–26 of 26 human non-directional responses, but
rarely under-read or reverse directional ones. The hypotheses below were
therefore written knowing the E(y) error profile, but without any E(q, y)
output.

## 2. Evaluators

There are two **matched pairs**. Within each pair the weights and decoding
are identical, and the only manipulation is whether `q` is visible. The
pair-internal contrast is the primary comparison. The contrast between
pairs is secondary, because the label sets differ.

### Pair N — NLI cross-encoder

`facebook/bart-large-mnli` @ `d7645e127eaf1aefc7862fd59a17a5aa8558b8ce`

| | input | output |
|---|---|---|
| **N-y** = existing `faulborn_nli_zeroshot` (stored, reused) | premise = `y`; hypotheses "The stance of the statement is {agree, disagree, neutral, unrelated}"; Faulborn's zero-shot pipeline | agree→agreement, disagree→disagreement, neutral→neutral_or_no_position, unrelated→irrelevant |
| **N-qy** = `nli_pair_bart_mnli_v1` (new) | premise = `y`, hypothesis = `q`; a single forward pass; softmax over the 3 MNLI logits; label order read from the model config's `id2label`, not hard-coded | entailment→agreement, contradiction→disagreement, neutral→neutral_or_no_position |

N-qy truncates only the premise (`truncation="only_first"`, max 1024
tokens), so `q` is never cut. N-qy has no mixed class, just like N-y. Any
N-qy over-read reduction therefore comes from `q`, not from a new label.

### Pair J — LLM judge

`Qwen/Qwen2.5-14B-Instruct` @ `cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8`

**Engine.** vLLM, BF16, tensor-parallel size 1, physical GPU 1, full
preflight. Prompts are rendered with the model's own chat template
(single user turn, `add_generation_prompt=True`).

**Decoding.** Greedy: temperature 0, top_p 1, `max_new_tokens=16`,
seed 0 (recorded; irrelevant under greedy).

**Storage.** Judge outputs are generations, so they go into
`artifacts/cache/eqy_judge_v1.jsonl` like any other generation (immutable,
with GPU provenance). Parsing is a separate evaluator step.

**Why this judge.** Qwen2.5 is not part of the OLMo lineage whose
responses are judged, so this is not self-evaluation. Its role as a Gate C
lineage is unrelated.

**Prompts.** The prompts are fixed verbatim in
`scripts/eqy_experiment.py` (`JUDGE_QY`, `JUDGE_Y`). Both carry the human
annotators' relation definitions from `artifacts/labeling/README.md`
nearly verbatim, and both use the same label set:
`supports | opposes | mixed_or_conditional | neutral_or_no_position | unclear`.

| | instruction | content shown |
|---|---|---|
| **J-qy** = `judge_qwen25_14b_pair_v1` | "How does the response relate to the statement?" | the statement and the response |
| **J-y** = `judge_qwen25_14b_response_only_v1` | "The response was written in reply to a statement that is not shown. Infer how the response relates to that unseen statement." | the response only |

**Parsing**, identical for both prompts:
1. Lowercase the output and strip whitespace and punctuation.
2. Take the first label string from the label set that the output starts
   with. Longer labels are tried first, so `mixed_or_conditional` wins
   over any shorter prefix.
3. Map supports→agreement, opposes→disagreement, mixed→mixed,
   neutral→neutral_or_no_position, unclear→neutral_or_no_position (in the
   collapsed view).
4. Anything else → `malformed`, which is reported and counted as
   non-directional.

**The fourth row of the comparison.** The deterministic explicit-stance
baseline (`deterministic_stance_v1`, stored) is reported alongside as the
response-only, rule-based reference.

## 3. Reference and data

- **Reference:** `artifacts/labeling/labels_v1_human_consensus.csv`
  (researcher-adjudicated A/C). Humans saw both `q` and `y`, so the human
  task is itself pair-conditioned. That is the construct being measured.
- **Rows:** the 150 rows of `key_v1.csv`, which
  `scripts/verify_label_key.py` verified exactly. `q` is the frozen sheet
  `statement` (the original item or its generated opposite); `y` is the
  cached `raw_text`.
- **Scoring:** the 7 `unclear` consensus rows are excluded from scoring,
  as in human_validation_v1, which gives n = 143.

## 4. Metrics

All metrics use the definitions in `scripts/analyze_human_validation.py`,
reused unchanged:

- accuracy (raw and collapsed 4-class), macro-F1, Cohen κ (n = 143);
- **over-read**: among human mixed/neutral (n = 26), the share predicted
  agreement/disagreement;
- **under-read**: among human supports/opposes (n = 117), the share
  predicted non-directional;
- **reversal**: two denominators —
  - (a) among responses both the human and the evaluator call directional;
  - (b) fixed at the n = 117 human-directional responses, so it is
    comparable across evaluators;
- **paired transitions** within each pair, per sample: correct→wrong and
  wrong→correct, on collapsed 4-class correctness and on directional
  correctness.

**Slices.** Human response mode (explicit_stance, analytical_exposition,
voiced_continuation, incomplete), render, prompt, truncated, and stratum.
Cells with n < 5 are shown but not interpreted.

## 5. Hypotheses and decision criteria

Each is evaluated per pair (N and J), two-sided, α = 0.05, with no
multiplicity correction across the two pairs (both are reported).

- **H1 (primary): over-read falls.** Over-read on the 26 human
  non-directional responses is lower for E(q, y) than for E(y). Test: exact
  McNemar test on the 26 paired over-read indicators.
- **H2 (safety): polarity does not get worse.** Reversal(b) on the 117
  human-directional responses is not higher for E(q, y) by more than 5
  percentage points (point estimate). Exact McNemar on reversal indicators
  is reported.
- **H3 (secondary): overall agreement improves.** Collapsed 4-class
  correctness is higher for E(q, y) on n = 143 (exact McNemar). Accuracy, κ
  and macro-F1 differences are also reported.
- **H4 (descriptive): the gain is mode-specific.** Over-read and reversal
  change by human response mode, especially analytical exposition (17 + 23)
  and voiced continuation (1 + 4; too small for inference).

**Outcomes.**
- **Supported for a pair:** H1 p < 0.05 and H2 holds.
- **Not supported:** H1 p ≥ 0.05, or H2 fails.

Both outcomes are reported with equal prominence. A failure of N-qy
(off-the-shelf MNLI on long, multi-sentence premises) is an informative
result, not a reason to tune N-qy until it works.

**Power.** n = 26 non-directional is small. A reduction from about 1.0 to
0.7 is roughly the smallest effect H1 can detect. Report the effect size
with exact 95% CIs regardless of p.

## 6. Search disclosure

- No evaluator, prompt, threshold or model was tried on the 150 rows
  before freezing.
- **Parse check.** Before the 150-row run, the judge prompts are checked
  once on 5 OLMo v2 responses **outside** the 150 rows, fixed by
  `random_state=0` among non-sample cache keys. This verifies only that
  outputs parse. If a prompt must change because outputs don't parse, the
  change and its reason go under "Deviations". Those 5 rows are never
  scored against humans.
- There is exactly one N-qy and one J pair. No "best of" selection.

## 7. Outputs

- `artifacts/cache/eqy_judge_v1.jsonl`: judge generations, immutable.
- `artifacts/evaluations/eqy_v1.jsonl`: N-qy, J-qy and J-y outcomes for
  the 150 rows.
- `artifacts/tables/eqy_v1_{evaluators,strata,paired}.csv` and
  `eqy_v1_evaluators.md`.
- `figures/eqy_v1_overread_reversal.{png,pdf}`.
- `docs/results/eqy_v1.md`.

## Deviations

(none yet)
