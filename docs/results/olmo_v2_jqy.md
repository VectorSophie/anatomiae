# OLMo v2 stage study re-read by J-qy (post-hoc sensitivity analysis)

**Status:** complete (2026-10-13). Measurement layer. No OLMo response was
regenerated.

**Protocol:** `docs/experiments/olmo_v2_jqy_protocol.md`, frozen at
`b00469e`.

**What this is and is not.** This is a post-hoc sensitivity analysis, not
an independent confirmation:

- J-qy (`judge_qwen25_14b_pair_v1`: Qwen2.5-14B-Instruct, shown the
  statement) was chosen *after* the 150-row human validation, because it
  agreed with humans best (κ 0.75, 0 reversals among 117
  human-directional responses).
- J-qy over-reads: it forces a stance on 19 of 26 human non-directional
  responses, and E_q_y v1 rejected the hypothesis that showing the
  statement removes over-read.
- The validation sample is not representative of this study.
- Direction indices describe measured response–statement relations under
  Faulborn's left/right item coding. They are not model ideology.

## Run record

- **Coverage:** 3,696 v2 responses. 3,137 non-empty responses were judged;
  559 empty ones were scored `malformed`, as every evaluator does.
- **Identical inputs:** 37 groups of identical (statement, response) pairs,
  81 responses in all, share one judgement each. That gives 3,093 unique
  judge requests.
- **Judge outputs:**
  - 0 errors;
  - 1 unparseable output (`malformed`, kept);
  - 73 hit the 16-token limit, 72 of them after a valid leading label.
- **Deviation D1:** the first judge run stopped while appending a duplicate
  request. 14 records were kept and the remainder was generated in a second
  batch.
- **Batch/determinism check:** on the 150 validation rows, the full-cache
  run reproduces E_q_y v1's outcome **150/150** and its raw judge text
  149/150.
- **GPU:** GPU 1 only, 250 W preflight, driver 610.57.04.

## Generated tables

| Table | Contents |
|---|---|
| `artifacts/tables/olmo_v2_jqy_metrics.csv` | every stage × render × prompt × evaluator: direction index, acquiescence, coverage |
| `olmo_v2_jqy_questions.{csv,md}` | questions A–C with item-bootstrap CIs, all evaluators, all items and class-A items |
| `olmo_v2_jqy_consistency.csv` | orig/inv consistency |
| `olmo_v2_jqy_transitions.csv` | response-level J-qy vs each old evaluator |
| `olmo_v2_jqy_overlap.csv` | batch/determinism check |
| `olmo_v2_jqy_exploratory_selection.csv` | exploratory selection check |

Figure: `figures/olmo_v2_jqy_direction_coverage.{png,pdf}`.

## Coverage first (question D)

J-qy, chat render. Rates are shares of all 176 responses per cell.

| stage | prompt | directional rate | mixed | neutral | direction index (directional n) |
|---|---|---:|---:|---:|---:|
| SFT | stance_first | 0.97 | 0.03 | 0.00 | 0.53 (171) |
| SFT | please_respond | 0.44 | 0.30 | 0.27 | 0.79 (77) |
| SFT | opinion | **0.20** | 0.14 | 0.66 | 0.78 (36) |
| DPO | stance_first | 0.94 | 0.06 | 0.00 | 0.61 (165) |
| DPO | please_respond | 0.45 | 0.30 | 0.25 | 0.87 (79) |
| DPO | opinion | 0.41 | 0.37 | 0.22 | 0.89 (73) |
| RLVR2 | stance_first | 0.96 | 0.04 | 0.00 | 0.55 (169) |
| RLVR2 | please_respond | 0.42 | 0.32 | 0.26 | 0.87 (74) |
| RLVR2 | opinion | 0.42 | 0.39 | 0.19 | 0.89 (74) |

Under the released prompts, J-qy reads only 20–45% of chat responses as
taking a side. Every released-prompt direction index below describes that
committed subset only.

**Base and raw-render cells.** These are in the metrics table but are not
interpreted. J-qy labels 22–28% of Base `please_respond` and `stance_first`
responses `unclear`. SFT given raw text is mostly empty: `other_rate` is
0.43–0.80.

## Question A — SFT → DPO, chat

| prompt | J-qy Δ [95% CI] | directional n, SFT → DPO | reconstructed classifiers (6) | zero-shot | rule-based |
|---|---|---|---|---|---|
| stance_first | **+0.07 [0.00, 0.14]** | 171 → 165 | +0.02 to +0.07 | +0.02 | +0.11 [0.01, 0.22] |
| please_respond | +0.08 [−0.04, 0.20] | 77 → 79 | +0.13 to +0.40 (2/6 CIs exclude 0) | −0.07 | undefined |
| opinion | +0.11 [−0.09, 0.35] | 36 → 73 | +0.09 to +0.38 (4/6) | +0.17 | undefined |

**Released Δ minus stance-first Δ** (joint item bootstrap):
- J-qy: +0.01 [−0.13, 0.14] for `please_respond` and +0.04 [−0.18, 0.28]
  for `opinion`;
- reconstructed classifiers: +0.07 to +0.36, with 1 of 6 CIs excluding 0
  for each prompt.

**Under the frozen reading rule, the large SFT → DPO shift seen under the
released prompts is not detected under J-qy.**
- J-qy's released Δ (+0.08, +0.11) is about the same size as the
  explicit-stance Δ (+0.07).
- The difference between them is near 0.

**Where the old shift came from.** The reconstructed classifiers' larger Δ
comes almost entirely from **SFT**. There they read a much lower
released-prompt direction (0.36–0.65 for seed 42) than J-qy (0.78). At DPO
and RLVR2, J-qy and the reconstructed classifiers converge near 0.7–0.9 (figure). The
rule-based evaluator has almost no directional responses there.

**Class-A items only:**
- `please_respond` +0.09 [0.00, 0.22];
- `opinion` −0.11 [−0.37, 0.00];
- `stance_first` +0.04 [−0.04, 0.13].

These cells are near the ceiling of the index, so several bootstrap bounds
sit exactly at 0. Nothing here supports a larger released-prompt shift.

## Question B — released vs stance-first within stage, chat

| stage | J-qy please_respond − SF | J-qy opinion − SF | reconstructed (6): CIs excluding 0 | zero-shot |
|---|---|---|---|---|
| SFT | **+0.25 [0.11, 0.40]** | **+0.25 [0.02, 0.45]** | −0.14 to +0.19, 0/6 and 0/6 | −0.16, −0.33 |
| DPO | **+0.26 [0.15, 0.38]** | **+0.28 [0.14, 0.44]** | +0.12 to +0.29, 3/6 and 5/6 | −0.25, −0.18 |
| RLVR2 | **+0.31 [0.16, 0.46]** | **+0.34 [0.21, 0.47]** | +0.15 to +0.41, 5/6 and 6/6 | −0.16, −0.15 |

Under J-qy, the released-prompt direction exceeds the stance-first direction
at **every** post-trained stage, with every CI excluding 0. That includes
SFT, where the reconstructed classifiers did not show it, and where
`olmo_stages_v2.md` §3 said "for SFT it does not".

The zero-shot classifier gives the opposite sign everywhere. It labels
nearly every response directional, including hedged ones.

**Class-A items:** +0.20 to +0.36 at all three stages. 5 of 6 CIs
exclude 0; DPO `opinion` is at +0.20 [−0.02, 0.41].

## Question C — DPO → RLVR2, chat

| prompt | J-qy Δ [95% CI] | old evaluators |
|---|---|---|
| stance_first | −0.06 [−0.11, −0.00] | −0.08 to +0.02; 0/8 CIs exclude 0 |
| please_respond | −0.00 [−0.10, +0.10] | −0.07 to +0.19; 2/7 defined CIs exclude 0 |
| opinion | +0.00 [−0.12, +0.14] | −0.06 to +0.08; 0/8 |

**Under the frozen rule, "remains small" is NOT formally met.**

- The J-qy stance-first CI just excludes 0 (upper bound −0.001), and its
  lower bound (−0.112) lies outside ±0.10.
- The released CIs include 0, but they reach slightly past ±0.10:
  - `please_respond` [−0.097, +0.102];
  - `opinion` [−0.121, +0.138].

What the data do show:
- every J-qy point estimate is between −0.06 and +0.00;
- the explicit-stance estimate is a small *decrease*, matching the
  rule-based evaluator's −0.08 [−0.17, 0.00];
- the DiD is +0.05 [−0.07, 0.17] and +0.06 [−0.09, 0.23].

So DPO → RLVR2 shows no large change. It is not established as "within
±0.10" at the frozen CI criterion, and it may include a small decrease in
explicit-stance direction.

## Consistency (faithful class-A pairs, chat, stance-first)

Consistency is the share of item pairs where the responses to a statement
and to its opposite take opposite stances.

| stage | J-qy | reconstructed as-written s42 | rule-based |
|---|---|---|---|
| SFT | 0.73 (n=45) | 0.74 (n=46) | 0.75 (n=40) |
| DPO | 0.79 (n=43) | 0.80 (n=45) | 0.89 (n=28) |
| RLVR2 | 0.73 (n=44) | 0.73 (n=45) | 0.92 (n=26) |

Same picture as the original study: about a quarter of faithful
statement/opposite pairs get the same stance from the model. Under
released prompts J-qy's consistency is 0.92–1.00, but only on 7–16 pairs
per cell.

## Where J-qy changes the old evaluators' reading of the same response

Chat render, 528 responses per prompt (3 stages × 176). Generated from
`olmo_v2_jqy_transitions.csv`.

**Old evaluator vs J-qy, released prompts:**

| old evaluator | polarity change (agree ↔ disagree) | J-qy commits, old does not | old commits, J-qy does not |
|---|---|---|---|
| reconstructed classifiers (6) | 2–17 (≤ 3.2%) | 43–83 | 13–24 |
| zero-shot | 16 (opinion), 32 (please_respond) | 2–3 | **298–345** |
| rule-based | 0 | 173–211 | 1 |

**Old evaluator vs J-qy, stance-first:** polarity changes are 0–5 of 528,
and responses read as the same direction are 414–505.

So J-qy almost never **reverses** an old evaluator's direction. It
disagrees about **commitment**:
- more committal than the rule-based evaluator and the reconstructed
  classifiers;
- far less committal than zero-shot.

The old evaluators' conclusions therefore differ from J-qy mainly through
*which responses enter the direction index*, not through opposite
readings.

## Exploratory, not in the frozen protocol: the released > stance-first gap is a selection effect

This analysis was added after seeing question B (table
`olmo_v2_jqy_exploratory_selection.csv`). It asks whether the gap arises
because the released prompts select which (item, polarity) pairs the model
commits on. For each stage, it compares:
- the stance-first direction on the pairs J-qy read as directional under
  the released prompt;
- the stance-first direction on all other pairs.

| stage, prompt | released direction (committed n) | stance-first on those same pairs | stance-first on the rest | committed − rest [95% CI] |
|---|---|---|---|---|
| SFT, please_respond | 0.79 (77) | 0.76 | 0.35 | +0.42 [0.16, 0.66] |
| SFT, opinion | 0.78 (36) | 0.89 | 0.44 | +0.44 [0.21, 0.64] |
| DPO, please_respond | 0.87 (79) | 0.82 | 0.40 | +0.42 [0.21, 0.63] |
| DPO, opinion | 0.89 (73) | 0.90 | 0.36 | +0.53 [0.31, 0.75] |
| RLVR2, please_respond | 0.87 (74) | 0.82 | 0.31 | +0.51 [0.25, 0.77] |
| RLVR2, opinion | 0.89 (74) | 0.89 | 0.26 | +0.63 [0.41, 0.84] |

On the item-polarities where the model commits freely, its explicitly
stated direction is about as high as the released-prompt direction J-qy
reads. The gap (0.53–0.61 stance-first overall vs 0.78–0.89 released)
comes from the items the model **does not** commit to when free. Asked
directly, it states a much more balanced pattern on those.

The reconstructed seed-42 classifiers show the same pattern at DPO and
RLVR2, with committed − rest of +0.42 to +0.53.

Under J-qy, at least, the released-prompt "excess direction" is not the
evaluator reading direction into neutral prose. The model **expresses a
position spontaneously mainly where its stated position is one-sided**, and
a direction index computed over committed responses inherits that selection.

## Revision of the working thesis

The handoff thesis said: *post-training changes expression and measurability
more than explicit direction, and response-only evaluators amplify apparent
directional shifts.*

Against J-qy, part by part:

1. **Kept.** Post-training changes response mechanics much more than
   explicit direction. Under J-qy, the stance-first direction moves SFT
   +0.53 → DPO +0.61 → RLVR2 +0.55.
2. **Kept, narrowed.** The large SFT → DPO released-prompt shift is
   evaluator-dependent. It is not detected under J-qy, and J-qy's
   released-prompt shift is about the size of the explicit-stance shift.
   The old shift is concentrated in the reconstructed classifiers' lower
   reading of SFT prose.
3. **Revised.** "Prose-inferred direction exceeds stated direction because
   of how direction is measured" is too simple:
   - under J-qy the gap exists at every post-trained stage, including
     SFT;
   - the exploratory check locates it mainly in **selective commitment**,
     an Expression effect: which items the model takes a side on without
     being asked;
   - that effect interacts with a Measurement choice: indexing direction
     only over committed responses.

   Evaluator misreading of prose is a smaller part on this evidence. One
   evaluator family is the exception: zero-shot forces stances on hedged
   prose and *lowers* the released index.
4. **Revised.** DPO → RLVR2 is "no large change", not "demonstrably within
   ±0.10". The explicit-stance estimate is a small decrease.

The revised working thesis:

> Post-training changes *whether and where* a model commits to a position
> far more than the direction of the positions it states when asked. A
> free-form direction index mixes stated direction with *selective
> commitment*. How much of that mixture an evaluator reports depends on its
> commitment threshold, more than on polarity errors.

## Limitations

- **Post hoc.** The evaluator was selected after validation. The
  selection-effect analysis was added after seeing B.
- **J-qy over-reads.** On the 150-row validation sample J-qy commits more
  often than humans on hedged responses (over-read 19/26). Its released
  directional rates (0.20–0.45) are therefore likely upper bounds on what
  humans would call committed. The selection analysis does not depend on
  where that threshold sits, but its magnitudes might.
- **Judge provenance.** One judge model, one prompt, greedy decoding. The
  judge belongs to a different lineage from OLMo, but it is still a model.
- **Statement quality.** 13 inversions are not true opposites. Class-A-only
  results are reported and agree in direction.
- **Item coding.** Faulborn's left/right coding defines the axis. Q36's
  coding is questionable (`olmo_stages_v2.md` §5).
