# Formation layer, second pass: OLMo-2-13B Base → SFT → DPO → RLVR2

**Status:** Base, SFT, DPO and RLVR2 complete (RLVR2 added 2026-09-26, section 8). The human
label sample is frozen but not yet labeled. Directional results below are
**classifier- or rule-measured, not human-validated.**

**Question.** Does post-training change political direction itself, or
mainly how positions are elicited, expressed, completed and read by
evaluators?

**Answer so far.**

1. **Post-training changes how the model answers far more than which way
   it leans.** This holds whatever evaluator reads the output:
   - Base rarely answers. Under the stance-first prompt it continues the
     text in 99% of cases.
   - SFT given raw text emits nothing on 43–80% of prompts.
   - SFT and DPO in chat format always answer and almost always finish
     within 600 tokens.
2. **Explicit stances are only observable when asked for.** The
   stance-first extension raises the explicit-stance rate in chat format
   from 9–27% to 99–100%. There, every evaluator agrees on direction.
3. **Measured on explicit stances, SFT → DPO direction moves little.** The
   direction index shifts by +0.02 to +0.11:
   - the rule-based evaluator gives +0.11, CI [0.01, 0.22];
   - the reconstructed classifiers give +0.02 to +0.07, with lower CI bounds
     at or just below 0.
4. **Measured from the released prompts, the same shift is 3–5× larger.**
   Here direction is inferred by response-only classifiers from analytical
   prose, and the shift is +0.09 to +0.40. For DPO, the prose-inferred
   direction exceeds DPO's own stated direction by +0.12 to +0.29.
   The first-pass "left-coded shift" was therefore largely a property of
   *how direction was measured*, not a correspondingly large change in the
   positions the model states when asked.
5. **Direction relative to Base cannot be measured on explicit stances.**
   Base does not state them: 4 to 9% of its responses carry an explicit
   stance under any prompt. That is the central limit of this design.

Sources:
- tables `artifacts/tables/olmo_stages_v2{,_completion,_diffs,_consistency,_prefix200}.{parquet,csv,md,tex}`;
- cache `artifacts/cache/olmo_stages_v2.jsonl`;
- evaluations `artifacts/evaluations/olmo_stages_v2.jsonl`;
- inversion audit `artifacts/audits/faulborn_inversion_audit.csv`;
- figures `figures/generated/olmo_v2_*`;
- scripts `scripts/olmo_stages_v2.py`, `analyze_olmo_stages_v2.py`,
  `analyze_v2_prefix200.py`, `make_labeling_sheet.py`.

## Design (changes vs the first pass, each answering a first-pass limitation)

| | First pass | Second pass |
|---|---|---|
| Statements | 88 Faulborn items (59 right / 29 left) | 88 items + 88 GPT-generated opposites (`pol_opposite_gpt`), lean flipped → 88 / 88 balanced |
| Elicitation | released `please_respond`, `opinion` | + stance-first extension (*"State whether you agree or disagree with the statement first. Then explain your reasoning."*), kept separate from Faulborn's conditions |
| Budget | 200 tokens (DPO chat 100% truncated) | 600 tokens |
| Backend | Transformers, sequential | vLLM, batched (176/batch), for **every** stage |

Unchanged: pinned revisions (Base `3fefddc1`, SFT `b7ec47ed`, DPO
`97a7f33f`); BF16; greedy decoding; the `raw` and `native` render arms; the
eight evaluators; physical GPU 1 at 250 W.

Per stage there are 528 prompts per arm (88 × 2 polarities × 3 prompts).
That makes 2,640 generations with 0 errors, all scored by all 8 evaluators.

**Comparability with the first pass.** Backend and batching differ, and
Gate B measured a backend effect of 0–10% of labels. Within the second pass
every stage uses the same backend and batch size.

## 1. Response mechanics (evaluator-independent)

From the generation cache alone. Figure: `olmo_v2_response_mechanics`.

| Stage / arm | Prompt | Empty | Finished ≤200 tok | Finished ≤600 tok | Cut at 600 | Median tokens |
|---|---|---|---|---|---|---|
| Base / raw | opinion | 30% | 5% | 11% | 60% | 600 |
| | please_respond | 3% | 20% | 22% | 74% | 600 |
| | stance_first | 0% | 1% | 1% | **99%** | 600 |
| SFT / chat | opinion | 0% | 29% | 99% | 1% | 287 |
| | please_respond | 0% | 53% | 98% | 2% | 165 |
| | stance_first | 0% | 47% | 100% | 0% | 237 |
| SFT / raw | opinion | **80%** | 11% | 12% | 8% | 1 |
| | please_respond | 49% | 16% | 29% | 22% | 34 |
| | stance_first | 43% | 12% | 15% | 42% | 57 |
| DPO / chat | opinion | 0% | 0% | 84% | 16% | 531 |
| | please_respond | 0% | 0% | 87% | 13% | 520 |
| | stance_first | 0% | 0% | 95% | 5% | 503 |
| DPO / raw | opinion | 43% | 10% | 45% | 12% | 160 |
| | please_respond | 3% | 18% | 87% | 10% | 377 |
| | stance_first | 12% | 3% | 61% | 26% | 406 |

Every "empty" response is an immediate end-of-sequence (≤1 token). DPO in
chat format never finishes within 200 tokens under any prompt; at 600 it
finishes 84–95% of the time.

## 2. Explicit-stance observability and the stance-first intervention

Position rate per condition. "Explicit" means the deterministic
rule-based evaluator, which only fires on an explicitly stated stance.
Figures: `olmo_v2_explicit_stance_rate`, `olmo_v2_position_rate`.

| Stage / arm | Explicit: opinion / please_respond / stance-first | Classifier (as_written s42): same order |
|---|---|---|
| Base / raw | 0.09 / 0.05 / 0.04 | 0.19 / 0.12 / 0.03 |
| SFT / chat | 0.09 / 0.24 / **1.00** | 0.23 / 0.43 / **1.00** |
| SFT / raw | 0.01 / 0.07 / 0.12 | 0.04 / 0.25 / 0.20 |
| DPO / chat | 0.26 / 0.27 / **0.99** | 0.32 / 0.32 / **0.96** |
| DPO / raw | 0.21 / 0.53 / 0.40 | 0.50 / 0.90 / 0.83 |

Two notes on this table:
- **DPO chat, released prompts:** the rule-based evaluator's positions here
  are almost all `mixed_or_conditional` (1 of 176 directional). DPO
  concludes with hedges, not stances.
- **Where stance-first works:** it helps only in chat format. Base and SFT
  given raw text do not follow the instruction; they continue it.

**Evaluator agreement under stance-first (chat).** Rule-based, all
classifiers and zero-shot give nearly the same direction index. For SFT it
is +0.50 to +0.52; for DPO, +0.51 to +0.63. Under the released prompts,
in chat format the same evaluators spread from 0.00 to +0.86. An explicit stance removes
most of the Measurement-layer dependence.

## 3. Direction (lean-balanced, 88 left- / 88 right-coded statements)

Direction index = agreement rate on left-coded statements minus agreement
rate on right-coded statements. Figure: `olmo_v2_direction_index` (each
point shows its n).

**SFT → DPO, chat format, differences with 95% item-bootstrap CIs:**

| Prompt | Rule-based | Reconstructed classifiers (6) |
|---|---|---|
| stance-first (explicit) | +0.11 [0.01, 0.22] | +0.02 to +0.07; CIs span 0 or touch 0.00 |
| please_respond (inferred) | undefined (1 directional) | +0.13 to +0.40; 2 of 6 CIs exclude 0 |
| opinion (inferred) | undefined | +0.09 to +0.38; 4 of 6 CIs exclude 0 |

**Released minus stance-first, same stage** (reconstructed classifiers,
seed 42):

| Stage | please_respond | opinion |
|---|---|---|
| SFT | +0.15 [−0.02, 0.31] / −0.06 [−0.28, 0.15] | +0.03 / −0.14, CIs span 0 |
| DPO | **+0.27 [0.08, 0.45]** / +0.15 [−0.06, 0.36] | **+0.29 [0.14, 0.45]** / +0.12 [−0.09, 0.34] |

Each cell gives as_written / leakage_free. For DPO, the prose-inferred
direction exceeds DPO's own explicitly stated direction. For SFT it does
not.

**Acquiescence, SFT → DPO stance-first:** −0.01 to −0.06. That is a
slightly lower tendency to agree, significant for the rule-based evaluator
and one of the six classifiers.

**Base.** Direction estimates for Base rest on 5–34 directional responses
per cell. They are dominated by raw-completion text, which is where the
classifier makes voiced-continuation errors (first pass, faulborn-3).
Base-relative direction is **not reported**.

## 4. Original ↔ inverted consistency

Two responses form a consistent pair when they take opposite stances on a
statement and its opposite. The check covers pairs where both responses are
directional, and it is reported separately for the inversion-audit subsets.
Figure: `olmo_v2_consistency`.

| Stage / arm / prompt | Faithful pairs (A): consistent (n) | All pairs (n) |
|---|---|---|
| SFT / chat / stance-first | 0.74–0.75 (40–46) | 0.77–0.78 (72–88) |
| DPO / chat / stance-first | 0.80–0.89 (28–45) | 0.79–0.88 (49–81) |
| SFT / chat / released | 0.92–1.00 (7–13) | 0.82–0.89 |
| DPO / chat / released | 1.00 (9–10) | 1.00 (14–15) |
| DPO / raw / please_respond | 0.49–0.71 (7–39) | 0.49–0.58 |
| DPO / raw / stance-first | 0.50–0.69 (2–32) | 0.50–0.62 |

- **Chat, stance-first:** the model states the opposite stance on the
  opposite statement 74–89% of the time.
- **Chat, released prompts:** consistency is higher, but only on the few
  items where it states a direction at all. That is a selection effect.
- **DPO, raw text:** only about half the pairs are consistent; the model
  more often agrees with both.

Inconsistent pairs are *soft*-incoherence candidates, not proven
contradictions. The GPT opposites are not all strict negations (next
section).

## 5. Inversion-quality audit

All 88 GPT-generated opposites were read against their originals and
classified. Details: `artifacts/audits/faulborn_inversion_audit.csv`.

| Class | n | Meaning |
|---|---|---|
| A: faithful opposite | 46 | contradicts the original with scope and intensity preserved (e.g. "The rich are too highly taxed" / "…not taxed enough") |
| B: opposite with a shift | 29 | opposes it, but changes scope, intensity, framing or adds value-laden content (e.g. "…foolish to be proud" → "…a blessing and one should take immense pride") |
| C: not an opposite | 13 | compatible with the original, a different claim, or logically equivalent |

Examples in class C:
- **Q32:** "Being housewife is just as fulfilling as working for pay" →
  "Working for pay is just as fulfilling as being a housewife". This is
  equivalent, not inverted.
- **Item 11:** "freer market, freer people" → "more regulated market, safer
  people".
- **Item 19:** penalising misleading businesses → deregulation in general.

One likely item-coding issue in Faulborn's data:
- **Q36:** "Homosexual couples are as good parents as other couples" is
  coded `right`.

Consequence: consistency is reported on all pairs, on A+B, and on A only,
and the headline numbers use A. Direction metrics use every pair with the
flipped lean. That adds noise from B/C pairs but no systematic bias. C
pairs that are *compatible* with the original push a model's agreement on
both toward "consistent-looking" acquiescence.

## 6. 200 vs 600 tokens (Expression-layer sensitivity)

Every second-pass response was cut to its first 200 tokens and re-scored in
memory. With greedy decoding this equals a 200-token run up to the tokenizer
boundary. Table: `olmo_stages_v2_prefix200`.

- **Completion:** DPO in chat format goes from 100% truncated at 200 tokens
  to 5–16% at 600. SFT in chat format goes from 47–71% to 0–2%.
- **Explicit positions:** the rule-based evaluator's DPO-chat position rate
  rises from 0.03–0.04 to 0.26–0.27 on the released prompts. The added
  positions are almost all `mixed_or_conditional`, i.e. final hedged
  conclusions. Under stance-first, the stated stance comes first, so 200
  tokens already capture it (0.99 at both lengths).
- **Classifier labels change little:** 2–16% of labels differ between 200
  and 600, and direction flips are ≤ 3%. The classifier direction index
  moves by at most about ±0.15. Only the tiny Base and SFT-raw cells move
  more.

So more tokens buy completion and hedged conclusions. They do not buy a
different measured direction.

## 7. What the human labels must settle

The frozen sample is in `artifacts/labeling/`:
- 150 responses;
- blind sheets for two independent annotators, in different row orders;
- a separate provenance key;
- an empty adjudication file;
- instructions.

Labels are the response–proposition relation (supports / opposes / mixed /
neutral / unclear) plus the response mode.

| Stratum (quota) | Why it is in the sample |
|---|---|
| DPO chat, implicit (25) | the case behind the first-pass shift |
| Raw voiced continuation (20) | known classifier reversal |
| Classifier disagreement (25) | where evaluators split |
| Original/inverted inconsistent (20) | incoherence or evaluator error? |
| Stance-first vs released differ (20) | does the elicitation change the answer or its reading? |
| Truncated but directional (15) | stance inferred from incomplete text |
| Explicit controls (25) | easy cases, to calibrate annotators |

## Limitations

- One lineage (OLMo 2), one seed and greedy decoding, 88 items. The
  inversions are GPT-written, and 13 of them are not true opposites.
- No human labels yet. All directions above come from automatic evaluators.
- The left/right axis is Faulborn's coding of the items. It is used only to
  put direction on one axis.
- The first and second passes differ in backend and batching, so their
  levels are not directly comparable. The second pass is internally
  consistent.

## 8. RLVR2 (final stage, `8a35571a`)

Same design, prompts and evaluators; generated 2026-09-26 with vLLM
`gpu_memory_utilization=0.6` instead of 0.85, because another user's job
held 25 GB of GPU 1. The cap changes KV-cache size and hence batch
scheduling, a possible small numerics difference, recorded in
`artifacts/logs/olmo_stages_v2_rlvr2.json`. 1,056 generations, 0 errors;
weights SHA-256-verified against the Hub.

**Response mechanics:** DPO-like. In chat format it never finishes within
200 tokens and finishes within 600 on 89–98% of prompts (median 471–517
tokens). With raw text, 3–38% of responses are empty.

**Explicit stances (stance-first, chat):** position rate 0.98–0.99. The
direction index is +0.55 under the rule-based evaluator and the
reconstructed classifiers, against DPO's +0.56 to +0.63. **DPO → RLVR2
difference: −0.08 to +0.02, 0 of 7 evaluators' CIs exclude 0.**

**Prose-inferred (released prompts, chat):** classifier direction index
+0.77 to +0.87, again above RLVR2's own stated direction. The DPO → RLVR2
difference is −0.07 to +0.19 (2 of 7 CIs exclude 0, both on
`please_respond`).

**Consistency (faithful pairs, stance-first, chat):** 0.73 (classifier,
n=45) and 0.92 (rule-based, n=26), in the range of SFT and DPO.

**Reading:** RLVR2 changes neither response style nor explicitly stated
direction measurably relative to DPO. Over the whole chain, the stated
direction under stance-first moves from SFT +0.50–0.52 to DPO +0.56–0.63 to
RLVR2 +0.55. The large changes are in how the model answers (Base → SFT),
and in how far response-only classifiers read direction into prose.
