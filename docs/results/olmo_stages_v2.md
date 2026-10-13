# Formation layer, second pass: OLMo-2-13B Base → SFT → DPO → RLVR2

**Status:** Base, SFT, DPO and RLVR2 complete (RLVR2 added 2026-09-26, section 8). The frozen 150-response human validation sample now has two independent human label sets
and completed researcher adjudication. The stage-level directional results below remain
**classifier- or rule-measured**, because the 150 validation sample deliberately oversamples
ambiguous cases and is not a representative stage-direction sample.

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
   *Sensitivity (§9):* re-read by the human-validated judge J-qy, the
   SFT → DPO released-prompt shift is +0.08 / +0.11, with CIs including 0.
   That is about the size of the explicit-stance shift, so the large shift
   is evaluator-dependent. The released > stated gap itself persists at
   every stage, and an exploratory check traces it mainly to *selective
   commitment* rather than to misread prose.
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
*Sensitivity (§9):* these CIs are now regenerated by
`scripts/olmo_v2_jqy.py`, which reproduces them exactly. Under J-qy,
released exceeds stance-first at SFT too: +0.25 [0.11, 0.40] and
+0.25 [0.02, 0.45].

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

## 7. Human validation of the evaluators (completed)

A frozen 150-response sample was drawn by quota from the strata where
automatic measurement is least trustworthy. Each stratum had a reason:

| Stratum (quota) | Why it is in the sample |
|---|---|
| DPO chat, implicit (25) | the case behind the first-pass shift |
| Raw voiced continuation (20) | known classifier reversal |
| Classifier disagreement (25) | where evaluators split |
| Original/inverted inconsistent (20) | incoherence or evaluator error? |
| Stance-first vs released differ (20) | does the elicitation change the answer or its reading? |
| Truncated but directional (15) | stance inferred from incomplete text |
| Explicit controls (25) | easy cases |

Two independent humans (A, C) labelled every response, and the researcher
adjudicated their 61 disagreements. Full results:
[`human_validation_v1.md`](human_validation_v1.md) and
[`eqy_v1.md`](eqy_v1.md).

**What the labels settled, on this deliberately difficult sample** (not a
representative sample of the stage study):

- **Humans agree on direction, not on commitment.** When both humans call a
  response directional, they agree on its direction 97/100 times. Their
  disagreements are about whether hedged or analytical text commits at
  all.
- **The response-only classifiers that produced §3's released-prompt
  numbers rarely flip direction.** They reverse 5–10% of the responses
  humans call directional. They rarely miss a stance humans see either:
  under-read is 1–14%.
- **They do force stances.** They assign agree/disagree to 14–26 of the 26
  responses humans judge mixed or neutral. 18 of those 26 are mixed, a
  class these classifiers do not have.
- **Humans read direction in analytical prose too.** They judged 23 of 40
  analytical-exposition responses directional, and the classifiers' direction
  matched theirs in 19–23 of those cases.
- **The rule-based evaluator is conservative.** It misses 60% of the
  stances humans see.
- **A judge that sees the statement fixes direction errors, not
  commitment.** Qwen2.5-14B-Instruct shown the statement reverses none of
  the 117 human-directional responses and has the best human agreement
  (κ 0.75). But it still forces a stance on 19 of the 26 non-directional
  responses. That is the preregistered hypothesis E_q_y H1, which was **not
  supported**. Most of those 19 are responses where one of the two humans
  also saw that direction.

The labels do not give a population direction estimate for any stage.
They do say which kind of error each evaluator makes. §9 re-reads this
whole study with the judge that sees the statement.

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

## 9. Sensitivity: the study re-read by J-qy (2026-10-13)

Every non-empty v2 response was re-scored with `judge_qwen25_14b_pair_v1`
(J-qy), the evaluator with the best human agreement in
[`eqy_v1.md`](eqy_v1.md). Nothing was regenerated.

This is a post-hoc sensitivity analysis: J-qy was chosen after validation,
and it still over-reads hedged responses. Protocol:
`docs/experiments/olmo_v2_jqy_protocol.md`. Full results:
[`olmo_v2_jqy.md`](olmo_v2_jqy.md). Sections 1–8 above are unchanged.

| Question (chat) | Original evaluators | J-qy (directional rate) | Verdict under the frozen rule |
|---|---|---|---|
| A. SFT → DPO, released | +0.09 to +0.40 (6 reconstructed classifiers) | +0.08 [−0.04, 0.20] / +0.11 [−0.09, 0.35] (0.44→0.45 / 0.20→0.41) | large shift **not detected**; ≈ explicit shift |
| A. SFT → DPO, stance-first | +0.02 to +0.11 | +0.07 [0.00, 0.14] (0.97→0.94) | small explicit shift, reproduced |
| B. released − stance-first | DPO/RLVR2 positive; SFT not | +0.25 to +0.34 at SFT, DPO and RLVR2, all CIs > 0 | gap at **every** post-trained stage |
| C. DPO → RLVR2 | −0.08 to +0.19 | −0.06 [−0.11, −0.00] stance-first; ≈ 0.00 released | no large change; "within ±0.10" **not** formally met |

**Response-level comparison.** J-qy almost never reverses an old
evaluator's direction: at most 3.2% of responses for the reconstructed
classifiers, 6% for zero-shot. It differs on **commitment**, that is, on
which responses count as directional.

**Exploratory, added after seeing B.** On the items the model commits to
under released prompts, its stance-first direction is about as high as the
released direction. The items it leaves uncommitted have a much lower
stance-first direction (committed − rest +0.42 to +0.63, all CIs > 0).

The released-prompt "excess" therefore looks mainly like **selective
commitment**, an Expression effect, inherited by any direction index
computed only over committed responses. It does not look like an evaluator
reading direction into neutral prose.
*Update:* this was prospectively confirmed on new prompts and generations (§10).

## 10. Selective commitment, prospectively confirmed (2026-10-13)

The §9 exploratory finding was tested on **new** generations, with five new
free-form prompts, under a protocol frozen before any output. It covered
SFT, DPO and RLVR2, chat, with 46 faithful-inversion items as the primary
set and all 88 as secondary. Full results:
[`selective_commitment_confirmatory_v1.md`](selective_commitment_confirmatory_v1.md).

- **Primary result: confirmed at all three stages.** The stance-first
  direction of item-polarities the model commits to in free form exceeds
  that of the uncommitted ones:

  | stage | gap [95% CI] |
  |---|---|
  | SFT | +0.24 [0.05, 0.41] |
  | DPO | +0.33 [0.15, 0.52] |
  | RLVR2 | +0.50 [0.27, 0.72] |

  All items: +0.35 / +0.41 / +0.49. The effect is smaller than the post-hoc
  discovery estimate of +0.42 to +0.63, most at SFT.
- **Commitment is stable, where it falls is not.** Free-form commitment
  stays at 0.28–0.31 across stages. When the explicit stance falls on the
  right pole of the item coding, the model rarely volunteers it: RLVR2
  never does so for 94% of those item-polarities.
- **DPO → RLVR2.** Explicit direction falls by −0.07 [−0.15, −0.005], and
  the selection gap rises by +0.17 [0.04, 0.31]. The free-form conditional
  direction is unchanged, at the ceiling.
- **Reproducibility note.** Re-generating identical stance-first requests
  in a different vLLM batch reproduced the text for only 10–71 of 176, but
  the J-qy stance for 169–176 of 176.

## Limitations

- One lineage (OLMo 2), one seed and greedy decoding, 88 items. The
  inversions are GPT-written, and 13 of them are not true opposites.
- All stage-level directions above come from automatic evaluators. Human labels (§7) validate those
  evaluators on a difficult sample; they are not a stage-direction estimate.
- The left/right axis is Faulborn's coding of the items. It is used only to
  put direction on one axis.
- The first and second passes differ in backend and batching, so their
  levels are not directly comparable. The second pass is internally
  consistent.
