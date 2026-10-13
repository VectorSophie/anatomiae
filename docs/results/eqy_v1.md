# E(y) vs E(q, y) — does showing the evaluator the proposition help?

**Status:** complete (2026-10-13). Measurement layer, with generation fixed.

**Protocol:** `docs/experiments/E_q_y_protocol.md`, frozen at `09e306d`
before any E(q, y) output. Deviation D1 (`e2c6942`, also before any
output) made **Pair J** (Qwen judge, with vs without the statement) the
primary intervention and **Pair N** (BART-MNLI) a secondary comparison.

**Data:** the 150 frozen validation responses, scored against the
researcher-adjudicated human consensus. 7 `unclear` rows are excluded, so
**n = 143**: 117 human-directional and 26 human-non-directional (18 mixed,
8 neutral).

**Same guardrail as `human_validation_v1.md`.** This is a difficult,
quota-sampled validity set. Rates are evaluator-error rates on it, not
population or stage-direction estimates.

## Run record

| step | result |
|---|---|
| parse check: 5 responses outside the 150, both J prompts | 10/10 parsed as bare labels; no prompt change |
| judge: Qwen2.5-14B-Instruct @ `cf98f3b3`, vLLM BF16, greedy, 16 tokens | 300/300 generated; 0 errors; 0 malformed; 2 hit 16 tokens, both after a valid leading label |
| GPU | physical GPU 1 only, 250 W preflight passed, driver 610.57.04 |
| score | N-qy (BART-MNLI @ `d7645e12`) and both judge parses → `artifacts/evaluations/eqy_v1.jsonl` (450 results) |

All numbers below come from `artifacts/tables/eqy_v1_{evaluators,paired,strata,exploratory_contested}.csv`.

## Preregistered results

### Evaluator summary (n = 143)

| evaluator | sees q | acc | macro-F1 | κ | under-read (n=117) | over-read (n=26) [95% CI] | reversal of human-directional (n=117) |
|---|---|---:|---:|---:|---:|---:|---:|
| deterministic_stance_v1 | no | 0.420 | 0.398 | 0.284 | 0.598 | 0.077 [0.02, 0.24] | 0.009 |
| N-y faulborn_nli_zeroshot | no | 0.734 | 0.405 | 0.553 | 0.009 | 1.000 [0.87, 1.00] | 0.094 |
| N-qy nli_pair_bart_mnli_v1 | yes | 0.490 | 0.286 | 0.182 | 0.120 | 0.923 [0.76, 0.98] | **0.291** |
| J-y judge, response only | no | 0.776 | 0.561 | 0.639 | 0.068 | 0.808 [0.62, 0.92] | 0.026 |
| **J-qy judge, with statement** | yes | **0.846** | **0.647** | **0.749** | **0.026** | 0.731 [0.54, 0.86] | **0.000** |

### Paired tests

"Fixed by q" counts samples where E(y) errs and E(q, y) does not;
"broken by q" counts the reverse.

| pair | test | n | E(y) | E(q,y) | fixed by q | broken by q | exact McNemar p |
|---|---|---:|---:|---:|---:|---:|---:|
| **J (primary)** | H1 over-read | 26 | 0.808 | 0.731 | 3 | 1 | 0.625 |
| **J (primary)** | H2 reversal | 117 | 0.026 | 0.000 | 3 | 0 | 0.250 |
| **J (primary)** | H3 4-class error | 143 | 0.224 | 0.154 | 11 | 1 | **0.006** |
| N (secondary) | H1 over-read | 26 | 1.000 | 0.923 | 2 | 0 | 0.500 |
| N (secondary) | H2 reversal | 117 | 0.094 | 0.291 | 10 | 33 | 0.0006 (worse) |
| N (secondary) | H3 4-class error | 143 | 0.266 | 0.510 | 10 | 45 | <0.0001 (worse) |

### Decision under the protocol

**Pair J (primary): the preregistered criterion is NOT met.**

- **H1 fails.** Showing the statement lowers over-read only from 21/26 to
  19/26 (3 fixed, 1 broken, p = 0.63).
- **H2 holds.** Reversals fall from 3/117 to 0/117.
- **H3, a secondary hypothesis, is clearly supported.** 4-class error falls
  from 22.4% to 15.4% (11 fixed vs 1 broken, p = 0.006), and κ rises from
  0.64 to 0.75.

**Pair N (secondary): conditioning makes the off-the-shelf cross-encoder
much worse.**

- All 33 of its new reversals are human-`opposes` responses read as
  entailment of the statement. That is consistent with topical overlap
  between a long response and the proposition being scored as entailment.
- Under D1 this does not decide the protocol. It shows that a
  pair-conditioned *input* alone does not make an evaluator valid.

## What the statement fixes, and what it doesn't

### Fixed: direction

With `q`, the judge reproduces the human direction on **114/117**
human-directional responses, with no reversals.

Its 11 fixes include all 3 J-y reversals, two of them on raw
continuations of the generated opposite statement (`inv`). It also fixes 5
responses that J-y under-read as mixed or neutral.

Where J-y fails, the response's meaning depends on which proposition it
answers, and J-y cannot see it. This is the response-only failure the
protocol targeted. It shows up as polarity and commitment errors on
directional responses, not as over-read.

### Not fixed: commitment on human non-directional responses

J-qy still assigns agree/disagree to 19 of the 26 responses the consensus
calls mixed or neutral, against 21 for J-y.

By human mode:

| human mode | n | J-y over-read | J-qy over-read |
|---|---:|---:|---:|
| analytical exposition | 17 | 0.82 | 0.76 |
| explicit stance (hedged) | 7 | 0.71 | 0.57 |

## Exploratory, not preregistered: the residual over-read is at the human commitment boundary

Of the 26 consensus non-directional responses, **20 had one of the two
independent humans (A or C) label them directional**. Over-read on the two
groups:

| evaluator | both humans non-directional (n=6) | one human directional (n=20) | forced direction = that human's direction |
|---|---:|---:|---:|
| J-qy | 3/6 | 16/20 | 16/16 |
| J-y | 4/6 | 17/20 | 16/17 |
| N-y zero-shot | 6/6 | 20/20 | 15/20 |
| deterministic | 0/6 | 2/20 | 2/2 |

When the q-aware judge "over-reads", it almost always chooses exactly the
direction that one independent human also saw. A and C never gave
opposite directions on these items.

The residual over-read therefore sits mostly on the **contested
commitment boundary**, the same boundary on which A and C disagree
(`human_validation_v1.md`). It is not evidence of invented direction.

The zero-shot response-only classifier differs:

- it forces a stance on all 26, including the 6 that both humans call
  non-directional;
- its forced direction matches the directional human in only 15/20 cases.

With n = 6 and n = 20, and an analysis chosen after seeing the results,
this is a lead to test, not a finding.

## Interpretation

1. **Conditioning on the proposition matters most for direction.** It
   matters less for commitment. For a capable judge, `q` removes polarity
   errors (3 → 0) and improves overall agreement with humans (κ
   0.64 → 0.75). It barely moves the decision to call a hedged answer
   directional.
2. **Over-read is mostly not a response-only artefact.** It is mostly a
   commitment-threshold mismatch. A q-aware judge with a mixed label still
   over-reads 73%, but nearly always in the direction one human also saw.
   Commitment is where the human reference itself is least reliable.

   This revises the hypothesis that motivated the experiment. The working
   thesis says response-only evaluators *amplify* apparent shifts. On this
   sample their distinctive harm is:
   - polarity error on proposition-dependent text: the zero-shot classifier
     reverses 11/117 human-directional responses, against 0 for J-qy;
   - forcing a stance where *both* humans see none: zero-shot 6/6, against
     J-qy 3/6.

   It is not committing on contested hedges, which every
   non-rule-based evaluator does.
3. **"Pair-conditioned" is not a property that confers validity.** Feeding
   `(q, y)` into an off-the-shelf NLI head made agreement with humans much
   worse.
4. **For the stage analyses:** among the evaluators tested, J-qy is the
   most human-consistent automatic reader of free-form responses. Its
   errors are still concentrated on hedged and analytical answers, so
   directional claims about free-form prompts should report the
   commitment-boundary share separately. Re-scoring the full v2 stage
   study with J-qy is a Measurement-layer extension that needs no
   regeneration.

## Limitations

- **Small n for the primary test.** n = 26 non-directional responses. The
  protocol's power note expected only large over-read reductions to be
  detectable. The observed −0.08 has wide uncertainty, so H1's failure
  means "no large reduction", not "no effect".
- **Sample composition.** One sample, quota-sampled toward difficult cases.
  Mode cells for voiced continuation (n=5) and incomplete (n=5) are too
  small to interpret.
- **Judge provenance.** One judge model (Qwen2.5-14B-Instruct, also a
  Gate C subject lineage, but not the OLMo lineage judged here), one prompt
  pair, greedy decoding. Prompt-paraphrase robustness is untested.
- **Reference construction.** The reference is researcher-adjudicated, and
  the human task was itself pair-conditioned, which favours E(q, y) by
  construction. Even so, H1 failed.
- **Pair N confound.** N's two members differ in task formulation, not only
  in `q` (Deviation D1).
- **Exploratory analysis.** The contested-boundary analysis was chosen
  after seeing results.
