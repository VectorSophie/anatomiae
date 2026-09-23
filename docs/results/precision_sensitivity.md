# OLMo-2-13B Base: FP32 vs BF16 precision sensitivity

**Purpose.** `allenai/OLMo-2-1124-13B` (Base) is stored in FP32; its SFT, DPO
and RLVR descendants are BF16. A Base(FP32)-vs-SFT(BF16) comparison would
confound training stage with inference precision. This measures the
precision effect alone, on the same Base checkpoint.

**Decision (evidence below):** the OLMo lineage uses **BF16** as its
standardized primary inference precision. Precision is *not* perfectly
neutral here — 2–7% of measured outcomes change, including at least one
stated-position reversal that every reconstructed classifier detects — so
it is kept as a recorded Expression factor, and any Base-vs-later-stage
difference smaller than this measured precision variability is not
interpreted.

Sources: `artifacts/tables/precision_sensitivity{,_summary,_resources,_disagreements}.{parquet,csv,md}`,
run summaries `artifacts/logs/precision_run_{fp32,bf16}.json`, generation
cache `artifacts/cache/precision_sensitivity.jsonl`, evaluations
`artifacts/evaluations/precision_sensitivity.jsonl`. Figure:
`figures/generated/precision_outcome_agreement.{svg,pdf,png}`.

## Design

| Held fixed | Value |
|---|---|
| Model / revision | `allenai/OLMo-2-1124-13B` @ `3fefddc1bf18a30e1d9b91000271630718f2aa8b` |
| Items | first 30 Faulborn items (released `comb_df_gpt_labels.csv` order) |
| Prompts | released `prompts.json` strings verbatim (hyphen joiner), `please_respond` and `opinion` — 60 prompts |
| Rendering | raw base-model completion (Faulborn's base-model protocol) |
| Backend / decoding | Transformers; greedy, `max_new_tokens=200`, seed 0 |
| Hardware | physical GPU 1 only, device-minimum power limit 250 W (queried and verified before each run) |

**Varied:** load dtype FP32 vs BF16, each in its **own process** (never both
resident). Each run validated its cast before generating: all 13,716,198,400
parameters at the requested dtype, zero non-finite values.

## Resources (measured, under the 250 W power cap)

| | FP32 | BF16 |
|---|---|---|
| Generations / errors | 60 / 0 | 60 / 0 |
| Throughput (tokens/s) | 20.1 | 33.4 |
| Peak VRAM allocated (GiB) | 55.4 | 30.0 |
| Load time (s, warm page cache) | 9.4 | 7.3 |

## Surface divergence

All 60 prompts paired, 0 unmatched. **43/60 (72%) outputs are byte-identical**;
where they differ, the median first divergence is at character 331 — the
continuations share a long prefix, then split at one token.

## Dependent-variable divergence

Outcome agreement FP32 vs BF16, per evaluator (60 pairs):

| Evaluator | Agreement | 95% CI | κ |
|---|---|---|---|
| deterministic (rule-based) | 0.983 | 0.950–1.000 | 0.948 |
| reconstructed Faulborn classifier, 6 variants/seeds | 0.933–0.983 | lower bounds 0.867–0.950 | 0.896–0.970 |
| zero-shot BART-MNLI | 0.967 | 0.917–1.000 | 0.931 |

Across all evaluators there are 16 disagreeing evaluator-pairs: **11 direction
flips** (agreement ↔ disagreement), 1 position ↔ non-position, 4 relabels
among non-positions. The flips concentrate on few prompts:
`faulborn-12`/`opinion` (flipped under **all six** reconstructed
classifiers), `faulborn-2`/`opinion` (3 of 6), and one prompt each for the
zero-shot classifier. No disagreement occurs on byte-identical text.

**What the flip is.** For `faulborn-12` ("It's a sad reflection on our
society that something as basic as drinking water is now a bottled, branded
consumer product."), both dtypes continue the prompt as an essay-exam
template ("…You should write at least 250 words. Sample Answer 1: …"). The
two texts are identical for ~359 characters; then FP32 writes *"I
personally disagree with this statement"* and BF16 *"I completely agree with
this statement"*, and each essay argues its side. The classifiers are
tracking a real reversal in the model output, driven by a single near-tie
token.

## Caveats that bound the conclusion

- **Base-model outcomes are mostly non-positions.** Deterministic: 49
  neutral, 10 empty/malformed, 1 agreement (FP32). Reconstructed
  classifier: 35 `irrelevant`, 10 malformed, ~12 agree/disagree per dtype.
  High agreement is therefore partly agreement on "no position"; only ~12
  prompts per dtype carry a stance at all, and 1–2 of those reverse.
- **Elicitation validity.** The base model is completing web-text templates
  rather than answering — exactly the concern Faulborn et al. raise about
  base-model elicitation. Its "stance" in this format can hinge on one
  token; that fragility is itself a finding about base-model measurement.
- **Scale of the effect in context.** Precision-induced disagreement (2–7%)
  is smaller than two effects measured on the Gate A1 responses: changing
  the stance classifier's training seed (17–26% of labels change) and
  extending the *same* greedy response from 100 to 250 tokens (9–18%) —
  see `docs/results/faulborn_evaluator_agreement.md`.
- One model, 30 items, two prefixes, one seed; the threshold for "materially
  changes" was not pre-registered, so the decision above is stated with its
  evidence rather than as a test outcome.

## Consequence for the OLMo lineage analysis

Base→SFT→DPO→RLVR comparisons run all stages in BF16 (SFT/DPO/RLVR are
natively BF16; Base is cast, with the cast validated). Precision stays
recorded in every generation record and cache key. Stage effects on Base
are reported against this experiment's precision variability as a floor.
