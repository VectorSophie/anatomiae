# Protocol: re-reading the OLMo v2 stage study with J-qy — post-hoc sensitivity analysis

**Status:** frozen 2026-10-13, before any J-qy output on the full v2 cache
(beyond the 150 validation rows already judged in E_q_y v1).

## What kind of analysis this is

This is a **post-hoc sensitivity analysis**, **not** a preregistered
confirmatory experiment. Everything relevant is already known:

- the stage results under the eight existing evaluators
  (`docs/results/olmo_stages_v2.md`);
- J-qy's behaviour on the 150-row validation sample (`docs/results/eqy_v1.md`).

J-qy was **chosen after** that validation because it had the best human
agreement: κ 0.75, and 0 reversals among 117 human-directional responses.
It is also known to over-read: it forces a stance on 19 of 26 human
non-directional responses. E_q_y H1 rejected the claim that conditioning on
the statement removes over-read.

The question is whether the conclusions of `olmo_stages_v2.md` survive
being read by the evaluator with the best human agreement. A "yes" is
robustness evidence, not independent confirmation. A "no" means the thesis
gets revised; the evaluator does not get tuned.

**Layer:** Measurement. **No OLMo response is regenerated.**

## Evaluator, frozen from E_q_y v1 with nothing changed

`judge_qwen25_14b_pair_v1` (J-qy):

- `Qwen/Qwen2.5-14B-Instruct` @ `cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8`;
- vLLM, BF16, tensor-parallel 1, physical GPU 1 after the full preflight;
- prompt `JUDGE_QY`, parser `parse_judge` and label mapping imported
  unchanged from `scripts/eqy_experiment.py`;
- greedy decoding, temperature 0, top_p 1, `max_new_tokens=16`, seed 0;
- the model's chat template, single user turn.

**Statement `q` per response.** It is built exactly as the validation
sheet built it (`make_labeling_sheet.py`): the Faulborn original for `orig`,
and `pol_opposite_gpt` for `inv`.

**The one extension needed to cover the full cache.** 559 of 3,696 v2
responses are empty (`raw_text.strip() == ""`). Like every other evaluator
(`FaulbornNLIEvaluator`, `DeterministicStanceEvaluator`), J-qy does not
judge these. They get outcome `malformed` with a note. Generation errors
would get `generation_error`; there are none.

**Storage.**
- Judge generations go to a **new** immutable cache,
  `artifacts/cache/olmo_v2_jqy.jsonl`.
- Outcomes go to `artifacts/evaluations/olmo_v2_jqy.jsonl`.
- The 150 validation rows are judged again in this run instead of reused
  from `eqy_judge_v1.jsonl`. That gives a determinism check: identical
  request identity, but different vLLM batch composition. Agreement with
  E_q_y v1 on those 150 is reported.

## Metrics — existing definitions only

The existing metrics are reused unchanged from `scripts/analyze_olmo_stages.py`:

- `metrics()`;
- `direction_index = agree_rate(left-coded) − agree_rate(right-coded)`
  among directional outcomes;
- `acquiescence`, `position_rate`;
- `boot_diff()`: a 95% item bootstrap in which an item's orig and inv
  resample together. It uses 2,000 resamples, `default_rng(0)`, ratio of
  sums, and reports NaN when the metric is undefined in more than 10% of
  resamples.

**Coverage**, required next to every direction estimate. For each cell it
reports the share of all responses, empty ones included:
- `directional_rate`: agreement + disagreement;
- `mixed_rate`;
- `neutral_rate`: neutral_or_no_position, including J-qy's `unclear`
  (`unclear_rate` is reported separately as well);
- `other_rate`: malformed, refusal, irrelevant and so on.

A direction index is never shown without its `directional_n` and
`directional_rate`.

**Evaluators compared, on every table:**
- J-qy;
- `deterministic_stance_v1`;
- the 6 reconstructed Faulborn classifiers;
- `faulborn_nli_zeroshot`.

## Primary questions, frozen

Primary render is **chat (`native`)**. Raw-render cells are reported but
not interpreted for A–C: Base and SFT given raw text mostly do not answer,
and the coverage tables show this. Released prompts are `please_respond`
and `opinion`; the explicit condition is `stance_first`.

### A. SFT → DPO, chat

The estimates:
- `direction_index(DPO) − direction_index(SFT)` with its `boot_diff` CI,
  for each prompt and each evaluator;
- under J-qy, a difference-in-differences
  `Δ_released − Δ_stance_first` per released prompt. It uses the same item
  resamples for all four cells, so the CI is joint.

**Reading rule, fixed now.** The released-prompt shift "remains under J-qy"
if J-qy's released Δ CI excludes 0 **and** its DiD CI excludes 0.

| J-qy released Δ CI | DiD CI | conclusion |
|---|---|---|
| excludes 0 | includes 0 | "a shift exists but is not larger than the explicit-stance shift" |
| includes 0 | — | "not detected under J-qy" |

### B. Released vs stance-first within stage (chat; SFT, DPO, RLVR2)

The estimate: `direction_index(released) − direction_index(stance_first)`,
same stage, with a `boot_diff` CI, for each released prompt and each
evaluator.

Under J-qy, released direction "exceeds" stance-first direction for a
stage/prompt if the CI lower bound is > 0.

These `boot_diff` values for the old evaluators also re-derive the §3
table, whose CIs were not produced by a committed script. Any discrepancy
is reported.

### C. DPO → RLVR2, chat

The estimate: `direction_index(RLVR2) − direction_index(DPO)` with a
`boot_diff` CI, for each prompt and each evaluator, plus the DiD as in A.

"Remains small" if J-qy's stance-first and released Δ CIs both include 0,
or both lie within ±0.10.

### D. Coverage

Coverage is reported with every A–C estimate. It is the guard against
reading a direction off a handful of committed responses.

## Also reported

- position and commitment rates for prompt × stage × render, all evaluators;
- original/inverse consistency, reusing the v2 code, for all pairs, A+B and
  A-only;
- **class-A sensitivity:** A–C re-estimated on the 46 items whose GPT
  opposite is a faithful inversion (`A_faithful_opposite`), counting both
  their orig and inv responses;
- **response-level change**: for every cached response, J-qy's outcome
  against each old evaluator's, classified as
  - same direction;
  - **polarity change** (agreement ↔ disagreement);
  - **commitment change**, in either direction (directional ↔
    mixed/neutral);
  - other (both non-directional but different).

  This is reported by stage × render × prompt, and keeps polarity
  disagreement separate from commitment disagreement.

## Interpretation rules, fixed now

- **Not independent confirmation.** J-qy was chosen after validation.
- **The validation sample is not representative.** J-qy's validation error
  rates do not transfer as population rates.
- **Polarity and commitment are kept separate.** On the validation sample,
  J-qy's errors are almost all commitment (over-read), not polarity.
- **Over-read is not eliminated.** J-qy is not claimed to eliminate
  over-read, and its directional share on hedged prose is expected to stay
  inflated relative to humans.
- **No model ideology.** Direction indices describe measured
  response–statement relations under Faulborn's left/right item coding.
  They are not a model's ideology.
- **Contradictions are kept.** A result contradicting the working thesis is
  reported as such, and the thesis is revised.

## Outputs

- `artifacts/cache/olmo_v2_jqy.jsonl` and
  `artifacts/evaluations/olmo_v2_jqy.jsonl`;
- `artifacts/tables/olmo_v2_jqy_{metrics,questions,consistency,transitions,overlap}.csv`,
  plus `olmo_v2_jqy_questions.md`;
- `figures/olmo_v2_jqy_direction_coverage.{png,pdf}`;
- `docs/results/olmo_v2_jqy.md`, plus a §9 sensitivity section in
  `olmo_stages_v2.md`. The original evaluator results there stay
  unchanged.

## Deviations

(none yet)
