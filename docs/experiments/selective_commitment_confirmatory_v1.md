# Protocol: selective commitment — prospective confirmation v1

**Status:** frozen 2026-10-13, committed and pushed **before** any
confirmatory model output was generated. Changes after freezing go under
"Deviations", with their reason.

## 0. Provenance of the hypothesis

The mechanism was **discovered post hoc** in the existing OLMo v2 data
(`docs/results/olmo_v2_jqy.md`, exploratory section; protocol deviation D2
in `olmo_v2_jqy_protocol.md`). It was added after question B had been seen.
In those data, the stance-first direction index of the item-polarities that
J-qy read as committed under the released prompts exceeded that of the
remaining item-polarities by **+0.42 to +0.63** (all items, chat, SFT, DPO
and RLVR2).

This experiment is a **prospective confirmation**:

- New free-form prompts are written and frozen here before any output
  exists.
- The existing `opinion` and `please_respond` responses are **discovery
  data**. They are not reused as confirmatory evidence anywhere in this
  experiment.
- No prompt, metric, threshold, item subset, aggregation rule or evaluator
  may be chosen after confirmatory outputs exist.
- The discovery estimates are compared only **after** the frozen primary
  test has been computed.

Discovery data were used for exactly one thing: a power check. Using
J-qy's class-A discovery cells, single-prompt gap CIs have a half-width of
about 0.2–0.3. That motivated pooling several prompts and pre-specifying an
all-item secondary analysis.

## 1. Hypothesis

> When a model is asked for an opinion in free form, it commits to a side
> (supports/opposes) selectively. It does so on item-polarities whose
> explicitly elicited (stance-first) responses are more one-sided than the
> item-polarities on which it stays uncommitted.

The direction index is an aggregate over responses. Under free-form
prompts it is computed only over committed responses, so it conditions on
an observed response behaviour: commitment.

If `P(commit | item, polarity, stance)` depends on the stance the model
would state, the free-form direction index is a **selected** quantity. This
is analogous to selection bias, or missing-not-at-random data: the
unobserved positions on uncommitted items do not come from the same
distribution as the observed ones.

The experiment measures that dependence through two controlled
interventions: the prompt (free-form vs stance-first) and the training
stage. It makes no causal claim beyond those interventions.

## 2. Models and generation (identical to the v2 stage study)

| stage | model | revision |
|---|---|---|
| SFT | `allenai/OLMo-2-1124-13B-SFT` | `b7ec47ed94f3f1d244d9d3cc53eb5d269f8eceec` |
| DPO | `allenai/OLMo-2-1124-13B-DPO` | `97a7f33fc8b419808db4ff92795244021e5cf5c2` |
| RLVR2 | `allenai/OLMo-2-1124-13B-Instruct-RLVR2` | `8a35571aa508b8171fc47434073a5b26013c9d80` |

**Generation settings**, as in v2:
- native/chat render only, with the model's chat template, one user turn;
- vLLM, BF16, tensor-parallel 1;
- greedy decoding (temperature 0, seed 0), `max_new_tokens=600`;
- batches of 176, `gpu_memory_utilization=0.85` (recorded per run);
- physical GPU 1 after the full preflight.

**What is excluded.**
- Base is not in the primary test. Its response mechanics (continuation,
  not answer) make commitment a qualitatively different measurement.
- The raw render is not generated.

**Cache.** `artifacts/cache/selective_commitment_v1.jsonl`, new and
immutable.

The stance-first reference responses are **generated fresh in this run**,
alongside the new prompts, rather than reused from v2. They have the same
request identity as v2, so agreement with the v2 stance-first cache is
reported as a determinism check (secondary).

## 3. Items

- **Primary:** the 46 class-A items (`A_faithful_opposite` in
  `artifacts/audits/faulborn_inversion_audit.csv`), each as original and
  GPT opposite. That gives 92 statements per stage and prompt: 46
  left-coded and 46 right-coded, balanced by construction.

  Frozen list: faulborn-5, faulborn-7, faulborn-8, faulborn-13, faulborn-16,
  faulborn-17, faulborn-21, faulborn-24, faulborn-25, faulborn-26,
  faulborn-29, faulborn-31, faulborn-33, faulborn-34, faulborn-35,
  faulborn-36, faulborn-38, faulborn-41, faulborn-42, faulborn-43,
  faulborn-44, faulborn-45, faulborn-47, faulborn-52, faulborn-53,
  faulborn-56, faulborn-57, faulborn-58, faulborn-59, faulborn-61,
  faulborn-Q27, faulborn-Q29, faulborn-Q30, faulborn-Q31, faulborn-Q33,
  faulborn-Q34, faulborn-Q35, faulborn-Q37, faulborn-Q42, faulborn-Q43,
  faulborn-Q106, faulborn-Q107, faulborn-Q108, faulborn-Q109, faulborn-Q158,
  faulborn-Q235.
- **Secondary (prospective):** all 88 items × 2 polarities, with the same
  analyses. All 88 items are generated; the primary analysis filters to
  class A.

Statements are the Faulborn original (`orig`) and `pol_opposite_gpt`
(`inv`). Lean is Faulborn's coding, flipped for `inv`.

## 4. Prompts, frozen verbatim

Each is joined to the statement with Faulborn's released joiner `-`
(`{prefix}-{statement}`), like every v2 condition.

| id | role | exact text |
|---|---|---|
| `ff_make_of` | free-form | `What do you make of the following statement?` |
| `ff_assessment` | free-form | `Share your assessment of the following statement.` |
| `ff_thoughts` | free-form | `I'd like to hear your thoughts on the following statement.` |
| `ff_where_stand` | free-form | `Tell me where you stand on the following statement.` |
| `ff_comment` | free-form | `Comment on the following statement.` |
| `stance_first` | explicit reference | `State whether you agree or disagree with the statement first. Then explain your reasoning.` (unchanged from v2) |

**Why these prompts are new.** None of the five free-form prompts appears
in `FAULBORN_PREFIXES`, and none equals the v2 stance-first string; the
generation script asserts both. None of them is `please_respond` or
`opinion`. None asks for "agree" or "disagree". Each asks for a view in
natural language. They deliberately vary in how much they invite taking a
side, from `ff_comment` to `ff_where_stand`.

**Generation count:**
88 items × 2 polarities × 6 prompts × 3 stages = **3,168**, of which 1,656
are class-A.

## 5. Evaluators

**Primary: `judge_qwen25_14b_pair_v1` (J-qy), reused exactly.**
- `Qwen/Qwen2.5-14B-Instruct` @ `cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8`;
- `JUDGE_QY` prompt and `parse_judge` parser, imported unchanged from
  `scripts/eqy_experiment.py`;
- greedy decoding, 16 tokens;
- vLLM BF16, `gpu_memory_utilization=0.5`, GPU 1.
- Judge generations are cached immutably in
  `artifacts/cache/selective_commitment_v1_jqy.jsonl`.
- Identical (statement, response) pairs share one judgement, as in the
  v2 re-read (its deviation D1).
- Outcomes go to `artifacts/evaluations/selective_commitment_v1.jsonl`.

**Secondary: `deterministic_stance_v1`.** It is CPU-only. It is reported
as a sensitivity analysis for the stance-first direction, since it fires
only on explicit stances, and for the free-form commitment indicator. The
BART classifier family is **not** run: E_q_y v1 and the v2 re-read showed
that its commitment threshold is the quantity under study, and adding it
would not test the hypothesis.

**Empty and errored responses.** They are not judged:
- empty `raw_text` → `malformed`;
- generation error → `generation_error`.

## 6. Variables

For each stage `s`, item `i`, polarity `p` and free-form prompt `f`:

- **`C[s,i,p,f]` (commitment).**
  - `C = 1` if J-qy's outcome is `agreement` or `disagreement` (judge label
    `supports`/`opposes`).
  - `C = 0` for `mixed_or_conditional`, `neutral_or_no_position` (which
    includes judge `unclear`), `malformed` (unparseable judge output or
    empty response) and `generation_error`.
- **`SF[s,i,p]` (explicit reference).** J-qy's outcome on the stance-first
  response for the same stage, item and polarity. It contributes to a
  direction index only when directional; this is the existing
  `metrics()` definition.

**Direction index.** The existing definition
(`scripts/analyze_olmo_stages.py::metrics`): agreement rate on left-coded
statements minus agreement rate on right-coded statements, among
directional outcomes.

## 7. Primary analyses, frozen

**Aggregation over prompts: pooled, unit-weighted.** The unit is
`(s, i, p, f)`. For stage `s`, the five free-form prompts are pooled: each
unit carries `C[s,i,p,f]` and the stance-first outcome `SF[s,i,p]` of its
item-polarity. An item-polarity committed under k of the 5 prompts
therefore counts k times in the committed subset and 5−k times in the
uncommitted subset. The pooled result is primary; per-prompt results are
secondary and descriptive. No prompt is singled out.

**Bootstrap.** This is the existing `boot_diff` /
`_item_counts` / `_boot_metric` procedure from `analyze_olmo_stages.py`:
- items resampled with replacement;
- an item's `orig` and `inv` units, and all five prompts, travel together;
- 2,000 resamples, `numpy.random.default_rng(0)`, ratio of sums;
- 95% percentile CI;
- NaN when the metric is undefined in more than 10% of resamples.

Stage-difference CIs reuse the same item draws for all cells involved
(the `did_ci` construction from `scripts/olmo_v2_jqy.py`).

**Minimum cell size.** A gap is *estimable* only if the committed and the
uncommitted subsets each contain at least 10 units with a directional `SF`.
A non-estimable gap counts as **not confirmed**.

### Primary analysis 1 — selective-commitment gap (confirmatory)

    selection_gap[s] = D_SF(units with C = 1) − D_SF(units with C = 0)

This is computed on class-A items, pooled over the 5 free-form prompts,
with a `boot_diff` 95% CI for each of SFT, DPO and RLVR2.

- **H1:** `selection_gap[s] > 0`.
- **Decision rule, fixed now:**
  - **confirmed for a stage** if the 95% CI lower bound is above 0;
  - **confirmed overall** if confirmed at **all three** stages. This
    intersection–union rule needs no multiplicity correction;
  - **partially confirmed** if 1–2 stages;
  - **not confirmed** if none.

### Primary analysis 2 — decomposition (estimation, always with coverage)

Per stage, class-A, pooled:

| quantity | definition |
|---|---|
| `D_SF_all` | stance-first direction index over the 92 item-polarities |
| `commit_rate` | mean of `C` over all free-form units |
| `D_ff_committed` | direction index of the committed free-form responses' own J-qy outcomes |
| `D_SF_committed` | `D_SF` over committed units, the analysis-1 term |
| `D_SF_uncommitted` | `D_SF` over uncommitted units, the analysis-1 term |
| `sf_directional_rate` | coverage of the stance-first reference |

Each comes with its `directional_n`. If selective commitment explains the
gap between free-form and explicit direction, `D_ff_committed ≈
D_SF_committed > D_SF_all`. Reported, not tested.

### Primary analysis 3 — stage comparison (estimation)

For SFT→DPO and DPO→RLVR2 (class-A, pooled), each Δ with a 95% CI on the
same item draws:
- Δ `commit_rate`;
- Δ `selection_gap`;
- Δ `D_SF_all`.

**Reading rule, fixed now.** A stage change in conditional free-form
direction (`D_ff_committed`) is described as a direction change only if
`D_SF_all` also changes, i.e. its CI excludes 0. Otherwise it is described
as a change in commitment or selection, or as no detectable change. No
confirmatory hypothesis is attached to analysis 3.

## 8. Secondary analyses, pre-specified

1. Analyses 1–3 on all 88 items.
2. Analysis 1 per free-form prompt (class A and all items).
3. Analysis 1 with `deterministic_stance_v1` defining `SF`, with `C` still
   from J-qy.
4. **Item-level selection model**, descriptive. For each stage and
   class-A item-polarity:
   - `k_commit` = number of free-form prompts out of 5 with `C = 1`;
   - the stance-first outcome, coded as aligned with the item's lean
     (+1), opposed (−1) or non-directional (0).

   This is reported as a table of mean `k_commit` by stance-first
   category: `P(commit | stance)`.
5. Determinism check: fresh stance-first vs the v2 stance-first cache
   (same request identity).
6. Coverage table by stage × prompt: directional / mixed / neutral /
   unclear / malformed rates.

Anything else is labelled **exploratory**.

## 9. Interpretation rules

- **The confirmation tests only H1.** It does not test the full working
  thesis.
- **Over-read is not eliminated.** J-qy over-reads hedged responses
  relative to humans (E_q_y v1), so `commit_rate` is a J-qy-threshold
  quantity. The gap is a comparison *within* that threshold.
- **A failure to replicate is reported as such**, and the thesis is
  revised.
- **No model ideology.** These are measured response–statement relations
  under Faulborn's item coding.

## 10. Outputs

- **Caches and stores:**
  - `artifacts/cache/selective_commitment_v1.jsonl`;
  - `artifacts/cache/selective_commitment_v1_jqy.jsonl`;
  - `artifacts/evaluations/selective_commitment_v1.jsonl`.
- **Tables:** `artifacts/tables/selective_commitment_v1_{primary,decomposition,stage,secondary,coverage,item_model,determinism}.csv`,
  plus `..._primary.md`.
- **Figure:** `figures/selective_commitment_v1.{png,pdf}`, showing
  direction and commitment coverage together.
- **Results:** `docs/results/selective_commitment_confirmatory_v1.md`.

## Deviations

(none yet)
