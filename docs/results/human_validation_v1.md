# Human validation v1 — automatic evaluators vs human consensus

**Status:** complete (2026-10-13). Measurement layer.

**Question.** On responses where automatic stance measurement is least
trustworthy, how often do anatomiae's eight evaluators agree with humans?
Where they fail, do they miss stances or invent them?

**Scope guardrail.** The 150 responses are a *measurement-validity sample*,
quota-sampled to oversample hard cases (DPO implicit answers, raw voiced
continuations, classifier disagreement, orig/inverse inconsistency,
stance-first vs released prompts, truncated directional outputs) plus 25
explicit controls. Every rate below is an evaluator-error rate **on that
difficult sample**. None of them is an estimate of OLMo stage direction, and
none should be reweighted into one without a representative sample.

## Reference and provenance

- **Reference:** `artifacts/labeling/labels_v1_human_consensus.csv`. This is
  a researcher-adjudicated consensus of two independent human label sets
  (A and C, both frozen). It is not an independent third rater.
- **Provenance key:** `artifacts/labeling/key_v1.csv`. It was written once
  on 2026-09-23 by `make_labeling_sheet.py` and kept out of the repo while
  annotation was blind. It is now versioned.
  `scripts/verify_label_key.py` checks every row against the immutable
  artifacts:
  - sheet response vs cached `raw_text`, byte for byte;
  - statement vs the Faulborn item or its opposite;
  - stage, render, prompt, polarity and truncation vs the cached request;
  - all 8 evaluator labels vs `artifacts/evaluations/olmo_stages_v2.jsonl`.

  Result: **150/150 exact**.
- **Analysis:** `scripts/analyze_human_validation.py` generates
  `artifacts/tables/human_validation_v1_{evaluators,strata,confusion,human_agreement}.csv`,
  `human_validation_v1_evaluators.md` and
  `figures/human_validation_v1_overread_by_mode.{png,pdf}`.

### Denominators

| set | n |
|---|---:|
| all frozen responses | 150 |
| human `unclear`, excluded from every evaluator metric | 7 |
| scored (multiclass accuracy, macro-F1, κ) | **143** |
| human directional (supports 60 + opposes 57): under-read and direction accuracy | **117** |
| human non-directional (mixed 18 + neutral 8): **over-read** | **26** |
| human and evaluator both directional: reversal | varies, 47–116 (column `reversal_n`) |

Slices are smaller still; the strata table gives `n` for every cell.

### Label-space note

Humans chose among supports / opposes / mixed_or_conditional /
neutral_or_no_position / unclear. The evaluators have different label sets:

- **Faulborn-style classifiers:** agree / disagree / neutral / unrelated.
  They have **no mixed class**.
- **Deterministic evaluator:** can also emit `safety_refusal`.

`accuracy_raw` scores each evaluator's own labels, so an extra label such as
`unrelated` always counts as an error. `accuracy`, `macro_f1` and `cohen_kappa`
map every non-position evaluator outcome to `neutral_or_no_position` first.
The directional metrics (over-read, under-read, reversal) do not depend on
this choice.

## Human A vs Human C

From `human_validation_v1_human_agreement.csv`:

| dimension | n | exact | κ |
|---|---:|---:|---:|
| relation | 150 | 0.700 | 0.552 |
| response mode | 150 | 0.767 | 0.564 |
| direction, when both humans call it directional | 100 | 0.970 | 0.940 |

Humans disagree about **whether** a response commits to a stance. When both
think it does, they almost never disagree about which way.

## Evaluators vs consensus

From `artifacts/tables/human_validation_v1_evaluators.md`. Rates have 95%
Wilson CIs in the CSV.

| evaluator | acc | macro-F1 | κ | dir. exact (n=117) | under-read (n=117) | **over-read (n=26)** | reversal |
|---|---:|---:|---:|---:|---:|---:|---:|
| deterministic_stance_v1 | 0.42 | 0.40 | 0.28 | 0.39 | 0.60 | **0.08** [0.02, 0.24] | 0.02 (n=47) |
| faulborn_nli_zeroshot | 0.73 | 0.41 | 0.55 | 0.90 | 0.01 | **1.00** [0.87, 1.00] | 0.09 (n=116) |
| reconstructed as-written, seeds 42/43/44 | 0.68–0.73 | 0.44–0.45 | 0.50–0.55 | 0.80–0.88 | 0.06–0.14 | **0.65–0.92** | 0.06–0.08 |
| reconstructed leakage-free, seeds 42/43/44 | 0.68–0.70 | 0.43–0.46 | 0.51–0.54 | 0.81–0.83 | 0.12–0.14 | **0.54–0.77** | 0.05–0.06 |

## Findings

### 1. The two evaluator families fail in opposite directions

**The deterministic evaluator is conservative.**
- It rarely forces a stance: over-read 2/26.
- It misses most stances humans see: under-read 0.60.
- It finds direction almost only in explicit openers: 41/85
  explicit-stance directional responses, but 4/23 analytical ones and 0/4
  voiced continuations.

**The response-only BART-MNLI classifiers are permissive.**
- They almost never miss a human-directional response: under-read
  0.01–0.14.
- They assign a direction to most responses that humans judge
  non-committal: over-read 14/26 to 26/26.

Both families have κ ≈ 0.28–0.55 against humans. Neither is a
safe standalone instrument for free-form political text.

### 2. Over-read is a forced-choice artefact concentrated on mixed answers

Of the 26 human non-directional responses:
- 18 are `mixed_or_conditional`, which no Faulborn-style classifier can output;
- 8 are `neutral_or_no_position`.

The zero-shot classifier assigns agree/disagree to:
- all 10 mixed analytical responses;
- all 7 mixed explicit-stance responses (for example "I partly agree…");
- all 7 neutral analytical responses.

The as-written reconstruction (seed 42) does the same on 9/10, 7/7 and
6/7 respectively.

The leakage-free reconstruction (seed 42) is markedly better on mixed
analytical prose (2/10) but not on mixed explicit stances (6/7). Its seed
variation is large: over-read ranges 0.54–0.77 across the three
leakage-free seeds.

Over-read in human response-mode terms (figure
`human_validation_v1_overread_by_mode`):

| evaluator | analytical exposition (n=17) | explicit stance (n=7) |
|---|---:|---:|
| zero-shot | 1.00 | 1.00 |
| as-written seed 42 | 0.88 | 1.00 |
| leakage-free seed 42 | 0.35 | 0.86 |
| deterministic | 0.00 | 0.29 |

So the error is **"commits where humans see no commitment"**, and it is not
confined to analytical prose: explicitly hedged answers are forced too.

### 3. Direction itself is mostly right when humans see one — including in analytical prose

This partly contradicts a simple "classifiers invent direction from
analytical prose" story, and it is preserved as such.

- Humans judged **23 of 40** analytical-exposition responses directional.
- On those 23, the seven response-only classifiers give the human direction
  in 19–23 cases. Six of them make **0 reversals**; leakage-free seed 44
  makes 2.
- Across all modes, direct supports↔opposes reversals are 5–10% of
  responses that both the human and the classifier call directional.

Reversals concentrate in:
- **Raw render:** 0.07–0.16 across the seven classifiers, vs 0.00–0.02
  native.
- **Base:** as-written seed 42 reverses 0.25 of the Base responses that both
  call directional. That is a small cell: 20 human-directional, fewer
  classifier-directional.
- **The `classifier_disagreement` stratum:** 0.11–0.33.

On the 4 voiced continuations humans called directional:
- zero-shot reverses all 4;
- as-written seed 42 gets all 4 right.

With n=4 this is illustrative only. It is consistent with the
response-only premise problem: whether a continuation "agrees" depends on
which proposition it continues.

### 4. Implication for the Formation-layer results

On this sample, the measurement problem of response-only classifiers is
mainly:

- **over-commitment:** mixed or neutral responses are read as
  agree/disagree;
- **some polarity error** on raw continuations.

It is mainly **not** wrong direction on prose that humans also read as
directional.

The earlier result that released-prompt classifiers show a larger
SFT→DPO shift than explicit stances (`olmo_stages_v2.md`) is therefore
compatible with two contributions:

- **(a) Over-read of non-committal answers.** This is an evaluator artefact.
- **(b) Genuine implicit direction in analytical prose.** Humans perceive
  this too.

This sample cannot apportion the shift between (a) and (b) at population
level. That requires the representative sample in backlog item A.

The next experiment tests (a) directly: does giving the evaluator the
proposition, and a mixed/conditional option, reduce over-read without
increasing reversals? It is preregistered under `docs/experiments/` before any result is computed.

## Limitations

- **Consensus reference:** the reference is researcher-adjudicated after
  seeing A and C. Adjudication picked A, C or a third label in 20/20/5
  relation conflicts.
- **Small non-directional set:** n = 26, so over-read CIs are wide; see the
  CSV.
- **Tiny mode cells:** voiced continuation n = 5, incomplete n = 5.
- **Unclear rows:** the 7 `unclear` responses are excluded, not scored as
  errors.
- **Seed instability:** reconstructed-classifier results vary across
  training seeds by up to 0.27 in over-read. Single-seed conclusions about
  "the" Faulborn classifier are unsafe. These are reconstructions, not the
  authors' released weights, which lack a weights file.
- **Not a population estimate:** the sample's stage, render and prompt
  composition is set by quotas, not by the experiment's population.
