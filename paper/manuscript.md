---
title: "What Does a Political Score Measure? Formation, Expression, and Evaluation in Language Models"
subtitle: "Working draft — not for circulation"
date: 2026-11-04
bibliography: ../CITATIONS.bib
---

<!--
DRAFT STATUS (2026-11-04)
- Every number comes from a committed artifact. Tables and figures come from
  scripts/make_paper_artifacts.py (paper/tables, paper/figures). Do not type numbers by hand.
- [PENDING-AUDIT]: sections that depend on the two human commitment audits. The completed
  sheets exist but are not yet imported (docs/results/research_status_2026-11-04.md).
- [PENDING-REESTIMATE]: depends on the frozen Qwen human selection-gap re-estimation
  (docs/experiments/selective_commitment_human_reestimate_qwen_v1.md).
- [CITE NEEDED: ...]: a claim that needs a reference not in CITATIONS.bib. Do not fill from memory;
  verify first, as for the existing entries.
- Markdown + pandoc citations, so it converts to a venue template without rewriting.
-->

# Abstract

Political-bias evaluations of language models usually report one number: a
direction score computed from the model's answers to political statements.
We argue that such a score is jointly produced by three layers. **Formation**
is how the model was trained. **Expression** is how a stance is elicited and
whether the model states one. **Measurement** is how an evaluator turns text
into a stance.

Our central empirical finding concerns the second and third layers. Asked for
an opinion in free form, post-trained models take an explicit side on only
about 28–39% of political statements. They take sides *selectively*: the
statements on which they commit are ones whose explicitly elicited stance is
markedly more one-sided than on the statements where they stay noncommittal.
A free-form direction index therefore conditions on a subset that the model
itself selects.

We confirm this *selective commitment* prospectively in OLMo-2-13B, at the
SFT, DPO and RLVR2 stages. The selection gaps are +0.23, +0.33 and +0.50 on a
direction index bounded in [−1, 1], every 95% CI above 0. It replicates
prospectively in Qwen2.5-14B-Instruct (+0.32) and its R1-distilled sibling
(+0.34). That is five checkpoints from two lineages.

Commitment is itself measured, and the measurement is contested. On a difficult
validation sample, humans disagree mostly about *whether* a response commits,
rarely about its direction (97/100 same direction when both see one). An LLM
judge shown the statement removes polarity errors but still reads most hedged
answers as committed. [PENDING-AUDIT: representative human audits of the
commitment variable in both lineages.] [PENDING-REESTIMATE: the Qwen gap
re-estimated with human-labelled commitment.] Until those return, the
magnitude of the Qwen estimate is reported as measurement-qualified.

# 1 Introduction

Claims that language models "lean" politically have become common
[@hagendorff2026inevitability; @feng2023pretrails; @santurkar2023opinionqa;
@faulborn2025littleleft; @lim2026biaseastwest]. Most such claims rest on a
pipeline:
1. present political statements;
2. elicit responses;
3. classify each response as agreeing or disagreeing;
4. aggregate into a direction score.

Each stage contains choices. Faulborn et al. already show that elicitation and
evaluation choices matter for the measured result [@faulborn2025littleleft].

This paper asks a narrower question: **what quantity does such a score
estimate?** We separate three layers:

- **Formation:** what training produced the model.
- **Expression:** whether, where and how the model states a position under a
  given elicitation.
- **Measurement:** what an evaluator counts as a stated position, and in which
  direction.

We do not ask whether a model "is" left or right. Every direction quantity here
is a measured relation between responses and statements under a fixed item
coding. None is an ideology.

Our main finding is a selection effect at the Expression/Measurement boundary.
Under free-form prompts, post-trained models commit to a side on roughly a
third of statements. The commitments are not a random third: on the statements
where the model volunteers a side, its explicitly elicited (stance-first)
stance is more one-sided than on the rest. In the language of survey
methodology, the free-form direction index is computed on outcomes missing not
at random [CITE NEEDED: missing-data / selection-bias reference, e.g. Rubin
1976 or Little & Rubin]. The same model can therefore show a near-ceiling
free-form direction and a markedly lower explicit direction over all items. A
free-form index can also stay flat while the explicit direction moves (§7.2).

Selection is defined through a commitment variable, and commitment is decided
by an evaluator. So we also study Measurement directly:
- human validation of eight evaluators;
- a preregistered test of whether conditioning the evaluator on the statement
  helps;
- two frozen human audits of the commitment variable used in the selection
  analyses;
- a frozen human re-estimation of the most judge-dependent result.

**Contributions.**
1. The Formation–Expression–Measurement decomposition, applied to political
   stance scoring (§3).
2. A prospectively confirmed and cross-lineage replicated *selective
   commitment* effect (§7–8). Every test was frozen before its data existed;
   evidence classes are labelled throughout (§5.5).
3. Evidence that the main measurement disagreement, between humans and between
   humans and evaluators, is about commitment, not direction (§6, §9).
4. A reusable protocol for checking whether an evaluator's commitment errors
   depend on the stance being measured. Done by stance-first pole, this is the
   test that separates selection from an evaluator artefact (§9).

# 2 Related work

**Measuring political leaning of LMs.**
- Feng et al. trace political leaning from pretraining data to downstream
  behaviour with the Political Compass Test [@feng2023pretrails].
- Santurkar et al. compare LM opinion distributions to US survey populations
  [@santurkar2023opinionqa].
- Faulborn et al. propose a theory-grounded measure and document sensitivity
  to elicitation [@faulborn2025littleleft]. We adopt their item bank, item
  coding and prompt joiner.
- Lim and Röttger examine prompt language and model origin [@lim2026biaseastwest].
- Hagendorff argues that some left-leaning measured bias is an expected
  consequence of alignment objectives [@hagendorff2026inevitability].

Our contribution is orthogonal to the direction any of these find. It concerns
which responses the direction is computed on.

**Prompt sensitivity and forced choice.** [CITE NEEDED: work on forced-choice
vs open-ended political evaluation; e.g. Röttger et al. 2024 "Political Compass
or Spinning Arrow?". Verify before adding to CITATIONS.bib.]

**LLM-as-judge validity.** [CITE NEEDED: LLM-as-judge reliability and
self-preference literature.] Our J-qy judge is Qwen2.5-14B-Instruct, which in
one arm judges its own outputs (§8).

**Models.** OLMo 2 [@olmo2] provides a fully open post-training chain. We also
use Qwen2.5-14B-Instruct and DeepSeek-R1-Distill-Qwen-14B [CITE NEEDED: Qwen2.5
technical report; DeepSeek-R1 report]. Amber [@llm360amber] and DataDecide
[@datadecide] are Formation resources used in pipeline validation, not in the
results here.

# 3 Framework: Formation, Expression, Measurement

![Figure 1. A reported political score passes through three layers. Selective commitment is an Expression-layer selection that the Measurement layer then scores.](figures/figure1_framework.pdf)

A reported direction score is a function of:
- the model's parameters (Formation);
- the elicitation and generation configuration (Expression);
- the evaluator and item coding (Measurement).

Within one lineage, changing the checkpoint while fixing Expression and
Measurement supports within-lineage attribution to a training stage. Changing
the evaluator on fixed generations isolates Measurement (no regeneration).
Changing the prompt on fixed checkpoints isolates Expression.

Comparisons across lineages differ in everything and are associational.

**The quantity at issue.** Let `C ∈ {0,1}` indicate that a free-form response
takes a side, and `SF` the model's stance when explicitly asked to state
agreement first. A free-form direction index uses only responses with `C = 1`.
If `P(C = 1 | item, SF)` depends on `SF`, the free-form index estimates explicit
direction on a selected subset, not over the item bank.

# 4 Data, models and elicitation

## 4.1 Items and inversion

- **Items:** 88 Faulborn statements [@faulborn2025littleleft], each with a
  left/right coding.
- **Inversion:** each statement is paired with a generated opposite statement
  (`inv`), whose coding is flipped. Inversion quality was audited
  (`artifacts/audits/faulborn_inversion_audit.csv`). **46 items are class A**
  (faithful opposite), which gives 92 item-polarities. Class A is the primary
  item set throughout; all 88 items form a pre-specified secondary set.
- **Balance:** class A is balanced by construction: 46 left-coded and 46
  right-coded statements.

## 4.2 Model lineages

| lineage | checkpoints | revision pinning |
|---|---|---|
| OLMo-2-1124-13B [@olmo2] | SFT, DPO, RLVR2 (Base in the earlier stage study only) | commit hashes in `docs/experiments/selective_commitment_confirmatory_v1.md` |
| Qwen2.5-14B | Instruct; DeepSeek-R1-Distill-Qwen-14B | `docs/experiments/selective_commitment_crosslineage_v1.md` |

Generation settings:
- native chat template; vLLM; BF16;
- greedy decoding; 600 new tokens (4,096 for R1, whose answer is the text after
  the last `</think>`).

Base checkpoints are excluded from the selection analyses: they continue text
rather than answer.

## 4.3 Elicitation conditions

There are five free-form prompts, frozen before any output, joined to the
statement with Faulborn's `-` joiner:
- "What do you make of the following statement?"
- "Share your assessment of the following statement."
- "I'd like to hear your thoughts on the following statement."
- "Tell me where you stand on the following statement."
- "Comment on the following statement."

None of them is a released Faulborn prompt, and none mentions agreement. The
explicit reference prompt is *stance-first*: "State whether you agree or
disagree with the statement first. Then explain your reasoning."

# 5 Measurement methods

## 5.1 Evaluators

- **J-qy (primary):** Qwen2.5-14B-Instruct as a judge. It sees the statement
  and the response, outputs supports / opposes / mixed_or_conditional /
  neutral_or_no_position / unclear, and decodes greedily with 16 tokens.
- **Rule-based** (`deterministic_stance_v1`): fires only on explicit stance
  openers. CPU-only.
- **Response-only BART-MNLI classifiers:**
  - Faulborn's zero-shot setup;
  - six reconstructions of their fine-tuned classifier, because the released
    weights are missing (`docs/results/faulborn_classifier_reproduction.md`).

## 5.2 Direction index and commitment

**Direction index:** the agreement rate on left-coded statements minus the
agreement rate on right-coded statements, among directional outcomes
(`analyze_olmo_stages.metrics`). It is a difference of two rates, so it lies in
[−1, 1].

**Commitment:** `C = 1` iff the evaluator outputs supports or opposes.

**Selection gap:**

    gap = D_SF(units with C = 1) − D_SF(units with C = 0)

- **Units:** item-polarity × free-form prompt, pooled over the five prompts and
  unit-weighted.
- **Bootstrap:** 95% item bootstrap with 2,000 resamples. An item's two
  polarities and all of its prompts travel together, so the five prompts are
  never treated as independent items.
- **Minimum cell size:** at least 10 directional SF units per subset.

## 5.3 Stance-first pole

An item-polarity's stance-first pole is:
- **left** if its SF agrees with a left-coded statement or disagrees with a
  right-coded one;
- **right** for the reverse;
- **non-directional** otherwise.

This is a property of a measured response under Faulborn's coding.

## 5.4 Human validation methodology

All human tasks use one taxonomy, of relation and response mode, defined in
`artifacts/labeling/README.md`.

There are two independent humans:
- **Annotator A** works in English.
- **Annotator C** uses French instructions and French statement translations,
  with the English original available and responses in English.

Disagreements are adjudicated by the researcher into a consensus reference. The
consensus is not a third independent rater.

There are four human artifacts:
1. a deliberately difficult 150-response validation sample (§6);
2. a representative audit of 150 OLMo free-form responses (§9);
3. a representative audit of 100 Qwen free-form responses (§9);
4. a pole-stratified re-estimation sample of 165 Qwen responses (§10).

Annotators are blind to:
- model, prompt and evaluator output;
- the stance-first response;
- item coding.

## 5.5 Evidence classes

| class | meaning here |
|---|---|
| confirmatory (preregistered) | protocol frozen and committed before the data existed |
| prospective replication | an amendment frozen before generation for the new lineage |
| post-hoc sensitivity | a re-analysis of existing generations with a protocol frozen before computing |
| post-result validation | a human audit of a measurement whose result was already known |
| frozen follow-up | an analysis triggered by a pre-specified decision rule, frozen before its labels exist |
| exploratory | chosen after seeing data |

# 6 Measurement validity on difficult cases

On a 150-response sample that deliberately oversamples hard cases
(`docs/results/human_validation_v1.md`):

**Human agreement.** Humans A and C agree on relation 70.0% of the time
(κ = 0.55). Among the 100 responses both call directional, they agree on
direction **97/100** (κ = 0.94). Human disagreement is about stance
observability, not support vs opposition.

**Evaluator agreement.**
- **Response-only classifiers:** they force a stance on 54–100% of the
  26 human non-directional responses, but rarely miss or reverse
  human-directional ones.
- **Rule-based evaluator:** it misses 60% of human-directional stances.

**Conditioning the judge on the statement (preregistered;
`docs/results/eqy_v1.md`).** The primary hypothesis, reduced over-read, was
not supported: 21/26 → 19/26, p = 0.63. Conditioning did:
- remove polarity reversals: 3/117 → 0/117;
- raise κ against humans from 0.64 to 0.75 (four-class error 22.4% →
  15.4%, p = 0.006).

**Exploratory finding.** The residual over-read sits where one of the two
humans also saw a direction: 16/20, always in that human's direction.

These rates are error rates on a difficult sample, not population estimates.

# 7 Selective commitment: prospective confirmation in OLMo-2-13B

Protocol `selective_commitment_confirmatory_v1`, frozen at `188998e` before any
output. 3,168 new generations.

## 7.1 Primary result

**Table 1** (`paper/tables/table1_selective_commitment.md`). The selection gap
is positive at all three post-trained stages, and every CI lower bound is
above 0:

| stage | gap [95% CI] |
|---|---|
| SFT | +0.23 [0.05, 0.41] |
| DPO | +0.33 [0.15, 0.52] |
| RLVR2 | +0.50 [0.27, 0.72] |

The preregistered intersection–union rule is met.

Free-form commitment is 0.28–0.31. Among committed responses, the free-form
direction (0.87–1.00) tracks the model's explicit direction on the same
item-polarities (0.82–0.97). Both exceed the explicit direction over all items
(0.64–0.71).

The uncommitted item-polarities are not explicitly neutral: asked directly,
the model still leans left-pole on them (0.48–0.60), just less one-sidedly.

## 7.2 Stage comparison

From DPO to RLVR2:
- the selection gap rises: +0.17 [0.04, 0.31];
- the explicit direction falls slightly: −0.07 [−0.15, −0.005];
- commitment rate and the free-form conditional direction are flat (the latter
  near ceiling).

A free-form index would have registered no change here. This is the selection
mechanism **hiding** a change, not inventing one.

## 7.3 Who volunteers

![Figure 2. J-qy free-form commitment rate by stance-first pole, class A. n = item-polarities (each × 5 prompts). Right-pole cells are small.](figures/figure2_commitment_by_pole_jqy.pdf)

Right-pole explicit stances are rarely volunteered. The mean number of the five
free-form prompts under which a right-pole item-polarity is committed falls
from 0.75 (SFT) to 0.12 (RLVR2), with right-pole n = 13–16 item-polarities.
Left-pole item-polarities stay at 1.5–1.9.

## 7.4 Robustness

- **Rule-based stance-first reading:** the gaps stay positive, at
  +0.26 / +0.44 / +0.45, all CIs above 0.
- **All 88 items:** +0.35 / +0.41 / +0.49.
- **Per prompt:** 15/15 point estimates are positive; 11/15 class-A CIs
  exclude 0.

# 8 Cross-lineage replication: Qwen2.5-14B

Amendment frozen at `ccabd8c`, before any Qwen generation; everything except
model identity is unchanged.

- **Qwen2.5-14B-Instruct (primary):** +0.32 [0.16, 0.49], replicating under
  the amendment's rule. Commitment rate 0.34; 79 left-pole and 10 right-pole
  class-A item-polarities.
- **R1-Distill-Qwen-14B (secondary):** +0.34 [0.14, 0.54].
- **Sibling contrast:** not detectable (+0.03 [−0.15, 0.21]). R1 declines to
  commit mostly as "mixed/conditional" rather than "neutral".

This is five checkpoints from two lineages, **not five independent model
families**. The three OLMo checkpoints share a base, and so do the two Qwen
checkpoints.

**Measurement qualification.** J-qy *is* Qwen2.5-14B-Instruct, so in this arm
it judges its own outputs. With the stance-first side read by the rule-based
evaluator instead, the class-A gap is +0.14 [0.00, 0.30]. That is the most
judge-dependent number in the study. Its human validation is in §9–10.

# 9 Human audits of the commitment variable [PENDING-AUDIT]

**Design.** Two frozen representative audits of J-qy's `C`:
- **OLMo:** 150 responses, 10 per stage × prompt cell, frozen at `ccabd8c`.
- **Qwen-Instruct:** 100 responses, 20 per prompt. This is a post-result
  validation.

**Analyses (frozen):**
- A↔C agreement;
- J-qy vs consensus: accuracy, κ, over-read and under-read with Wilson CIs, and
  direction reversals;
- **the deciding analysis: J-qy error by stance-first pole.**

**What would make the gap an evaluator artefact.** J-qy over-reading more on
left-pole item-polarities, or under-reading more on right-pole ones.

**Secondary calculations, frozen in code before the labels were imported:**
- the evaluator measurement effect: human minus J-qy commitment, and the
  left−right commitment asymmetry under each;
- pole-conditioned error differences with item-cluster bootstrap, Newcombe CI
  and Fisher exact p (descriptive);
- A-only and C-only sensitivity analyses.

Table 2 and Figure 3 are generated automatically once the completed sheets
are imported.

[PENDING-AUDIT: insert Table 2, Figure 3, and the reading-rule outcome for
Qwen. The prose must follow the keyed results.]

The provisional expectation from a pre-analysis outside the repository is:
- J-qy errors concentrate at the commitment boundary;
- direction reversals among jointly directional responses are rare;
- the OLMo asymmetry survives;
- Qwen shows pole-dependent error.

It is **not** a result. It is recorded only to explain why §10 was frozen.

# 10 Re-estimating the Qwen gap with human commitment [PENDING-REESTIMATE]

Protocol `selective_commitment_human_reestimate_qwen_v1`, frozen at `9776f7f`
before any new label.

**Estimand.** The primary Qwen gap with one change, `C_J-qy` → `C_human`.

**Sample.**
- Item-polarity clusters stratified by stance-first pole:
  - all 10 right-pole clusters;
  - all 3 non-directional clusters;
  - 20 of 79 left-pole clusters (seed 20261103);
  - five prompts each.
- 165 responses: 23 reuse audit labels and 142 are new (Table 4a).

**Estimation.**
- **Weights:** design weights `N_h/n_h`.
- **Interval:** the frozen item bootstrap run in two phases. It reproduces the
  frozen J-qy result exactly on the census.
- **Design check:** on the sampled units, J-qy gives +0.26 [0.09, 0.49]. The
  paired human−J-qy difference on the same units is therefore the primary
  comparison.

**Reading rule (frozen).**
- **Survival:** the human gap's CI lower bound is above 0.
- **Magnitude:** read from the CI of the paired difference.
- **If it does not survive:** the cross-lineage claim is restricted to OLMo.

[PENDING-REESTIMATE: Table 4b.]

# 11 Discussion

**A score is a conditional quantity.** In both lineages, a free-form political
direction index is mostly explicit direction on a model-selected subset. It
should be reported with its coverage: the share of statements on which the
model committed.

Comparing models or stages by free-form direction alone conflates three
things:
- how often a model commits;
- where it commits;
- what it would say if asked.

The DPO→RLVR2 contrast (§7.2) is a concrete case in which the explicit stance
moves while the free-form index cannot.

**Selection, not only stance.** The phenomenon is not "the model is more
left-pole than it seems" or "less". The model's *volunteering* is
stance-dependent. That is a property of Expression, and it has a
missing-not-at-random structure.

We do not show *why* right-pole stances are withheld. Candidates include safety
tuning, hedging norms on particular topics and item properties. None is tested
here.

**Measurement decides commitment.** Humans themselves disagree mostly about
commitment. Judges and classifiers differ mainly in their commitment threshold.
Direction, once commitment is agreed, is comparatively stable. The selection
effect is therefore defined relative to a measurement threshold, and whether it
survives a human threshold is an empirical question. §9–10 answer it, not the
judge.

[PENDING: final wording of the headline claim depends on §9–10.]

# 12 Limitations

- **Two lineages, one item bank, one primary evaluator family.** The
  checkpoints are not independent families.
- **The evaluator is an LLM judge.** In the Qwen-Instruct arm it judges its own
  outputs.
- **Item bank:**
  - English only;
  - generated inversions, with 46 items rated faithful;
  - one left/right coding [@faulborn2025littleleft].
- **Decoding:**
  - greedy only; stochastic decoding is untested;
  - batch-dependent text divergence under greedy BF16 vLLM (the measured
    stance was stable: 169–176/176 same outcome).
- **Small right-pole cells:** n = 8–16 item-polarities per checkpoint on
  class A.
- **Human references:**
  - two annotators plus researcher adjudication;
  - the French presentation for one annotator adds a translation layer, which
    is display-only.
- **Detecting the effect is not explaining it:** no causal claim beyond
  the prompt and training-stage interventions.

# 13 Reproducibility

- **Caches:** every generation and judgement is cached immutably and versioned
  (`artifacts/cache`, `artifacts/evaluations`).
- **Protocols:** frozen in `docs/experiments/` with their commit SHAs.
  Deviations are recorded.
- **Tables and figures:** produced by scripts from those artifacts; this
  paper's come from `scripts/make_paper_artifacts.py`.
- **Human-label keys:** local-only while annotation is blind.
- **Reproducibility unit:** the measured outcome, not byte identity
  (`docs/results/selective_commitment_confirmatory_v1.md` §5).
- **Hardware:** CUDA work used a single GPU with a power-limit preflight
  (`docs/reproducibility.md`).

# 14 Conclusion

A political score from free-form answers is not a direct readout of a model's
position. It is the explicit direction on the statements the model chose to
take a side on, as judged by an evaluator's commitment threshold.

This selective commitment is prospectively confirmed in one lineage and
replicated in a second. Whether its magnitude in the second survives
human-measured commitment is the last open measurement question, and it is
frozen and awaiting labels.

Reporting coverage, explicit-elicitation contrasts and evaluator validity
alongside any direction score would make such scores interpretable.

# References
