# Selective commitment — prospective confirmation v1

**Status:** complete (2026-10-13). Expression × Measurement.

**Protocol:** `docs/experiments/selective_commitment_confirmatory_v1.md`.
It was frozen, committed and CI-verified at `188998e` **before** any
confirmatory output existed. Two deviations followed, both after the
results:
- D1: an exploratory stance-level determinism check;
- D2: cosmetic relabelling of one secondary table.

**Data:** all new. 3,168 OLMo-2-13B generations (SFT, DPO and RLVR2; chat;
pinned revisions; vLLM BF16 greedy; 600 tokens), all judged by J-qy with
its E_q_y v1 settings unchanged:

- generation: 0 errors, 0 empty;
- judging: 3,154 unique judge requests, 0 errors, 0 malformed.

No discovery-phase `opinion` or `please_respond` response was used.

Sources:
- tables: `artifacts/tables/selective_commitment_v1_*.csv` and
  `selective_commitment_v1_primary.md`;
- figure: `figures/selective_commitment_v1.{png,pdf}`.

## 1. Preregistered confirmatory finding

**H1: the selection gap is positive at every post-trained stage.** The gap
is `D_SF(committed units) − D_SF(uncommitted units)`: class-A items, the
5 new free-form prompts pooled, unit-weighted, 95% item bootstrap.

| stage | committed units / 460 | D_SF committed (n directional) | D_SF uncommitted (n) | **selection gap [95% CI]** | decision |
|---|---:|---:|---:|---|---|
| SFT | 128 | 0.82 (126) | 0.58 (329) | **+0.24 [0.05, 0.41]** | confirmed |
| DPO | 140 | 0.93 (140) | 0.60 (295) | **+0.33 [0.15, 0.52]** | confirmed |
| RLVR2 | 144 | 0.97 (143) | 0.48 (302) | **+0.50 [0.27, 0.72]** | confirmed |

**Confirmed overall** under the frozen intersection–union rule: the lower
bound is above 0 at all three stages. All six cells cleared the minimum
cell size.

On new prompts the model had never been measured with, its spontaneous
commitments fall on item-polarities whose explicitly elicited stance is
markedly more one-sided than that of the item-polarities it leaves
uncommitted.

## 2. Primary estimation: decomposition (class A, pooled)

| stage | D_SF_all (SF directional rate) | commit rate (free-form) | D_ff among committed | D_SF on committed units | D_SF on uncommitted units |
|---|---|---:|---:|---:|---:|
| SFT | 0.65 (0.99) | 0.28 | 0.87 | 0.82 | 0.58 |
| DPO | 0.71 (0.95) | 0.30 | 1.00 | 0.93 | 0.60 |
| RLVR2 | 0.64 (0.97) | 0.31 | 0.99 | 0.97 | 0.48 |

The free-form direction among committed responses (0.87–1.00) is close to
what the same model states explicitly on those same item-polarities
(0.82–0.97). It sits well above the explicit direction over all items
(0.64–0.71). The free-form index is therefore mostly a **conditional**
quantity: explicit direction on a selected subset, observed on only 28–31%
of free-form responses.

The uncommitted item-polarities are not explicitly neutral. Asked
directly, the model still takes a mostly left-pole stance on them (D_SF
0.48–0.60), just a less one-sided one.

The direction index here is computed under Faulborn's left/right item
coding. It describes response–statement relations, not an ideology.

## 3. Primary estimation: stage comparison (class A, pooled, joint item bootstrap)

| change | commit rate | selection gap | D_SF_all (explicit) | D_ff committed (conditional) |
|---|---|---|---|---|
| SFT → DPO | +0.03 [−0.02, 0.07] | +0.09 [−0.10, 0.32] | +0.06 [−0.02, 0.15] | +0.13 [0.00, 0.32] |
| DPO → RLVR2 | +0.01 [−0.02, 0.04] | **+0.17 [0.04, 0.31]** | **−0.07 [−0.15, −0.005]** | −0.01 [−0.05, 0.00] |

Reading under the frozen rule:

- **SFT → DPO.**
  - The conditional free-form direction rises by +0.13, with a CI lower
    bound at 0.000. On all items it is +0.13 [0.03, 0.25].
  - The explicit direction does not change detectably.
  - So this is **not** described as a direction change. Commitment rate
    and selection gap do not change detectably either: the stage change in
    conditional direction cannot be attributed to any one component.
- **DPO → RLVR2.**
  - The explicit direction **decreases** slightly (−0.07). That replicates
    the −0.06 [−0.11, −0.00] of the v2 J-qy re-read on new generations.
  - The selection gap **increases** (+0.17).
  - The commitment rate is flat.
  - The conditional free-form direction is unchanged, at the ceiling of
    the index (≈1.0). A free-form index would have shown no change, while
    the explicitly stated direction moved down and commitment became more
    selective. This is the selection mechanism hiding a change, not
    inventing one.
- **Across post-trained stages, *how often* the model commits under
  free-form prompts barely moves** (0.28 → 0.30 → 0.31). What moves is
  *where* it commits.

## 4. Pre-specified secondary analyses

### All 88 items (pooled)

| stage | gap [95% CI] |
|---|---|
| SFT | +0.35 [0.18, 0.51] |
| DPO | +0.41 [0.23, 0.60] |
| RLVR2 | +0.49 [0.29, 0.68] |

All three are confirmed.

Stage comparison on all items:
- DPO → RLVR2 explicit direction: −0.04 [−0.09, −0.004];
- DPO → RLVR2 gap: +0.07 [−0.02, 0.17];
- SFT → DPO conditional direction: +0.13 [0.03, 0.25], with explicit
  +0.06 [−0.01, 0.13].

### Per prompt (not singled out; descriptive)

| subset | point estimates > 0 | CI lower bound > 0 |
|---|---:|---:|
| class A | 15 / 15 stage × prompt cells | 11 / 15 |
| all items | 15 / 15 | 14 / 15 |

The class-A cells without a CI above 0:
- SFT `ff_assessment` and `ff_comment`;
- SFT `ff_where_stand`: not estimable, only 7 committed units;
- DPO `ff_make_of`.

The all-items cell without one is SFT `ff_where_stand` (14 committed
units). `ff_where_stand` ("Tell me where you stand on…") yields the
**lowest** commitment, 0.08–0.16. Asking directly where the model stands,
in free form, produces more hedging than the other paraphrases.

### Stance-first read by the rule-based evaluator instead of J-qy

Gaps:
- class A: +0.26 [0.07, 0.45] / +0.44 [0.21, 0.70] / +0.45 [0.21, 0.71];
- all items: +0.44 / +0.46 / +0.42, all CIs > 0.

The result does not depend on J-qy reading the explicit stance.

### P(commit | stance-first pole)

Mean number of the 5 free-form prompts under which an item-polarity is
committed:

| stage | stance-first on left pole: mean k (never committed) | stance-first on right pole: mean k (never) | n left / right |
|---|---|---|---|
| SFT | 1.52 (48%) | 0.75 (62%) | 75 / 16 |
| DPO | 1.82 (46%) | 0.38 (77%) | 74 / 13 |
| RLVR2 | 1.93 (42%) | 0.12 (**94%**) | 73 / 16 |

These are class-A items. On all items the right-pole means are
0.68 / 0.53 / 0.45, and the right-pole cells are larger (34–40).

When the model's explicit stance falls on the right pole of Faulborn's
coding, it rarely volunteers it in free form, and this asymmetry grows
across post-training. `P(commit | item, stance)` is **not** constant.

The free-form direction index is therefore estimated on a sample selected
on the outcome. This is the structure of missing-not-at-random data: the
positions not volunteered are systematically different from those that
are.

### Coverage

| prompt type | directional | mixed | neutral |
|---|---|---|---|
| free-form | 0.08–0.36 | 0.07–0.36 | 0.28–0.85 |
| stance-first | 0.94–0.97 | — | — |

There are no empty responses and no malformed judgements. Truncation at 600
tokens is 0–7% for SFT and up to 41% for DPO free-form.

### Determinism of request identity (text)

The stance-first requests are identical to v2's (176/176 per stage). Yet
only 71, 10 and 10 of 176 texts are identical (SFT, DPO, RLVR2). Greedy
BF16 vLLM output depends on batch composition. **Byte-identical
regeneration is not guaranteed by request identity alone.**

## 5. Exploratory (deviation D1)

Does the stance survive that text divergence? The J-qy outcome on fresh vs
v2 stance-first responses:

| stage | same outcome | direction flips | median common prefix |
|---|---|---|---|
| SFT | 175/176 | 0 | ≈ 700 chars |
| DPO | 169/176 | 1 | ≈ 680 chars |
| RLVR2 | 176/176 | 0 | ≈ 660 chars |

The texts diverge late, and the measured stance is essentially stable. The
reproducibility unit for this project should therefore be the **measured
outcome**, with text-level divergence reported, not byte identity.

## 6. Comparison with the discovery estimate (only after the primary test)

| | SFT | DPO | RLVR2 |
|---|---|---|---|
| Discovery (v2 J-qy, all items, single released prompt) | +0.42, +0.44 | +0.42, +0.53 | +0.51, +0.63 |
| Confirmation (new prompts, all items, pooled) | +0.35 | +0.41 | +0.49 |
| Confirmation (class A, pooled) — primary | +0.24 | +0.33 | +0.50 |

The effect replicates in sign at every stage. It is somewhat smaller than
the post-hoc discovery estimates, most at SFT. That is the expected
shrinkage of an effect first seen after looking.

The stage ordering also replicates: the gap is smallest at SFT and largest
at RLVR2.

## 7. Interpretation (conservative)

**Supported prospectively:**
- In OLMo-2-13B's post-trained chat models, free-form political-opinion
  prompts elicit an explicit side on only about 30% of statements.
- The statements on which the model volunteers a side are ones where its
  explicitly elicited stance is more one-sided, and specifically more often
  on the left pole of Faulborn's coding.
- A free-form direction index is therefore largely the model's explicit
  direction on a selectively committed subset.

**Within the controlled interventions (prompt format, training stage):**
- the selection gap is present at SFT;
- it is larger after RLVR2 than after DPO;
- the commitment *rate* does not change between post-trained stages.

**Not shown:**
- **Cause.** Why the model withholds right-pole stances is not shown; it
  could be safety tuning, hedging norms on particular topics, or item
  properties.
- **Generality.** One lineage, one item bank, one evaluator.
- **Commitment threshold.** `C` comes from J-qy, which over-reads hedged
  text relative to humans (E_q_y v1). The gap compares within J-qy's
  threshold. The stance-first side is robust to the evaluator (rule-based
  sensitivity); the commitment side has not been human-validated on these
  new responses.

**Revised working thesis.** The previous version said post-training
changes *whether and where* a model commits. Among post-trained stages
these data support **where**, not **whether**:

> Among post-trained OLMo-2-13B stages, free-form prompts elicit a side on
> a stable ~30% of political statements. The model volunteers stances
> selectively: mostly where its explicitly elicited stance is one-sided
> toward the left pole of the item coding, rarely where it is on the right
> pole. That selectivity grows from DPO to RLVR2. Free-form direction
> indices therefore condition on a selected subset. They can stay flat
> while explicit direction moves, and they exceed the explicit direction
> over all items. Evaluator choice mostly changes the commitment threshold.

Base → SFT changes in commitment come from the v2 response-mechanics
results, not from this experiment.

## 8. Handoff — recommended next step

**Recommendation: cross-lineage replication**, under this protocol
unchanged except for model identity:
- Qwen2.5-14B-Instruct, and DeepSeek-R1-Distill-Qwen-14B as the reasoning
  variant of the same base;
- the frozen prompts, items, evaluator, metric and decision rule.

Selective commitment is now prospectively confirmed in one lineage. Its
generality is the main open question for a paper claim.

**In parallel (small, cheap):** a human audit of the commitment indicator
`C` on a **random** sample of the new free-form responses, e.g. 100–150
stratified by stage and prompt. `C` is the one variable here that has not
been human-validated on representative data.

**Not yet: manuscript drafting.** The central claim is now confirmed in one
lineage only, and its commitment variable has no human validation on
representative responses.

Do not start the cross-lineage run without first freezing a short protocol
amendment. It must name the models, pinned revisions and chat templates,
and keep everything else identical.
