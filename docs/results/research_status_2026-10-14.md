# anatomiae — consolidated research status (2026-10-14)

**Scope.** Selective commitment in political-opinion elicitation, after
three pieces of work: the OLMo confirmation, the Qwen cross-lineage
replication (both arms), and the freezing of two human commitment audits.

**Pending: both human audits.** Their sheets are unlabelled, so questions
2 and 3 below cannot be answered yet. Their answers are not anticipated
here.

No further model families are planned by default.

## The claim under test

> Asked for an opinion in free form, post-trained models commit to a side
> on only about 30–40% of political statements. They commit selectively:
> on item-polarities whose explicitly elicited (stance-first) stance is
> more one-sided than that of the item-polarities they leave uncommitted.
> A free-form direction index therefore measures explicit direction on a
> selected subset.

The direction index uses Faulborn's left/right item coding. It describes
measured response–statement relations, not model ideology.

## 1. Is selective commitment replicated across OLMo and Qwen?

**Yes**, under one frozen protocol: the same prompts, items, judge (J-qy),
metric and decision rule. Gaps are class-A items, five prompts pooled,
with 95% item-bootstrap CIs.

| model | status of the test | gap [95% CI] | all items | commit rate |
|---|---|---|---|---|
| OLMo-2-13B-SFT | confirmatory | +0.24 [0.05, 0.41] | +0.35 | 0.28 |
| OLMo-2-13B-DPO | confirmatory | +0.33 [0.15, 0.52] | +0.41 | 0.30 |
| OLMo-2-13B-RLVR2 | confirmatory | +0.50 [0.27, 0.72] | +0.49 | 0.31 |
| Qwen2.5-14B-Instruct | prospective replication (primary) | +0.32 [0.16, 0.49] | +0.37 | 0.34 |
| DeepSeek-R1-Distill-Qwen-14B | prospective replication (secondary) | +0.34 [0.15, 0.54] | +0.23 | 0.39 |

These are two unrelated lineages and five checkpoints, and every CI lower
bound is above 0. Sources: `selective_commitment_confirmatory_v1.md` and
`selective_commitment_crosslineage_v1.md`.

## 2. Does it survive a human commitment audit?

**Pending.**
- **Sample:** 150 representative OLMo free-form responses, 10 random per
  stage × prompt cell (`docs/experiments/commitment_audit_v1.md`).
- **Tooling:** `scripts/analyze_commitment_audit.py`, unit-tested. Its
  `check` step validates the completed sheets and builds the A↔C
  adjudication queue without opening the key.
- **Deciding analysis (frozen):** J-qy's over-read and under-read **by
  stance-first pole**. The gap is an evaluator artefact only if J-qy
  over-reads disproportionately on left-pole item-polarities or under-reads
  on right-pole ones.

Global accuracy alone will not settle it.

## 3. Is the Qwen result robust to self-judging?

**Partly addressed; pending human labels.**

What is known:
- J-qy is Qwen2.5-14B-Instruct, so in the primary Qwen arm it judges its
  own outputs.
- With the stance-first side read by the rule-based evaluator instead, the
  Qwen-Instruct gap shrinks to +0.14 [0.00, 0.30] on class A, and +0.30
  [0.11, 0.49] on all items.
- In OLMo and R1, the rule-based sensitivity stays clearly positive
  (+0.26 to +0.45 and +0.31).

So the stance-first side of the Qwen-Instruct result is the most
judge-dependent number in the set.

What is frozen: a post-result validation audit of `C` on 100 random
Qwen-Instruct free-form responses (`commitment_audit_qwen_v1.md`). It is
**not** an independent confirmation of the replication. It validates the
commitment measurement.

## 4. Does R1-style distillation preserve, strengthen or weaken the effect?

**Preserves it.** The Q-R1 − Q-Inst contrast is +0.03 [−0.15, 0.21] on class
A and −0.14 [−0.30, 0.02] on all items; neither is detectable.

R1 changes *how* it declines to commit:
- mostly mixed/conditional rather than neutral;
- stance-first directional only 0.77 of the time, against 0.94–0.97 for
  the other arms.

## 5. Evidence classes

| class | finding | document |
|---|---|---|
| **Confirmatory (preregistered)** | E_q_y H1 not supported: showing the judge the statement does not reduce over-read. H3 supported: 4-class agreement improves (κ 0.64 → 0.75). | `eqy_v1.md` |
| **Confirmatory (preregistered)** | Selective commitment, OLMo SFT/DPO/RLVR2: confirmed at all three stages | `selective_commitment_confirmatory_v1.md` |
| **Prospective replication** | Qwen2.5-14B-Instruct (primary, replicates); R1-Distill (secondary, positive) | `selective_commitment_crosslineage_v1.md` |
| **Post-hoc sensitivity** | OLMo v2 read by J-qy: the large SFT→DPO released-prompt shift is not detected; released > stance-first at every stage; DPO→RLVR2 shows no large change | `olmo_v2_jqy.md` |
| **Exploratory** | The original discovery of selective commitment (v2, +0.42 to +0.63); E_q_y's residual over-read sits on the contested human commitment boundary; stance stability under batch-dependent text divergence; R1's mixed-heavy mechanics | `olmo_v2_jqy.md`, `eqy_v1.md`, `selective_commitment_confirmatory_v1.md` §5 |
| **Evaluator validation (on a difficult sample)** | Human A/C consensus vs 8 evaluators: response-only classifiers over-read, the rule-based evaluator under-reads | `human_validation_v1.md` |
| **Post-result validation (frozen, pending labels)** | OLMo commitment audit; Qwen self-judging audit | `commitment_audit_v1.md`, `commitment_audit_qwen_v1.md` |

Descriptive, pre-specified secondaries in the confirmatory and replication
runs:
- `P(commit | stance-first pole)`: right-pole stances are rarely
  volunteered in OLMo and Q-Inst. The right-pole cells are small: n = 8–16
  on class A.
- The DPO → RLVR2 rise in selection gap (+0.17), with flat commitment
  rate.

## 6. Recommendation

**Manuscript drafting, conditional on the two human audits**, and no new
model experiment.

The central claim is confirmed in one lineage and prospectively
replicated in a second (both siblings). Further lineages would add sample
count but would not answer the remaining reviewer-critical question.

That question is whether the commitment variable `C`, measured only by an
LLM judge, makes errors related to the stance-first pole. A reviewer will
ask exactly this. The two frozen audits answer it, and labelling them is a
human task.

- **If the audits show approximately pole-independent J-qy errors:**
  draft. The evaluator-artefact explanation is then weakened directly.
- **If they show pole-dependent errors:** the single remaining experiment
  is to **re-estimate the selection gap with human-labelled `C`** on a
  dedicated sample. That sample should be stratified on the stance-first
  pole and large enough in the rare right-pole cells, which the
  representative audits cannot be. The claim would be revised to whatever
  that estimate supports.

Drafting can begin on the parts that do not depend on the audits:
- the measurement-layer results (E_q_y, the human validation);
- the elicitation and selection framing;
- the methods.

The headline claim's wording should wait for the audits.
