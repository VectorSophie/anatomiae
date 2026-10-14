# Selective commitment — cross-lineage replication v1 (Qwen2.5-14B family)

**Status:** complete.
- Primary arm (Qwen2.5-14B-Instruct): 2026-10-13T09:15:05Z, `e14a823`.
- Secondary arm (DeepSeek-R1-Distill-Qwen-14B): generated and analysed
  2026-10-14.

**Amendment:** `docs/experiments/selective_commitment_crosslineage_v1.md`,
frozen and CI-green at `ccabd8c` before any Qwen generation. Everything not
named there is identical to the OLMo confirmatory protocol.

**Data:** 1,056 new Qwen2.5-14B-Instruct generations (`cf98f3b3`, chat,
default system prompt, vLLM BF16 greedy, 600 tokens):
- 0 errors, 0 empty, 7 truncated;
- 1,056 J-qy judgements: 0 errors, 0 malformed.

Sources:
- tables: `artifacts/tables/selective_commitment_xl_v1_*.csv`;
- figure: `figures/selective_commitment_xl_v1.{png,pdf}`.

## 1. Primary decision (preregistered)

**Selection gap** = stance-first direction on committed units minus that on
uncommitted units. Class-A items, the 5 free-form prompts pooled, item
bootstrap.

| arm | committed units / 460 | D_SF committed (n) | D_SF uncommitted (n) | **gap [95% CI]** | decision |
|---|---:|---:|---:|---|---|
| Qwen2.5-14B-Instruct | 157 | 0.98 (157) | 0.66 (288) | **+0.32 [0.16, 0.49]** | **replicates** |

The amendment's primary criterion for the Qwen lineage is met.

## 2. Decomposition (with coverage)

| subset | D_SF_all (SF directional rate) | free-form commit rate | D_ff among committed | D_SF on committed | D_SF on uncommitted |
|---|---|---:|---:|---:|---:|
| class A | 0.77 (0.97) | 0.34 | 0.98 | 0.98 | 0.66 |
| all items | 0.67 (0.95) | 0.34 | 0.89 | 0.89 | 0.53 |

The pattern is the OLMo pattern.

- **Partial commitment:** free-form prompts elicit a side on about a third
  of statements.
- **The free-form index tracks the selected subset:** among those, the
  free-form direction equals the model's explicit direction on the same
  item-polarities. That is above its explicit direction over all items.
- **The uncommitted items still lean:** asked directly, the model takes
  mostly left-pole stances on them too (0.53–0.66), just less one-sided
  ones.

## 3. Pre-specified secondary analyses

| analysis | result |
|---|---|
| All 88 items, pooled | +0.37 [0.20, 0.52] |
| Per prompt, class A | 5/5 CIs > 0 (+0.27 to +0.38) |
| Per prompt, all items | 5/5 CIs > 0 (+0.29 to +0.50) |
| Stance-first read by the rule-based evaluator, class A | **+0.14 [0.00, 0.30]** (lower bound just above 0) |
| Stance-first read by the rule-based evaluator, all items | +0.30 [0.11, 0.49] |

The rule-based sensitivity on class-A items is markedly weaker than with
J-qy (+0.14 vs +0.32). In OLMo the two agreed (+0.26 to +0.45). This
matters because of the self-judging confound below: on the stance-first
side, an evaluator other than Qwen itself gives a smaller gap.

**P(commit | stance-first pole)**, class A: mean number of the 5 prompts
under which the item-polarity is committed.

| stance-first pole | n | mean k | never committed |
|---|---:|---:|---:|
| left pole | 79 | 1.96 | 49% |
| right pole | 10 | 0.20 | 90% |

On all items the right pole has n = 28: mean k 0.61, never committed 79%.
Only 10 class-A item-polarities fall on the right pole, so this table is
coarse.

**Coverage.** Free-form directional rates are 0.26–0.39 (`ff_where_stand`
is lowest again, at 0.26), against 0.95 for stance-first. Mixed responses
are 0.24–0.38.

## 4. Cross-family comparison (descriptive, associational)

| | OLMo SFT | OLMo DPO | OLMo RLVR2 | Qwen2.5-Instruct |
|---|---|---|---|---|
| gap, class A | +0.24 | +0.33 | +0.50 | +0.32 |
| commit rate | 0.28 | 0.30 | 0.31 | 0.34 |
| D_SF_all, class A | 0.65 | 0.71 | 0.64 | 0.77 |

The size and coverage are similar across two unrelated lineages. They
differ in everything, so no attribution is made.

## 5. Caveats specific to this arm

- **Self-judging (disclosed in the amendment §5).** J-qy is
  Qwen2.5-14B-Instruct, so here it judges its own outputs.
  - The rule-based stance-first sensitivity is weaker in this arm, though
    still marginally positive. That is consistent with some judge-dependent
    inflation, but it does not show it.
  - The human commitment audit (`docs/experiments/commitment_audit_v1.md`)
    covers OLMo responses only. A separate post-result Qwen audit
    (`docs/experiments/commitment_audit_qwen_v1.md`) is frozen and awaits
    labels.
- **Default system prompt.** Qwen's chat template injects it. It is part
  of the native format, kept by design.
- **The R1-distilled sibling** is reported in §7. It was added after this
  section was first written.

## 6. Status of the thesis

Selective commitment now holds prospectively in OLMo-2-13B (SFT, DPO,
RLVR2) and in Qwen2.5-14B-Instruct, under identical prompts, items,
evaluator and decision rule. The Qwen replication carries the self-judging
confound.

## 7. Secondary arm: DeepSeek-R1-Distill-Qwen-14B (2026-10-14)

**Provenance.**
- Weights at `1df85071`, all 4 safetensors SHA-256-verified against the Hub
  (`artifacts/logs/r1distill_weights_sha256.json`).
- 1,056 generations under the amendment's R1 rules: 4,096 tokens, greedy,
  answer = text after the last `</think>`.
- 0 errors. 9 responses ran out of budget inside `<think>` and so have
  **no answer**; under the frozen rule they count as `C = 0` or a
  non-directional SF.
- 1,047 J-qy judgements: 0 errors, 0 malformed.

### Secondary decision (frozen rule; does not decide replication)

| arm | committed / 460 | D_SF committed (n) | D_SF uncommitted (n) | gap [95% CI] | |
|---|---:|---:|---:|---|---|
| R1-Distill-Qwen-14B | 180 | 0.96 (174) | 0.62 (196) | **+0.34 [0.15, 0.54]** | confirmed |

**All items:** +0.23 [0.09, 0.37]. That is smaller than Qwen-Instruct's
+0.37, but the CIs overlap (see the contrast below).

### Decomposition, class A

| | Q-Inst | Q-R1 |
|---|---:|---:|
| D_SF_all (SF directional rate) | 0.77 (0.97) | 0.78 (**0.80**) |
| free-form commit rate | 0.34 | 0.39 |
| D_ff among committed | 0.98 | 0.98 |
| D_SF on committed / uncommitted | 0.98 / 0.66 | 0.96 / 0.62 |

### Sibling contrast Q-R1 − Q-Inst (descriptive, joint item bootstrap)

| quantity | class A | all items |
|---|---|---|
| commit rate | +0.05 [−0.01, 0.11] | +0.03 [−0.01, 0.07] |
| selection gap | +0.03 [−0.15, 0.21] | −0.14 [−0.30, 0.02] |
| D_SF_all | +0.01 [−0.08, 0.10] | +0.05 [−0.04, 0.15] |
| D_ff committed | +0.00 [−0.00, 0.01] | −0.04 [−0.10, 0.01] |

None of these differences is detectable. R1-style distillation onto the
same base **preserves** the effect on the primary item set. The smaller
all-items gap is not distinguishable from Qwen-Instruct's.

### Pre-specified secondary analyses

| analysis | result |
|---|---|
| Stance-first read by the rule-based evaluator | class A +0.31 [0.06, 0.61]; all items +0.25 [0.01, 0.48] |
| Per prompt | 7/10 CIs > 0 |

The per-prompt cells that miss:
- class A `ff_make_of`: +0.24 [−0.02, 0.53];
- class A `ff_comment`: +0.20 [−0.10, 0.47];
- all items `ff_comment`: +0.10 [−0.15, 0.33].

All 10 point estimates are positive.

**Response mechanics differ from every other arm.**
- R1's non-commitment is mostly **mixed/conditional**: 0.42–0.57 of
  free-form responses, against neutral 0.05–0.17.
- Its stance-first responses are directional only 0.77 of the time (all
  items), against 0.94–0.97 for the other arms. R1 often concludes
  "conditionally" even when asked to state a side first.
- `ff_where_stand` does not depress commitment in R1 (0.37). In OLMo it was
  0.08–0.16, and in Q-Inst 0.26.

**P(commit | SF pole):**

| subset | left pole: mean k (n) | right pole: mean k (n) | non-directional: mean k (n) |
|---|---|---|---|
| class A | 2.59 (66) | 0.38 (8) | 0.33 (18) |
| all items | 2.47 (116) | 1.32 (19) | 0.32 (41) |

On all items, right-pole stances are volunteered more often than in Q-Inst
(0.61, n = 28). With these small right-pole cells this is descriptive only.

### Reading

R1-style reasoning distillation onto the Qwen2.5-14B base **preserves** the
selection gap on the primary items. It neither detectably strengthens nor
weakens it relative to the chat-tuned sibling.

R1 does change *how* it declines to commit: mostly "mixed/conditional"
rather than neutral, including under stance-first. The judge is again a
Qwen2.5 model judging a sibling of the same base. The self-judging caveat
(§5) applies here in weaker form.
