# Selective commitment — cross-lineage replication v1 (Qwen2.5-14B family)

**Status:** primary arm complete (2026-10-14). The secondary arm,
DeepSeek-R1-Distill-Qwen-14B, is **pending**: its weights are still
downloading.

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
    covers OLMo responses only.
- **Default system prompt.** Qwen's chat template injects it. It is part
  of the native format, kept by design.
- **One arm.** The R1-distilled sibling, which tests whether a different
  post-training of the same base shows the gap, is pending.

## 6. Status of the thesis

Selective commitment now holds prospectively in OLMo-2-13B (SFT, DPO,
RLVR2) and in Qwen2.5-14B-Instruct, under identical prompts, items,
evaluator and decision rule. The Qwen replication carries the self-judging
confound.

## 7. Pending: Q-R1 (secondary)

Weights: `deepseek-ai/DeepSeek-R1-Distill-Qwen-14B@1df85071`, about 30 GB.
The download runs detached with the project's resumable retry loop
(status in `artifacts/logs/dl/r1distill.status`). When it finishes, run:

```bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1
bash scripts/gpu_preflight.sh
uv run python scripts/selective_commitment_xl_v1.py generate --arm q_r1
uv run python scripts/selective_commitment_xl_v1.py judge     # judges only new responses
uv run python scripts/selective_commitment_xl_v1.py score     # appends only new results
uv run python scripts/selective_commitment_xl_v1.py analyze   # adds the q_r1 arm + sibling contrast
```

Then complete this document, replacing this section.
