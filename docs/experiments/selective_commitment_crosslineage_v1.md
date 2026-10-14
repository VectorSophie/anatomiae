# Protocol amendment: selective commitment — cross-lineage replication v1

**Status:** frozen 2026-10-13T09:08:04Z (`ccabd8c`), committed and pushed **before** any
cross-lineage generation.

**Parent protocol:** `selective_commitment_confirmatory_v1.md` (frozen at
`188998e`, confirmed in OLMo-2-13B at `17d0490`).

**Everything not named in this amendment is identical to the parent:**
- the five free-form prompts and the stance-first reference, verbatim, with
  the `-` joiner;
- the 46 class-A items as the primary set and all 88 as secondary, orig and
  inv;
- J-qy (`judge_qwen25_14b_pair_v1`) with its exact model revision, prompt,
  parser, greedy decoding, 16 tokens and vLLM BF16 settings;
- the definitions of `C` and `SF`;
- the direction index;
- the pooled, unit-weighted selection gap;
- the item bootstrap (2,000 resamples, `default_rng(0)`, 95%);
- the minimum cell size of 10 directional SF units per subset;
- the treatment of empty and errored responses.

## 1. Why this lineage

The OLMo confirmation is one lineage. The Qwen2.5-14B foundation has two
independently post-trained descendants that are pinned and locked anchors
in this project:

| arm | model | revision | role here |
|---|---|---|---|
| **Q-Inst** | `Qwen/Qwen2.5-14B-Instruct` | `cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8` | **primary replication target**: a standard chat post-training, analogous to the OLMo stages |
| **Q-R1** | `deepseek-ai/DeepSeek-R1-Distill-Qwen-14B` | `1df8507178afcc1bef68cd8c393f61a886323761` | **secondary**: reasoning distillation onto the same base; the response format differs (below) |

Base (`Qwen2.5-14B`) is excluded, for the same reason as OLMo Base:
continuation, not answering.

## 2. Generation (per arm)

**Shared settings.**
- Chat template: the model's own, from its pinned tokenizer, single user
  turn, `add_generation_prompt=True`. No system message is added by us.
  - For Q-Inst, this template injects Qwen's default system prompt ("You are
    Qwen, created by Alibaba Cloud. You are a helpful assistant."). That
    is part of the model's native chat format and is kept.
  - Q-R1's template ends the prompt with `<｜Assistant｜><think>\n`.
- vLLM, BF16, tensor-parallel 1, `gpu_memory_utilization=0.85`, batches of
  176.
- Physical GPU 1 after the full preflight.
- Greedy decoding (temperature 0, seed 0).

**Q-Inst.** `max_new_tokens=600`, identical to the parent.

**Q-R1, declared differences, fixed now.**
- **Budget: `max_new_tokens=4096`.** R1-style models write a reasoning
  block before the answer. At 600 tokens most responses would end inside
  `<think>` with no answer, which would make commitment unmeasurable
  rather than measured.
- **Judged response `y`:** the text after the **last** `</think>` in the
  generation, stripped.
  - If there is no `</think>` (budget exhausted while reasoning), or nothing
    follows it, the response has no answer. It counts as **empty**: `C = 0`
    for free-form, and a non-directional `SF` for stance-first.
  - The reasoning text is never judged.
- **Decoding:** greedy, kept for identity with every other arm. DeepSeek
  recommends sampling (temperature 0.6), and greedy decoding can loop.
  Loops show up as missing `</think>` and are reported in coverage.

The raw generation, including `<think>`, is cached unmodified. Extraction
happens only at judging time.

**Caches.**
- generations: `artifacts/cache/selective_commitment_xl_v1.jsonl`;
- judge generations: `artifacts/cache/selective_commitment_xl_v1_jqy.jsonl`;
- outcomes: `artifacts/evaluations/selective_commitment_xl_v1.jsonl`.

## 3. Hypothesis and decision rule

**H1 (unchanged)**, computed per arm on class-A, pooled over the five
prompts:

    selection_gap[arm] = D_SF(C = 1) − D_SF(C = 0) > 0

- **Primary decision:** H1 **replicates in the Qwen lineage** if the 95%
  CI lower bound for **Q-Inst** is above 0.
- **Secondary:** the same rule applied to Q-R1 is reported, but it does not
  decide replication.
- A non-estimable gap (fewer than 10 directional SF units in either
  subset) counts as **not replicated** for that arm.

Primary analysis 2, the decomposition with coverage, is reported for both
arms, unchanged. Primary analysis 3 (stage comparison) does not apply,
because Q-Inst and Q-R1 are siblings, not stages. In its place, a
**descriptive** sibling contrast Q-R1 − Q-Inst is reported:
- commit rate;
- selection gap;
- `D_SF_all`;
- `D_ff_committed`.

It uses the same joint item bootstrap and carries no hypothesis.

## 4. Pre-specified secondary analyses

These are the same as the parent's §8:
- all 88 items;
- per prompt;
- rule-based `SF`;
- `P(commit | SF pole)`;
- coverage.

The parent's determinism check needs earlier generations of the same
requests, which do not exist for these arms, so it is replaced by:
- **Q-R1 answer-extraction coverage:** the share of responses with a
  `</think>` followed by non-empty text, by prompt.

## 5. Disclosed confound: self-judging

J-qy *is* Qwen2.5-14B-Instruct. In the Q-Inst arm it judges its own
outputs. In the Q-R1 arm it judges a sibling built on the same base.

Self-preference or familiarity with its own hedging style could shift `C`
in either direction. This is not corrected here: changing the evaluator
would break identity with the confirmed OLMo test, and no other judge has
been human-validated.

Mitigations:
- The rule-based `SF` sensitivity removes J-qy from the stance-first side.
- The planned human audit of `C` (`docs/experiments/commitment_audit_v1.md`)
  is the validation route for the commitment side.

A Q-Inst replication must be read with this confound named.

## 6. Interpretation rules

These are the same as the parent's:
- **Cross-family comparison is associational.** OLMo and Qwen differ in
  everything, so cross-family differences in gap size are reported
  descriptively, not attributed.
- **No model ideology.**
- **Failure is kept.** A failure to replicate is reported as such, and the
  thesis is limited to the OLMo lineage.

## 7. Outputs

- Tables: `artifacts/tables/selective_commitment_xl_v1_*.csv` and
  `..._primary.md`.
- Figure: `figures/selective_commitment_xl_v1.{png,pdf}`.
- Results: `docs/results/selective_commitment_crosslineage_v1.md`.

## Deviations

(none yet)
