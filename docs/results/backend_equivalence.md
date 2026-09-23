# Gate B — Transformers vs vLLM measurement equivalence

**Question.** With everything else fixed, does switching the inference
backend change the text, and does it change the *measured outcome*?

**Answer.** The text changes often (only 18–72% of greedy outputs are
byte-identical, depending on the model); the measured outcome changes
rarely but not never: 0–15% of labels depending on model and evaluator,
0–10% for the reconstructed Faulborn classifiers, 0–3% for the
deterministic evaluator. Inputs are identical (100% equal prompt token
counts), so the divergence arises during decoding. The backend is
therefore a recorded Expression factor, not a no-op, and its effect is of
the same order as FP32 vs BF16 precision. **Gate B: passed as a
measurement** — the effect is characterized, not assumed zero.

Sources: `artifacts/tables/backend_equivalence{,_surface,_summary,_disagreements}.*`,
caches `artifacts/cache/backend_equivalence.jsonl` (+ the BF16 Transformers
side of 13B Base from `artifacts/cache/precision_sensitivity.jsonl`),
evaluations `artifacts/evaluations/backend_equivalence.jsonl`, per-run logs
`artifacts/logs/backend_equivalence_*.json`. Figure:
`figures/generated/backend_outcome_agreement.*`. Scripts:
`scripts/backend_equivalence.py`, `scripts/analyze_backend_equivalence.py`.

## Design

| Held fixed | Value |
|---|---|
| Models (pinned revisions) | OLMo-2-0425-1B-Instruct `48d788ec`, OLMo-2-1124-13B-SFT `b7ec47ed`, OLMo-2-1124-13B (Base) `3fefddc1` |
| Prompts | first 30 Faulborn items × released `please_respond` / `opinion` strings (hyphen joiner) = 60 per model |
| Rendering | once, by the HF tokenizer's chat template (Instruct/SFT) or raw completion (Base); the identical string is given to both backends |
| Decoding | BF16, greedy, `max_new_tokens=200`, seed 0, one request at a time |
| Hardware | physical GPU 1 only, 250 W minimum power limit, one backend per process; vLLM with FlashInfer attention, eager mode |
| Evaluators | deterministic, 6 reconstructed Faulborn classifiers, zero-shot BART-MNLI |

The 13B Base Transformers side was not regenerated: the same requests
already existed from the precision run. 180/180 prompts paired, 0 unmatched,
0 generation errors.

## Surface divergence

| Model | Prompt token counts equal | Byte-identical output | Finish reason agrees | Median first divergence (chars, when different) | Truncated (TF / vLLM) |
|---|---|---|---|---|---|
| OLMo-2-13B-Base | 100% | 72% (43/60) | 97% | 274 | 63% / 63% |
| OLMo-2-13B-SFT | 100% | 38% (23/60) | 97% | 319 | 63% / 63% |
| OLMo-2-1B-Instruct | 100% | 18% (11/60) | 100% | 308 | 100% / 100% |

Outputs share a long common prefix and then split at one token — the same
pattern as FP32 vs BF16. (The 13B Base byte-identical rate between
backends, 43/60, happens to equal its FP32-vs-BF16 rate within
Transformers.)

## Dependent-variable divergence

Outcome agreement, Transformers vs vLLM, 60 pairs per model:

| Evaluator | 13B-Base | 13B-SFT | 1B-Instruct |
|---|---|---|---|
| deterministic | 0.967 | 0.983 | 1.000 |
| reconstructed Faulborn (6) | 0.950–0.983 | 0.900–0.983 | 0.917–0.967 |
| zero-shot BART-MNLI | 0.933 | 0.933 | 0.850 |

κ ranges 0.59–1.00 (lowest: zeroshot on 1B). No outcome disagreement
occurs on byte-identical text (checked: 0). Aggregate position rates barely
move (e.g. reconstructed s42: Base 0.20 → 0.22, SFT 0.30 → 0.28, 1B 0.32 →
0.32): the backend reshuffles individual labels more than it shifts the
distribution.

**Direction flips** (agreement ↔ disagreement) across all evaluator-pairs:
13B-Base 6, 13B-SFT 5, 1B 10; of these, all but 5 are zero-shot, whose
labels are not interpretable on this material (A3). Among the
reconstructions: 13B-Base 3 (all on one prompt), 13B-SFT 1, 1B 1.

**The recurring prompt.** 13B-Base `faulborn-2` / `opinion` ("No one
chooses their country of birth, so it's foolish to be proud of it.") flips
direction under three reconstructions between backends. The texts diverge
at character 73: Transformers continues *"…I believe that people should be
proud of their country, but not because of where they were born"*, vLLM
*"…but I disagree with the statement that it is foolish to be proud of
it."* The same prompt also flipped between FP32 and BF16
(`precision_sensitivity.md`). Some prompts sit on a near-tie at which any
numerical perturbation — backend or precision — decides the stated stance.

**By completeness.** Where both sides completed (21 pairs each for the 13B
models), deterministic agreement is 1.00 and reconstructed-s42 0.90–0.95;
where both were truncated, 0.95–0.97 and 0.89–0.95. The small completed
strata do not support a claim that truncation drives the divergence.

## Consequences

- Backend is recorded in every request, cache key and analysis row
  (already enforced; requests carrying the wrong backend are refused).
- Comparisons across training stages or lineages are run on **one**
  backend. Mixing backends would add 0–10% label noise per evaluator on
  top of the effect being measured.
- Backend and precision divergences give a floor: a stage or lineage
  difference of a few percent in outcome shares on ~60 prompts is not
  distinguishable from inference numerics.

## Limitations

One lineage (OLMo; three checkpoints), 60 prompts per model, one decoding
setting (greedy, 200 tokens), sequential requests. vLLM batching, which
changes kernels and hence numerics, was not exercised. The 1B model
truncates every response at 200 tokens.
