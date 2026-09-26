# Annotation precheck v1

## Provenance

- A: `labels_v1_annotator_A_human.csv` — labels assigned by the human researcher. AI was used only to package/export the completed annotations; it did **not** choose the labels.
- B: `labels_v1_annotator_B_agent_completed.csv` — agent-completed label set.
- Review: `human_agent_review_v1.csv` — the same human researcher re-read the 38 A/B disagreement cases after seeing both labels. It is a sensitivity artifact, not an independent-human consensus set.

The compact files store labels keyed by `sample_id`; the original frozen sheets remain the source of truth for statement/response text.

## Integrity checks

- 150 rows in A and 150 rows in B.
- No duplicate sample IDs.
- No missing required labels or confidence values.
- All relation/mode values are in the allowed taxonomy.
- A and B contain the same 150 sample IDs.
- Statement and response text matched exactly between the supplied A and B files for all 150 samples.
- The human-after-agent review contains all 38 disagreement samples, with no duplicates, missing adjudicated labels, or taxonomy violations.

## Human A vs agent B

| Dimension | Exact agreement | Cohen's kappa |
|---|---:|---:|
| Relation to proposition | 132/150 (88.0%) | 0.792 |
| Response mode | 126/150 (84.0%) | 0.690 |

There were 38 unique samples with a disagreement on relation and/or response mode.

## Human review sensitivity

After seeing the agent label and re-reading the 38 disagreements, human A changed:

- relation on 8/150 total responses (5.3%);
- response mode on 14/150 total responses (9.3%).

The reviewer selected a third relation label (different from both original A and B) on 2 cases, and a third mode label on 1 case. This indicates actual re-judgment rather than mechanically selecting one annotator.

## Interpretation limitation

Annotator A is the current independent human reference. Annotator B is an agent, so A-vs-B agreement is **not human inter-rater reliability**. The 38-case review is also not independent human-human adjudication because the reviewer saw the agent labels.

A second independent human (`Annotator C`) should label the frozen sample blind before the paper reports human-human reliability or consensus human labels. Until then, automatic-evaluator validation may use A as the primary human reference and the 38-case review only as a sensitivity check.

The frozen 150-response sample intentionally oversamples ambiguity/evaluator disagreement. It validates evaluator behavior but is not an unweighted representative sample for estimating population-level OLMo stage direction.
