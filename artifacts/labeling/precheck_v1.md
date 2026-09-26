# Annotation precheck v1

Two completed label sets were supplied for the frozen 150-response sample.

## Provenance

- A: `labels_v1_annotator_A_human.csv` — labels assigned by the human researcher. AI was used only to package/export the completed annotations; it did **not** choose the labels.
- B: `labels_v1_annotator_B_agent_completed.csv` — agent-completed label set.

The compact files store labels keyed by `sample_id`; the original frozen sheets remain the source of truth for statement/response text.

## Integrity checks

- 150 rows in A and 150 rows in B.
- No duplicate sample IDs.
- No missing required labels or confidence values.
- All relation/mode values are in the allowed taxonomy.
- A and B contain the same 150 sample IDs.
- Statement and response text match exactly between the supplied A and B files for all 150 samples.

## Agreement

| Dimension | Exact agreement | Cohen's kappa |
|---|---:|---:|
| Relation to proposition | 132/150 (88.0%) | 0.792 |
| Response mode | 126/150 (84.0%) | 0.690 |

There are 38 unique samples with a disagreement on relation and/or response mode. They are listed in `adjudication_v1_pending.csv`.

## Interpretation limitation

Annotator A is a genuine human annotation set and may be used as the current human reference for this sample. Annotator B is an agent, so A-vs-B agreement is **not human inter-rater reliability**. It is useful as an AI-vs-human consistency/sanity check and for locating ambiguous cases. A second independent human annotator remains preferable if the paper intends to report human-human reliability or use consensus human labels for directional claims.
