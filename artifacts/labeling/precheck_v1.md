# Annotation precheck v1

## Provenance

- A: `labels_v1_annotator_A_human.csv` — labels assigned independently by the human researcher. AI was used only to package/export the completed annotations; it did **not** choose the labels.
- B: `labels_v1_annotator_B_agent_completed.csv` — agent-completed auxiliary evaluator.
- C: `labels_v1_annotator_C_human.csv` — second independent human annotation set, complete 150/150.
- Human-agent review: `human_agent_review_v1.csv` — A re-read the 38 A/B disagreements after seeing both labels; sensitivity artifact only.
- Human-human adjudication: `human_AC_adjudication_completed.csv` — all 61 A↔C disagreement cases resolved.
- Final reference: `labels_v1_human_consensus.csv` — 150-row researcher-adjudicated human consensus.

The compact files store labels keyed by `sample_id`; frozen sheets remain the source of truth for statement/response text.

## Integrity checks

Human C retained set:
- 150 rows, same 150 sample IDs as A;
- no duplicate IDs;
- no missing relation, mode, or confidence values;
- all relation/mode values are inside the frozen taxonomy;
- English statement text matches A on 150/150;
- model response text matches A on 150/150;
- no impossible `empty` labels on non-empty responses.

The retained C set is a blind rerun. An earlier C attempt was excluded before accepting any human-human reliability result because the annotator misunderstood the task. The rerun used clarified French instructions and French statement translations, kept the original English statement available as reference, and left model responses in original English.

## Human A vs Human C

| Dimension | Exact agreement | Cohen's kappa |
|---|---:|---:|
| Relation to proposition | 105/150 (70.0%) | 0.552 |
| Response mode | 115/150 (76.7%) | 0.564 |

There are 61 unique samples with a disagreement on relation and/or mode:
- 45 relation disagreements;
- 35 mode disagreements;
- 19 disagree on both.

The key directional result is much stronger than the raw five-way relation agreement:
among the 100 responses that **both humans** labeled directional (`supports` or `opposes`), 97/100 (97.0%) receive the same direction, Cohen's kappa ≈ 0.940. Only 3 cases are direct `supports` ↔ `opposes` reversals.

Most disagreement is therefore about **stance observability** — whether analytical/hedged prose is directional at all — rather than which direction it expresses.

Relation agreement by Human-A response mode:
- `explicit_stance`: 81/95 (85.3%);
- `analytical_exposition`: 15/38 (39.5%);
- `voiced_continuation`: 4/7 (57.1%).

## Human A vs agent B

| Dimension | Exact agreement | Cohen's kappa |
|---|---:|---:|
| Relation to proposition | 132/150 (88.0%) | 0.792 |
| Response mode | 126/150 (84.0%) | 0.690 |

This is human-vs-agent consistency, **not** human inter-rater reliability.

## Adjudication and interpretation

Adjudication is complete. The original A and C label sets remain unchanged.
The researcher resolved all 61 disagreement cases after seeing both labels, so the final
artifact is described as a **researcher-adjudicated human consensus reference**, not an
independent third-party consensus.

Final relation distribution:
- supports: 60
- opposes: 57
- mixed_or_conditional: 18
- neutral_or_no_position: 8
- unclear: 7

Final mode distribution:
- explicit_stance: 92
- analytical_exposition: 40
- incomplete: 10
- voiced_continuation: 5
- empty: 1
- refusal: 1
- other: 1

Adjudication was not a mechanical preference for one annotator. On relation conflicts it
selected A / C / a third label in 20 / 20 / 5 cases; on mode conflicts, 19 / 13 / 3.

The frozen 150-response sample intentionally oversamples ambiguity/evaluator disagreement.
It is valid for evaluator validation and human-human reliability on that sample, but it is
**not** an unweighted representative sample for estimating population-level OLMo stage direction.
