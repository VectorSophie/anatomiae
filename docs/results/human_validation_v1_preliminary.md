# Human validation v1 — preliminary status

**Status:** two independent human annotation sets complete; A↔C adjudication pending.

This document records facts available before the local-only frozen `key_v1.csv`
is joined back to evaluator outputs. It does not report automatic-evaluator accuracy yet.

## Primary human reference

Annotator A is the human researcher and independently labeled all 150 frozen responses.
AI assistance was used only to package/export the annotations, not to choose labels.

Relation labels:

| relation | n |
|---|---:|
| supports | 71 |
| opposes | 61 |
| mixed_or_conditional | 13 |
| unclear | 5 |
| neutral_or_no_position | 0 |

Response mode labels:

| mode | n |
|---|---:|
| explicit_stance | 95 |
| analytical_exposition | 38 |
| voiced_continuation | 7 |
| incomplete | 6 |
| other | 3 |
| empty | 1 |

All 150 A labels used confidence 3/3.

## Independent human C

The retained Annotator C set contains 150/150 labels in
`artifacts/labeling/labels_v1_annotator_C_human.csv`.

C was blind to A labels, agent-B labels, model/training-stage identity, evaluator outputs,
and political coding. To avoid the task misunderstanding observed in an earlier discarded
attempt, the final C protocol used French instructions and a French translation of each
statement while preserving the original English statement as an expandable reference and
leaving the model response in its original English text. The entire 150-response set was
relabeled from scratch.

The earlier C attempt is not used as a scientific annotation set because it showed clear
protocol noncompliance (for example, labeling non-empty responses as `empty` and answering
the proposition personally rather than classifying the model response). The exclusion and
rerun occurred before any human-human reliability claim was accepted.

C confidence distribution:

| confidence | n |
|---|---:|
| 1 | 9 |
| 2 | 20 |
| 3 | 121 |

## Human A vs Human C

Integrity checks passed:

- same 150 sample IDs;
- statement text matches 150/150;
- response text matches 150/150;
- no duplicate IDs;
- no missing required labels;
- all labels are in the frozen taxonomy.

| dimension | exact agreement | Cohen's kappa |
|---|---:|---:|
| relation | 105/150 (70.0%) | 0.552 |
| response mode | 115/150 (76.7%) | 0.564 |

There are 61 unique samples with at least one disagreement:
45 relation disagreements and 35 mode disagreements.

The politically important result is stronger than the raw multiclass relation score:
among the 100 responses that **both humans** judged directional
(`supports` or `opposes`), 97/100 (97.0%) received the same direction.
Only 3 cases are direct `supports` ↔ `opposes` reversals.

Most relation disagreements are therefore about whether an answer is sufficiently
committal to count as directional versus `mixed_or_conditional` /
`neutral_or_no_position`, not about opposite political direction.

The A–C disagreement queue is:
`artifacts/labeling/human_AC_adjudication_pending.csv`.

## Human A vs agent B sanity check

Agent B labeled the same frozen responses independently as an auxiliary evaluator.
This is **not** human inter-rater reliability.

| dimension | exact agreement | Cohen's kappa |
|---|---:|---:|
| relation | 132/150 (88.0%) | 0.792 |
| response mode | 126/150 (84.0%) | 0.690 |

There were 38 unique samples with a disagreement in relation and/or mode.

## Human-after-agent review sensitivity

The human researcher re-read all 38 A-vs-agent disagreement cases after seeing both labels.
This review is stored separately in `artifacts/labeling/human_agent_review_v1.csv` and
must not replace the original independent A labels.

Relative to original A, that review changed:

- relation on 8/150 responses (5.3%);
- response mode on 14/150 responses (9.3%).

## Sampling limitation

The 150-response set was quota-sampled to oversample ambiguous/evaluator-disagreement
cases. It is a measurement-validity sample, not a representative sample of the full
OLMo stage experiment. Unweighted stage-direction estimates from these 150 responses
would therefore be methodologically invalid.

## Next analysis

1. Adjudicate the 61 A↔C disagreement cases.
2. On the research workstation where the local-only frozen key exists, run:

```bash
uv run python scripts/analyze_human_validation.py
```

This computes automatic-evaluator accuracy, macro-F1, Cohen's kappa,
directional over-read/under-read, and errors by response mode/stage/render/prompt.
For a human-grounded population-level direction estimate, draw a representative sample
or use an explicitly justified sampling-weight estimator.
