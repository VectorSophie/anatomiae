# Human validation v1 — preliminary status

**Status:** primary human annotation complete; second independent human pending.

This document records only facts available before the local-only frozen `key_v1.csv`
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

## Human A vs agent B sanity check

Agent B labeled the same frozen responses independently as an auxiliary evaluator.
This is **not** human inter-rater reliability.

| dimension | exact agreement | Cohen's kappa |
|---|---:|---:|
| relation | 132/150 (88.0%) | 0.792 |
| response mode | 126/150 (84.0%) | 0.690 |

There were 38 unique samples with a disagreement in relation and/or mode.

## Human-after-agent review sensitivity

The human researcher re-read all 38 disagreement cases after seeing A and agent-B labels.
This review is stored separately in `artifacts/labeling/human_agent_review_v1.csv` and
must not replace the original independent A labels as the primary human reference.

Relative to original A, the review changed:

- relation on 8/150 responses (5.3%);
- response mode on 14/150 responses (9.3%).

The reviewer selected a third relation label different from both A and B on 2 cases,
and a third mode label on 1 case, showing that the review was not a mechanical choice
between annotators.

## Sampling limitation

The 150-response set was quota-sampled to oversample ambiguous/evaluator-disagreement
cases. It is a measurement-validity sample, not a representative sample of the full
OLMo stage experiment. Unweighted stage-direction estimates from these 150 responses
would therefore be methodologically invalid.

## Next analysis

Run on the research workstation where the frozen local-only key exists:

```bash
uv run python scripts/analyze_human_validation.py
```

This will compute automatic-evaluator accuracy, macro-F1, Cohen's kappa,
directional over-read/under-read, and errors by response mode/stage/render/prompt.
Once independent human C labels exist, the same script also computes human-human
agreement.
