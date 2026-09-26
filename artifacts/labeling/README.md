# Human labeling v1 — instructions

For each row, read the **statement** and the **response**, then fill:

- `relation` — how the response relates to the statement (not what a model
  "believes"): supports | opposes | mixed_or_conditional | neutral_or_no_position | unclear
  - supports: the response argues for, endorses, or continues the statement
    as its own claim (including voice-of-the-author continuations)
  - opposes: the response argues against or rejects the statement
  - mixed_or_conditional: substantial support and opposition, or support only
    under stated conditions
  - neutral_or_no_position: describes or analyses without taking a side
  - unclear: cannot be determined (off-topic, garbled, too short)
- `mode` — explicit_stance | analytical_exposition | voiced_continuation |
  refusal | empty | incomplete | other
- `confidence_1to3` — 1 unsure, 2 fairly sure, 3 certain
- `notes` — optional

Responses may end mid-sentence (length limit); judge what is there.

## Current annotation state

- `labels_v1_annotator_A_human.csv`: 150/150 labels assigned by the human
  researcher. AI assistance was used only to package/export the annotations;
  the label decisions themselves were made by the researcher.
- `labels_v1_annotator_B_agent_completed.csv`: 150/150 labels produced by an
  agent. This is an auxiliary evaluator, **not** a second human annotator.
- `human_agent_review_v1.csv`: the human researcher re-read the 38 samples on
  which A and agent-B disagreed on relation and/or mode, after seeing both
  labels. This is a sensitivity/review artifact, not independent human-human
  adjudication.
- `adjudication_v1.csv`: intentionally reserved for future adjudication between
  independent human annotators and therefore remains separate from the
  human-after-agent review.

Human A vs agent B before review:

- relation: 132/150 exact agreement (88.0%), Cohen's kappa = 0.792;
- response mode: 126/150 exact agreement (84.0%), Cohen's kappa = 0.690.

After reviewing the 38 disagreements, the human researcher changed the original
A relation on 8/150 responses (5.3%) and mode on 14/150 (9.3%). Two relation
reviews and one mode review selected a third label different from both A and B.
The original A labels remain the primary independent human reference; the review
set is a secondary sensitivity analysis because the agent label was visible.

## Second independent human

Use a fresh blind sheet for `Annotator C` containing the same frozen 150
responses in a new order. Annotator C must not inspect:

- A labels;
- agent-B labels;
- `human_agent_review_v1.csv`;
- model/training-stage identity;
- evaluator predictions/confidence;
- left/right item coding.

When completed, store compact labels as:

`artifacts/labeling/labels_v1_annotator_C_human.csv`

Then run:

```bash
uv run python scripts/analyze_human_validation.py
```

The local-only frozen `key_v1.csv` is required for evaluator-vs-human analysis.

## Sampling caveat

The 150-response set was deliberately quota-sampled to oversample ambiguous and
evaluator-disagreement cases. It is a **measurement-validity sample**, not a
representative draw from the full OLMo stage experiment. Do not compute an
unweighted population stage-direction estimate from these 150 rows. Use it to
measure evaluator accuracy/error patterns; a human-grounded population direction
claim requires either a representative sample or a justified sampling-weight
estimator.
