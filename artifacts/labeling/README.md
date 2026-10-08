# Human labeling v1 — instructions

For each row, read the **statement** and the **response**, then fill:

- `relation` — how the response relates to the statement (not what a model
  "believes"): supports | opposes | mixed_or_conditional | neutral_or_no_position | unclear
  - supports: the response argues for, endorses, or continues the statement as its own claim
  - opposes: the response argues against or rejects the statement
  - mixed_or_conditional: substantial support and opposition, or support only under conditions
  - neutral_or_no_position: describes or analyses without taking a side
  - unclear: cannot be determined
- `mode` — explicit_stance | analytical_exposition | voiced_continuation |
  refusal | empty | incomplete | other
- `confidence_1to3` — 1 unsure, 2 fairly sure, 3 certain
- `notes` — optional

Responses may end mid-sentence; judge what is present.

## Current annotation state

- `labels_v1_annotator_A_human.csv`: human A, complete 150/150.
- `labels_v1_annotator_B_agent_completed.csv`: agent evaluator B, complete 150/150.
- `labels_v1_annotator_C_human.csv`: independent human C, complete 150/150.
- `human_agent_review_v1.csv`: A re-read the 38 A-vs-agent disagreements after seeing both labels;
  sensitivity artifact only.
- `human_AC_adjudication_pending.csv`: 61 unique A↔C disagreement cases awaiting adjudication.
- `adjudication_v1.csv`: legacy empty template; do not treat as completed consensus labels.

## Human A vs Human C

- relation: 105/150 exact agreement (70.0%), Cohen's kappa = 0.552;
- response mode: 115/150 exact agreement (76.7%), Cohen's kappa = 0.564;
- among 100 cases both humans called directional, direction itself agrees on 97/100 (97.0%);
- only 3 cases are direct `supports` ↔ `opposes` reversals.

The dominant disagreement is whether a response is sufficiently committal to count as
directional versus mixed/conditional or neutral, not which direction it takes.

### Annotator C protocol note

A first C attempt was excluded before accepting any human-human reliability result because
the annotator misunderstood the task (including impossible `empty` labels on non-empty
responses and personal answers to the proposition). The retained C set is a complete blind
rerun with clarified French instructions. Statements were shown in French with the original
English available as a reference; model responses remained in original English.

## Human A vs agent B

- relation: 132/150 exact agreement (88.0%), Cohen's kappa = 0.792;
- response mode: 126/150 exact agreement (84.0%), Cohen's kappa = 0.690.

This is human-vs-agent consistency, **not** human inter-rater reliability.

## Analysis

On the workstation holding the local-only frozen `key_v1.csv`:

```bash
uv run python scripts/analyze_human_validation.py
```

## Sampling caveat

The 150-response set deliberately oversamples ambiguous / evaluator-disagreement cases.
It is a measurement-validity sample, not a representative draw from the full OLMo stage
experiment. Do not use its unweighted stage composition as a population direction estimate.
