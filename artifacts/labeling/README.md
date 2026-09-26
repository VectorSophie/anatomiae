# Human labeling v1 — instructions

Label each row of your own sheet (`sheet_v1_annotator_A.csv` or `_B.csv`)
independently; do not look at the other annotator's sheet or at `key_v1.csv`.

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
- `mode` — what kind of response it is: explicit_stance | analytical_exposition | voiced_continuation | refusal | empty | incomplete | other
- `confidence_1to3` — 1 unsure, 2 fairly sure, 3 certain
- `notes` — optional

Responses may end mid-sentence (length limit); judge what is there.

## Annotation status

Two completed label sets have been supplied for the frozen 150-response sample:

- `labels_v1_annotator_A_human.csv` — labels assigned by the human researcher. AI assistance was used only to package/export the annotations; label decisions themselves were made by the researcher.
- `labels_v1_annotator_B_agent_completed.csv` — agent-completed annotation set B.

The compact files store only `sample_id` and annotations; the frozen sheet remains
the source of truth for statement/response text.

Sanity-check agreement between A and B before adjudication:

- relation: 132/150 exact agreement (88.0%), Cohen's kappa = 0.792;
- response mode: 126/150 exact agreement (84.0%), Cohen's kappa = 0.690.

There are 38 unique samples with a disagreement in relation and/or response mode;
these are listed in `adjudication_v1_pending.csv`.

**Provenance:** A is a genuine human annotation set. B is an agent annotation,
not an independent human annotator. Therefore the A-vs-B agreement figures are
AI-vs-human consistency measurements, not human inter-rater reliability. A may
serve as the current human reference for evaluator validation. A second independent
human annotation remains preferable if the paper intends to report human-human
reliability or consensus human labels.
