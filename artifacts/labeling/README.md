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
