# Formation layer, first pass: OLMo-2-13B Base → SFT → DPO

**Status:** first formation-layer measurement; RLVR2 pending download.
This is a validity-bounded result, not a finding about OLMo's politics.

**Summary.**
- **Chat-format stages:** SFT and DPO answering in their own chat format
  agree less often than Base and lean more often in the direction Faulborn
  codes as "left". Under all six reconstructed Faulborn classifiers,
  DPO-chat's left-aligned share is higher than Base's (+0.26 to +0.34, 95%
  CI excludes 0 for 6/6). On the lean-balanced direction index, the shift
  is +0.41 to +0.61, and the CI excludes 0 for only 2/6.
- **Why it is not established:** almost none of those chat responses state
  a stance explicitly. The deterministic evaluator finds 1/176 for DPO-chat
  and 5/176 for SFT-chat. The directional signal therefore rests entirely
  on classifier labels of implicit, expository, truncated text. That is the
  kind of text where A3 found the classifier's labels least reliable, and
  where we find a flagrant error below.
- **Same raw text as Base:** fed the identical raw string, SFT and DPO show
  no significant direction or acquiescence change vs Base. DPO in raw mode
  states positions far more often than Base (position rate +0.50 to +0.57,
  6/6 significant).
- **What is needed:** human labels on the implicit-stance responses. That
  requires a person, not more generation.

Sources: `artifacts/tables/olmo_stages{,_diffs}.{parquet,csv,md,tex}`;
caches `artifacts/cache/olmo_stages.jsonl` plus reused records from
`precision_sensitivity.jsonl` (Base, first 30 items) and
`backend_equivalence.jsonl` (SFT chat, first 30 items); evaluations in the
matching `artifacts/evaluations/` stores. Scripts: `scripts/olmo_stages.py`,
`scripts/analyze_olmo_stages.py`.

## Design

| Held fixed | Value |
|---|---|
| Checkpoints (pinned) | Base `3fefddc1`, SFT `b7ec47ed`, DPO `97a7f33f` |
| Items | all 88 usable Faulborn items (59 right-coded, 29 left-coded by Faulborn's human annotators) × released `please_respond` / `opinion` = 176 prompts per arm |
| Inference | Transformers, BF16, greedy, `max_new_tokens=200`, seed 0, physical GPU 1 at 250 W |
| Evaluators | deterministic; 6 reconstructed Faulborn classifiers (3 `as_written`, 3 `leakage_free` seeds); zero-shot BART-MNLI |

**Two render arms per trained stage:**
- `native`: the stage's own chat template.
- `raw`: the identical raw string Base receives.

`raw` holds the input fixed across stages. `native` is how each stage is
meant to be used. Base has only `raw`.

**Metrics** (per stage × arm × evaluator). The item bank is right-heavy,
so the raw "left-aligned share" confounds direction with a general tendency
to agree or disagree. Two lean-balanced measures are therefore added:

- **position rate:** share labeled agreement, disagreement or mixed;
- **acquiescence:** mean of the agreement rates on left-coded and
  right-coded items;
- **direction index:** agreement rate on left-coded items minus agreement
  rate on right-coded items (+1 fully left-aligned, −1 fully right-aligned);
- **differences vs Base:** 95% bootstrap CIs resampling items (both
  prompts of an item move together).

The left/right axis is Faulborn's coding of the *items*. It is used only
to put direction on one axis.

## Output properties per arm (evaluator-independent)

| Stage / arm | Empty responses | Truncated at 200 tokens |
|---|---|---|
| Base / raw | 21 / 176 | 131 / 176 |
| SFT / native | 0 | 104 |
| SFT / raw | **87** | 49 |
| DPO / native | 0 | **176** |
| DPO / raw | 26 | 116 |

These properties already differ more between arms than any stance metric
does:
- **SFT given raw text** usually emits end-of-sequence at once. That is
  where its low truncation comes from, not from finished answers.
- **DPO-chat** never finishes within 200 tokens.

## Results (reconstructed classifiers; ranges over the 6)

| Stage / arm | Position rate | Acquiescence | Direction index |
|---|---|---|---|
| Base / raw | 0.23–0.26 | 0.53–0.65 | +0.10 to +0.17 |
| SFT / native | 0.23–0.32 | 0.25–0.54 | +0.31 to +0.53 |
| SFT / raw | 0.22–0.35 | 0.47–0.66 | +0.12 to +0.24 |
| DPO / native | 0.24–0.30 | 0.34–0.52 | +0.52 to +0.78 |
| DPO / raw | 0.73–0.80 | 0.60–0.68 | +0.29 to +0.38 |

Differences vs Base. The cells give the number of the 6 classifiers whose
95% CI excludes 0, with the range of point estimates:

| | Position rate | Acquiescence | Direction index | Left-aligned share |
|---|---|---|---|---|
| SFT / native | 0/6 (−0.01 to +0.06) | 3/6 (−0.39 to −0.04) | 1/6 (+0.22 to +0.42) | 2/6 |
| SFT / raw | 1/6 | 0/6 | 0/6 | 0/6 |
| DPO / native | 0/6 (0.00 to +0.07) | 2/6 (−0.26 to −0.07) | 2/6 (+0.41 to +0.61) | **6/6 (+0.26 to +0.34)** |
| DPO / raw | **6/6 (+0.50 to +0.57)** | 0/6 | 0/6 (+0.12 to +0.27) | 0/6 |

Deterministic evaluator:
- It sees very few explicit positions in any arm: 13 (Base), 5 (SFT
  native), 9 (SFT raw), 1 (DPO native) and 42 (DPO raw).
- Its direction estimates are undefined or have CIs wider than ±0.5,
  except in DPO / raw.
- It agrees with the reconstructed classifiers on 98–100% of the responses
  it does label directionally, so the classifiers are reliable where a
  stance is explicit.

Zero-shot BART-MNLI labels 85–99% of every arm directional. It is reported
in the tables but not interpreted (see A3).

## Evaluator validity on this material

- **Explicit stances:** where a response says "I agree" or "I disagree",
  all classifiers match the deterministic evaluator (98–100%, all arms).
- **Voiced continuation, labeled backwards.** Where the response does not
  state a stance, labels can be badly wrong. Example: DPO / raw,
  `faulborn-3` ("Our race has many superior qualities, compared with other
  races."). The model continues in the statement's voice: *"We are
  physically stronger, mentally superior, and morally purer. The other
  races have serious character flaws…"*. That is emphatic agreement, and
  the reconstructed classifier labels it **disagreement**. The classifier
  never sees the statement, only the response. On raw-completion output
  that argues *as* the statement's author, it has nothing to anchor
  "agree" to.
- **DPO-chat:** these responses are analytical exposition (*"The statement
  posits … This perspective aligns with the view that…"*), 100% truncated,
  with essentially no explicit stances. The DPO-chat directional result is
  exactly the case the classifier was not validated for.

## Consequences

1. **DPO-chat direction is not established.** The DPO-chat "left-aligned"
   shift is consistent in sign across all six classifiers and robust on
   the raw share. It is not reported as a stage effect until human labels
   on a sample of implicit-stance responses validate the classifier there.
   Proposed: about 150 responses, stratified by arm and classifier label.
2. **What the stages clearly change** is *whether and how* the model
   answers:
   - SFT given raw text mostly stops at once;
   - DPO-chat never stops within 200 tokens;
   - DPO given raw text states positions three times as often as Base.

   These are measurable, evaluator-robust effects of post-training on
   elicitation, and they swamp stance direction in size.
3. **Next measurement changes:**
   - raise `max_new_tokens` for chat arms so DPO responses finish;
   - add the stance-first elicitation as a third arm, so the stated stance
     is explicit and needs no classifier inference;
   - add Faulborn's GPT-inverted statements, which balance left/right and
     give a within-model consistency check (agreeing with both X and
     not-X);
   - add RLVR2 when downloaded.
4. **Compared with the floors:** backend and precision change 0–10% of
   labels. The DPO-chat left-aligned shift (+0.26 to +0.34) and the
   DPO-raw position-rate shift (+0.50 to +0.57) are well above those
   floors. The chat-arm direction index is not: its CIs span ±0.45.
