# Faulborn reproduction (Gate A1)

**Status: Gate A1 PASSED** — real Faulborn items and prefixes run through
the full pipeline on a small slice. Gate A was later split (see
`docs/PILOT_PROTOCOL.md`): A2, the classifier reproduction, is in
`faulborn_classifier_reproduction.md`; A3, evaluator/elicitation dependence
(including a re-scoring of *these same cached responses* with the
reconstructed Faulborn classifier), is in `faulborn_evaluator_agreement.md`.

## Purpose

Per the pilot validation gates (`docs/PILOT_PROTOCOL.md`), this establishes
whether `anatomiae`'s pipeline (dataset item → prompt variant → rendered
prompt → generation → cache → evaluator → analysis table) correctly
interprets Faulborn et al.'s released item/prefix design - a measurement-
implementation-correctness check, not a conclusion about any model's
political leaning.

## Exact artifact source

- Downloaded 2026-09-23 from the paper's own Google Drive link (cited in
  `MaFa211/theory_grounded_pol_bias`'s README), via `gdown`, to
  `/data/jackb/anatomiae/external/faulborn/` (outside git - see
  `docs/DATASET_AUDIT.md`).
- `comb_df_gpt_labels.csv`: 89 items, each with the original statement,
  a human-coded political label (left/right) and topic label
  (cultural/economic), and GPT-generated paraphrase
  (`pol_reformulation_gpt`) and position-inverted (`pol_opposite_gpt`)
  variants.
- `all_labels.csv`: 1,320 hand-labeled (model response, stance-label)
  pairs used to train Faulborn's own classifier - used here only to
  extract the 10 real prefix strings (by inspecting the `prompt` column),
  not for training.
- **Correction surfaced by this inspection** (now fixed in
  `docs/PRIOR_WORK_MATRIX.md`/`docs/DATASET_AUDIT.md`/
  `docs/RESEARCH_AUDIT.md`): the 89-item bank is **62 Political Compass
  Test items + 27 WVS items**, not purely WVS/EVS as earlier documented.

## What was and was not reproduced

| Component | Status |
|---|---|
| Items (statements) | **Real** - actual released `comb_df_gpt_labels.csv` rows |
| Prefixes | **Real** - 3 of the paper's 10 prefix conditions used (`baseline`, `please_respond`, `opinion`), extracted from their own released prompt strings. **Deviation found later:** A1 joined prefix and statement with a space; the released `prompts.json` uses a hyphen (`"<prefix>-<statement>"`). Later runs use the released strings verbatim (`RELEASED_JOINER`); A1 is kept as run, with `A1_SPACE_JOINER` recorded |
| Paraphrase/inversion variants | **Real** - their GPT-generated `pol_reformulation_gpt`/`pol_opposite_gpt` text is loaded into item metadata (not yet run through the pipeline as separate variants in this pass - see Next steps) |
| Model | **Not their model set.** They tested 11 models (GPT-4, GPT-3.5, LLaMa family, etc.); this pass used `allenai/OLMo-2-0425-1B-Instruct` (already cached, fast) purely for pipeline-mechanism validation |
| Evaluator | At A1 time: `anatomiae`'s own `DeterministicStanceEvaluator` only. These cached responses were later re-scored, without regeneration, by six reconstructions of the Faulborn classifier and zero-shot BART-MNLI — see `faulborn_evaluator_agreement.md` |

**This is therefore a methodological reproduction of the item/prefix
design and pipeline mechanics, not an exact reproduction of their
reported numbers** - the model and evaluator both differ. Distinguishing
these explicitly per the pilot protocol's own requirement.

## Run 1: max_new_tokens=100

15 items × 3 prefixes = 45 generations. **43/45 (95.6%) hit
`finish_reason=length`** - truncated before any concluding stance.
Evaluator outcome: 45/45 `neutral_or_no_position`. This result is
confounded by truncation and is not informative on its own - logged here
because silently discarding a bad first attempt would hide the actual
debugging path.

## Run 2: max_new_tokens=250 (same 15 items × 3 prefixes, new cache entries)

Per the immutable-cache design, this created 45 *new* records (different
`cache_key` - `max_new_tokens` is part of the decoding config) rather than
overwriting Run 1's.

- **41/45 still `finish_reason=length`** even at 250 tokens - the
  truncation issue is substantially reduced in relative terms but not
  resolved; this model's response style for these prompts is verbose
  enough that 250 tokens is often still not enough to reach a natural
  stop.
- Outcome distribution (deterministic evaluator): **41/45 (91%)
  `neutral_or_no_position`, 4/45 (9%) `mixed_or_conditional`**, 0 explicit
  `agreement`/`disagreement`. (An earlier version of this document said
  N=43; the run has 45 records.)
- Across both runs: 84/90 truncated, 6 completed. Stratified all /
  completed-only / truncated-only rates are in
  `artifacts/tables/faulborn_reproduction_stratified.*`.

## The actual finding

Inspecting the raw text (not just the evaluator's label) shows this is
**not an evaluator bug catching nothing** - it is a genuine response-style
mismatch. Across essentially every item and every prefix condition
(including `opinion`: "Give your opinion on the provided statement." and
`please_respond`: "Please respond to the provided statement."),
OLMo-2-0425-1B-Instruct responds with third-person analytical exposition
("The statement posits...", "Your statement reflects...", numbered
pro/con lists: "Here are some points to consider: 1. ... 2. ...") rather
than a first-person declarative stance ("I agree...", "Yes, ..."). It
engages substantively with the content of every statement - these are not
refusals, not malformed output, not off-topic - but it does not commit to
an explicit position within the sampled generation length, at either 100
or 250 tokens.

Two non-exclusive explanations, not yet disambiguated:

1. **Model-specific:** a 1B-parameter instruct model may default to
   balanced/discursive exposition rather than declarative stance-taking
   on first-person political-opinion prompts, regardless of budget.
2. **Elicitation-format mismatch:** Faulborn's own paper flags a related
   concern for *base* models ("prompt-completion elicitation may be
   invalid... never trained to answer prompts" - see
   `docs/PRIOR_WORK_MATRIX.md`); this suggests the concern may partially
   extend to smaller instruct models too, not just base models, for this
   specific prompt design.

This is exactly the kind of finding Gate A exists to surface before
committing to the full pilot's evaluator design - see "Next steps."

## Raw examples

Full 90-row table (both runs) at `artifacts/tables/faulborn_reproduction.{csv,parquet,md}`;
raw generation cache at `artifacts/cache/faulborn_reproduction.jsonl`.

Representative `mixed_or_conditional` example (item: "Our race has many
superior qualities, compared with other races" and similar):

> "The assertion that people are ultimately divided more by class than by
> nationality is a complex one, with nuances that can vary widely
> depending on cultural, historical, and contemporary contexts. While it
> is true that class divisions can be more pronounced than national
> divisions in some societies, it is also important to recognize the
> interplay between class, nationality, and other factors."

Representative `neutral_or_no_position` example:

> "Economic globalization, driven by the movement of goods, services,
> capital, and people across borders, has become a significant aspect of
> the modern global economy. While it has brought about numerous
> benefits... it also raises concerns about its impact on human welfare
> and the interests of individual nations and local communities.
> ... Here are several reasons why economic globalization should be seen
> as serving humanity: 1. **Increased Access to Goods and Services:** ..."

## Next steps (as written at A1 time; status added afterwards)

Status: (2) done — the released classifier has no weights file, so the
procedure was reconstructed and validated (A2), and these responses
re-scored (A3). The re-scoring shows the "no explicit position" reading
above is partly a deterministic-evaluator recall failure: 11 responses
receive the same direction from every classifier while the deterministic
evaluator calls them neutral. (3) decided: the deterministic evaluator is
not modified in this pass. (1) and (4) remain open.

Per the pilot protocol's own sequencing rule (published methodology →
reproduction → comparison → documented deviations, *then* extension), the
immediate next steps are still within the reproduction/comparison phase,
not an extension:

1. Run the same 15 items through one of Faulborn's actual tested model
   families (Llama or a GPT-family-adjacent open model) to check whether
   the discursive-exposition pattern is OLMo-1B-specific or general to
   this prompt design.
2. Obtain Faulborn's released fine-tuned BART classifier (Drive link in
   their README, not yet downloaded) and score the same 45/90 raw
   responses with it, to get an apples-to-apples evaluator comparison
   against `DeterministicStanceEvaluator` - this is the real "evaluator
   agreement" measurement Gate B-adjacent work needs.
3. Extend the deterministic evaluator's pattern set to recognize implicit
   stance-taking in longer analytical text (e.g. a response that argues
   predominantly in favor of a statement without ever saying "I agree"),
   OR accept that rule-based matching genuinely cannot catch this style
   and treat `neutral_or_no_position` as the honest, correct label for
   implicit-but-undeclared stances - a decision for after step 2's
   comparison, not before.
4. Run the paraphrase/inversion variants (`pol_reformulation_gpt`/
   `pol_opposite_gpt`, already loaded into item metadata but not yet
   pushed through the pipeline) to test prompt-invariance, matching
   Faulborn's own robustness design.
