# Gate A2 — Faulborn stance-classifier reproduction

**Status:** exact reproduction of the released *weights* is **blocked** (the
released checkpoint has no weights file). The released *procedure* was
reconstructed and validated against the authors' own released test split:
a reconstruction following the released data flow reproduces their reported
numbers within seed variation; a leakage-free reconstruction, otherwise
identical, scores about **0.12 macro-F1 lower**.

Source of truth for every number below:
`artifacts/tables/faulborn_classifier_validation.{parquet,csv,md,tex}` (per
classifier and seed) and
`artifacts/tables/faulborn_classifier_validation_by_variant.{...}` (across
seeds). Figure: `figures/generated/faulborn_classifier_validation.{svg,pdf,png}`.

## 1. What was released, and what is missing

Canonical sources: code at `MaFa211/theory_grounded_pol_bias`
(`stance_detector/finetune-bart.py`, `stance_detector/eval.py`); model files
in the "stance detector model files" Google Drive folder linked from the
README; data (incl. the released train/test splits and reported evaluation)
in the data Drive folder. Downloaded 2026-09-23.

The model folder is the step-1750 checkpoint of a HF `Trainer` run on
`facebook/bart-large-mnli` (`BartForSequenceClassification`, NLI head
contradiction/neutral/entailment, transformers 4.37.0.dev0). It contains
`config.json`, tokenizer files, `scheduler.pt`, `rng_state.pth`,
`trainer_state.json`, `training_args.bin` — and **no weights file**
(`model.safetensors` / `pytorch_model.bin`). Verified from the raw Drive
folder listing, not only gdown's; no copy on the HF Hub, in GitHub
releases/LFS, or other branches. SHA-256 of the files present:

| File | SHA-256 |
|---|---|
| config.json | `0044548915945c3657196f1d303205bcea4b0a8c7e9b3b3bdd79856d8c7027b0` |
| trainer_state.json | `a1c2903301b681a2ea3f73ecdc3bfd33d846bb93b7b236f11a6313e365c83925` |
| training_args.bin | `870a7a564bfaa8d71911b661b45e7aa70db7e740ef6a1520f88fe4704516820b` |
| tokenizer.json | `4219589aab2c08e1e5e9a077d313df9cd8aaee168f5f170e5d394a4deec3e3b5` |
| vocab.json | `ed19656ea1707df69134c4af35c8ceda2cc9860bf2c3495026153a133670ab5e` |
| merges.txt | `1ce1664773c50f3e0cc8842619a93edc4624525b728b188a9e0be33b7726adc5` |
| scheduler.pt | `d9200ec75a799bc795ed590c72f57ab5edd8dc066e889096176ce1d6c3c7b2f9` |
| rng_state.pth | `50c3f5874891f9c201be40eddfa59729c69373737e94167b11b0fe1b9f66da84` |

An optional, non-blocking request to the authors is logged in
`docs/HUMAN_ACTION_REQUIRED.md` (#2).

## 2. The released procedure (reproduced exactly)

From `eval.py`: HF `zero-shot-classification` pipeline; hypothesis template
`"The stance of the statement is {}"`; candidate labels
`['disagree', 'agree', 'neutral', 'unrelated']`; premise = the **response
text alone** (the classifier never sees the proposition); label = argmax,
confidence = max score. Implemented in
`src/anatomiae/evaluators/faulborn_nli.py`. Their `num_labels=4` call
argument is omitted (no effect on the pipeline's output; rejected by current
transformers). Label mapping onto anatomiae's taxonomy: agree→agreement,
disagree→disagreement, neutral→neutral_or_no_position, unrelated→irrelevant
(native labels are kept alongside).

## 3. Discrepancies found inside the released materials

1. **Released script vs. released checkpoint:** `finetune-bart.py` sets
   `max_steps=1750`; the released `trainer_state.json` records
   `max_steps=2160` (at `global_step=1750`). The released script is not
   exactly the run that produced the released checkpoint.
2. **Released script vs. released data file:** the script assigns two column
   names to its training CSV; the released `all_labels.csv` has three
   (`text, labels, prompt`), so the script does not run on the released file
   as-is.
3. **The saved split does not reproduce** from the released CSV (shuffle
   seed 42, `datasets` 5.0.1): neither the exact order nor the set of test
   texts matches the released `test_set`.
4. **Six texts appear in both released splits** — bare Likert digits
   (`'1'`, `'2'`, `'4'`, `'5'`), `'nan'`, `'User '` — and `'1'` carries
   inconsistent gold labels (unrelated and disagree) for identical text.
5. **Data flow, as written:** the script saves a 1,056/264 train/test split
   to disk, but then builds NLI training pairs from the **full shuffled
   1,320-response set** and re-splits those pairs 80/20. Responses from the
   saved test split therefore contribute training pairs, and the saved test
   split is what `eval.py` reports on.

Because of (1) and (3), whether the authors' actual run had the data flow in
(5) cannot be verified from the released materials alone. Section 5 measures
what that data flow does instead of asserting it.

## 4. Reconstruction

`scripts/train_faulborn_classifier.py`, identical to `finetune-bart.py` in
base model, NLI formulation, one entailment + one random-contradiction pair
per response, and hyperparameters (lr 2e-5, max_steps 1750, batch 4, warmup
500, weight decay 0.2, fp32). Deviations: dynamic padding (attention-masked,
same outputs); a seeded contradiction-label RNG (theirs is unseeded, hence
irreproducible); no wandb/intermediate evaluation. Two variants, each with
**2,112 training pairs** by construction:

| Variant | Data flow | Training pairs whose premise is a released-test response |
|---|---|---|
| `as_written` | pairs from all 1,320 responses, pair-level 80/20 re-split (released script's flow) | 459 |
| `leakage_free` | pairs from the released 1,056-row train split only | 42 (only the six duplicated trivial texts in §3.4) |

Three training seeds each (42, 43, 44); ~193 s per run on physical GPU 1 at
its 250 W minimum power limit. Weights and a provenance JSON per run under
`/data/jackb/anatomiae/models/faulborn_classifier_reconstruction/` (outside
git). These are anatomiae reconstructions, never the authors' weights.

## 5. Validation on the released test split

Macro-F1 (scikit-learn semantics) on the released 264-response test split,
at the same confidence thresholds as `eval.py`:

| Classifier | conf ≥ 0.0 | conf ≥ 0.9 | Faulborn reported (0.0 / 0.9) |
|---|---|---|---|
| zero-shot bart-large-mnli (exact, public weights) | 0.305 (N=264) | 0.588 (N=35) | 0.269 (N=264) / 0.449 (N=47) |
| `as_written` reconstruction, 3 seeds | **0.875** [0.871–0.879] | **0.927** [0.907–0.948] (N≈211) | 0.873 / 0.934 (N=182) |
| `leakage_free` reconstruction, 3 seeds | **0.749** [0.743–0.753] | **0.832** [0.823–0.839] (N≈199) | 0.873 / 0.934 |

Brackets: min–max across the three seeds. Per-seed rows, with bootstrap 95%
CIs over test items, are in `faulborn_classifier_validation.parquet`.

**Reading:**
- The `as_written` reconstruction reproduces the reported fine-tuned
  numbers: the reported values fall inside the three-seed range at both
  thresholds. This is the evidence that the reconstruction faithfully
  implements the released procedure.
- The `leakage_free` reconstruction, differing *only* in whether test
  responses contribute training pairs, scores ~0.12 lower at conf ≥ 0 and
  ~0.10 lower at conf ≥ 0.9; seed ranges do not overlap with the reported
  values or with `as_written`.
- This is **consistent with** the reported classifier performance being
  inflated by train/test contamination via the data flow in §3.5. It is not
  proof that the authors' run had that flow (§3.1, §3.3).
- The zero-shot baseline matches at conf ≥ 0 (0.305 vs 0.269; their value
  lies inside our item-bootstrap CI 0.260–0.351) but not on the
  high-confidence subset (N=35 vs 47) — plausibly confidence-calibration
  differences between transformers 4.37 and 4.57.

## 6. Consequence for anatomiae

- The Faulborn-procedure classifier is used as an **evaluator whose error
  rate is known to be substantial**: held-out macro-F1 ≈ 0.75 over all
  responses (≈ 0.83 on the confidence-filtered subset), not ≈ 0.87 / 0.93.
  No result is reported with this classifier as ground truth.
- Evaluation uses the `as_written` reconstruction as the primary "Faulborn
  classifier" (it is the one that reproduces their behavior), with the
  `leakage_free` and zero-shot variants alongside. Our pilot responses are
  in neither variant's training data.
- Gate A2 status: **validated reconstruction; exact released-weights
  reproduction blocked**.
