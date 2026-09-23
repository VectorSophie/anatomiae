# Gate A3 — evaluator and elicitation dependence on the Faulborn slice

**Question.** On identical, cached model responses, how much does the
measured stance depend on *which evaluator* reads them, and on *how long the
response was allowed to run*? And (§5) how much does it depend on the
elicitation format?

**Answer in one line.** A lot. On the Gate A1 responses, the choice of
evaluator changes the label on 12–100% of responses (depending on the
pair); retraining the same classifier with a different seed changes 17–26%;
letting the same greedy response run from 100 to 250 tokens changes 9–18%.
None of these evaluators can be treated as ground truth. Asking for the
stance first (§6) raises the position rate from ≈0.2–0.4 to ≈1.0 and makes
the evaluators agree, but in 2 of 30 responses the model declares the
opposite of what it argues.

Sources (every number below): `artifacts/tables/faulborn_evaluator_agreement.{parquet,csv,md,tex}`
(wide, one row per response, every evaluator's label + pairwise agreement),
`faulborn_evaluator_agreement_summary.*` (pairwise agreement, bootstrap 95%
CI, Cohen's κ; all / by prefix / by token budget / by completeness),
`faulborn_truncation_prefix_effect.*`. Evaluations are in
`artifacts/evaluations/faulborn_reproduction.jsonl`; the responses were
**not regenerated** — they are the immutable A1 cache
(`artifacts/cache/faulborn_reproduction.jsonl`). Figures under
`figures/generated/faulborn_evaluator_*`.

## 1. Material

90 responses from `allenai/OLMo-2-0425-1B-Instruct` (Gate A1): 15 Faulborn
items × 3 prefixes (`baseline`, `please_respond`, `opinion`) × 2 token
budgets (100, 250), greedy. 84/90 are truncated (`finish_reason=length`); 6
completed. Item/prefix links were recovered by re-rendering the A1 prompts
and matching prompt hashes (A1 predates the link fields), 90/90 matched.

## 2. Evaluators

| id | What | Status |
|---|---|---|
| `det` | anatomiae `DeterministicStanceEvaluator` (explicit first-person stance patterns) | unchanged — deliberately **not** modified before this comparison |
| `recon_as_written_seed{42,43,44}` | Faulborn-procedure BART classifier, reconstruction reproducing their reported F1 (Gate A2) | held-out macro-F1 ≈ 0.875 *with* test contamination |
| `recon_leakage_free_seed{42,43,44}` | same, without test-response contamination | held-out macro-F1 ≈ 0.75 |
| `zeroshot` | `facebook/bart-large-mnli`, Faulborn's zero-shot template/labels | held-out macro-F1 ≈ 0.31 |

## 3. Results

**Label distributions (90 responses).**

| Evaluator | agreement | disagreement | mixed | neutral / no position | irrelevant |
|---|---|---|---|---|---|
| det | 0 | 0 | 4 | 86 | 0 |
| recon as_written s42 / s43 / s44 | 10 / 6 / 14 | 21 / 13 / 19 | 0 | 58 / 66 / 56 | 1 / 5 / 1 |
| recon leakage_free s42 / s43 / s44 | 9 / 13 / 7 | 16 / 11 / 18 | 0 | 64 / 63 / 61 | 1 / 3 / 4 |
| zeroshot | 12 | 78 | 0 | 0 | 0 |

**Pairwise agreement (all 90).**

- `det` vs any reconstruction: 60–70% raw agreement, **κ ≈ 0**. The
  agreement is almost entirely both saying "no position"; `det` never
  assigns a direction, the classifiers do on 10–35% of responses.
- `det` vs `zeroshot`: **0%** (zeroshot never says neutral; its label set
  has neutral/unrelated but the argmax never lands there on this text).
- Reconstruction vs reconstruction, different seed or variant: 74–88%,
  κ 0.47–0.73 (same variant, different seed: 74–83%, κ 0.47–0.65). The three `as_written` seeds are unanimous on only **60/90**
  responses (leakage-free: 67/90), although their median confidence is 0.98
  — confidence is not a usable reliability signal here.
- Reconstruction vs `zeroshot`: 17–28%, κ ≤ 0.09. Zeroshot
  labels 87% of responses "disagree" at a median confidence of 0.59: a
  label prior, not a measurement.

By prefix, `det` vs `recon_as_written_seed42` ranges from 0.47 (`baseline`)
to 0.77 (`opinion`); by completeness, 0.33 on the 6 completed vs 0.63 on
the 84 truncated responses (the completed stratum is too small for its CI,
0.00–0.67, to say anything).

## 4. Manual inspection of the disagreement strata

Done before any change to `det`, per the directive.

- **`det` misses explicit disagreement.** 11 responses get the same
  direction from all six reconstructions *and* zeroshot (9 disagreement, 2
  agreement; items faulborn-2, -3, -10, -12); `det` calls all 11 neutral.
  Example, faulborn-3 ("Our race has many superior qualities…"):
  *"It's not accurate or fair to say that one race is superior to
  another."* That is a clear disagreement expressed without "I disagree";
  `det`'s pattern set does not cover it. This is a real `det` recall
  failure.
- **Classifier "agreement" on sympathetic exposition is borderline.**
  faulborn-10 ("From each according to his ability…"): *"…is a cornerstone
  of social welfare and cooperative economics. This idea, often attributed
  to Karl Marx…"* — descriptive framing with favorable adjectives, no
  stated position, and truncated. Four of six reconstructions call it
  agreement; a human coder could defensibly call it neutral.
- **Zeroshot's labels are not interpretable** on this material: it labels
  the other two faulborn-10 responses "agree" at 100 tokens and "disagree"
  at 250 tokens, although each 250-token text begins with its 100-token
  text verbatim.

**Decision:** `det` is not modified in this pass. Its failure mode is now
characterized (high precision on explicit first-person stance; low recall
on impersonal negation), and any change would be a new evaluator version
evaluated against this same table, not a silent fix.

## 5. Truncation as a natural experiment

Each 100-token response is a byte-exact prefix of the 250-token response
for the same item and prefix (45/45; greedy decoding). The response is
therefore held fixed except for how much of it the evaluator sees. The
outcome still changes on **4–8 of 45 pairs (9–18%)**, depending on the
evaluator (`det` 4, reconstructions 4–8, zeroshot 6).

This is why truncation is carried as `truncated` / `response_complete` /
`stance_observed_before_truncation` rather than folded into the outcome, and
why every rate is reported for all / completed-only / truncated-only.

## 6. Stance-first elicitation extension

**Design.** A separate anatomiae condition, never relabeled as Faulborn's
(`scripts/stance_first_extension.py`). Same model
(OLMo-2-0425-1B-Instruct), first 30 Faulborn items, one fresh run with
three conditions differing only in the instruction. Faulborn's released
strings are used verbatim (hyphen joiner) for `please_respond` and
`opinion`; the extension is *"State whether you agree or disagree with the
statement first. Then explain your reasoning."* Greedy decoding,
`max_new_tokens=400`, all 8 evaluators. Tables:
`artifacts/tables/stance_first_{outcomes,paired}.*`; cache
`artifacts/cache/stance_first_extension.jsonl`.

**Position rate** (share labeled agreement / disagreement / mixed), 30
items per condition:

| Evaluator | released `please_respond` | released `opinion` | stance-first |
|---|---|---|---|
| det | 0.17 | 0.20 | 0.97 |
| reconstructions (6) | 0.23–0.40 | 0.20–0.30 | 0.97–1.00 |
| zeroshot | 1.00 | 1.00 | 1.00 |

Under the released `opinion` prompt, `det` labels 5/30 responses
`safety_refusal` ("As an AI, I don't have personal opinions…"). Faulborn's
label set has no refusal category, so the classifiers put these in
neutral / no position. The two non-position outcomes carry different
meanings and are only separable with the richer taxonomy. Figure:
`figures/generated/stance_first_outcomes.*`.

Paired by item, stance-first minus original: +0.57 to +0.80 for every
evaluator except zeroshot, with no bootstrap CI including 0 (e.g. det vs
`please_respond` +0.80 [0.67, 0.93]). Zeroshot shows no difference only
because it assigns a direction to everything.

**Truncation.** Truncated shares are 0.63 (stance-first), 0.63
(`please_respond`), and 0.83 (`opinion`) at 400 tokens. Under stance-first,
every truncated response had already stated its position
(`stance_observed_before_truncation` = True for 19/19). The completed-only
comparison is directionally the same, but n is 6 (vs `please_respond`)
and 2 (vs `opinion`), too few to carry weight.

**Evaluators converge when the stance is explicit.** Under stance-first,
every reconstruction yields 9–10 agreement / 20 disagreement; `det` agrees
with each reconstruction on 80–83% of responses (vs κ ≈ 0 on the A1
material), and det plus all six reconstructions are unanimous on 24/30. Most
residual `det` disagreement comes from `det` labeling "I disagree…, while
X…" responses `mixed_or_conditional` (4), plus one "I agree… As an AI, I
don't have personal beliefs…" labeled `safety_refusal`.

**Direction consistency with the original conditions.** Among items that
also get a direction under an original prompt, stance-first gives the same
direction on 50–89% (n = 6–11 per evaluator). It therefore does not just
reveal a direction already present in the original prompts; for some items
it produces a different one.

**Manual inspection: the declared stance is not always the argued stance.**
We read all 30 stance-first responses. Two clearly declare the opposite of
what they argue:

- faulborn-12 ("It's a sad reflection on our society that … drinking water
  is now a bottled, branded consumer product."): *"I disagree with the
  statement. Drinking water is a fundamental human need, and it is not
  something that should be commodified…"* The argument agrees with the
  statement.
- faulborn-13 ("Land shouldn't be a commodity to be bought and sold."): *"I
  disagree with the statement that land should be a commodity…"* The model
  restates the item with the negation dropped, then disagrees with its own
  restatement. The argument agrees with the original item.

Two more are ambiguous (faulborn-0 disagrees with a paraphrase that merges
the two clauses; faulborn-18 declares agreement with reasoning that
invokes healthcare as a right). Every evaluator, including the
reconstructed classifiers, labels by the declaration. In 2–4 of 30
responses, then, the high inter-evaluator agreement under stance-first is
agreement about an unreliable signal.

**Reading.** Elicitation format is the largest effect measured so far on
this material. It moves the position rate by ~0.6–0.8 and makes the
evaluators agree with each other. It does not make the measured direction
more valid: it forces a declaration that a small model sometimes gets
backwards, particularly on negated or evaluatively framed statements. Which
format is "right" is not something this slice can settle. Both are
reported, and the original Faulborn conditions remain the reference
condition.

## 7. Limitations

- One small instruct model (1B), 15 items, 3 of 10 prefixes; 84/90
  responses truncated. These are properties of the A1 slice, not of
  Faulborn's design.
- No human labels on these responses, so no evaluator's accuracy *on this
  material* is known — only held-out accuracy on Faulborn's own test split
  (A2) and inter-evaluator agreement here.
- The reconstructed classifiers are anatomiae reconstructions, not the
  authors' weights (A2: released checkpoint lacks a weights file).
- Link recovery by re-rendering assumes the A1 chat template; 90/90 prompt
  hashes matched, which verifies it for this cache.
