# Human action required

Normally this file should be empty. Each entry documents a blocker that
genuinely requires the human researcher, and what is proceeding in parallel.

---

## 1. Accept gated licenses: Llama 3.1 and Gemma 3

**Resource:**
- `meta-llama/Llama-3.1-8B` and `meta-llama/Llama-3.1-8B-Instruct`
- `google/gemma-3-12b-pt` and `google/gemma-3-12b-it`

**Action required:** accept the gated model terms from the corresponding Hugging Face account.
These are optional external-validity controls and do not block the locked anchors.

---

## 2. (Optional, non-blocking) Request missing Faulborn classifier weights

The released stance-detector folder contains checkpoint metadata but no model weight file.
An exact reproduction remains blocked unless the authors provide the weights.
The leakage-free reconstruction remains available in the meantime.

---

## 3. Adjudicate independent human A↔C disagreements

**Current state:**

- Human A: complete, 150/150.
- Human C: complete, 150/150, independent blind rerun.
- Relation agreement: 105/150 (70.0%), Cohen's kappa = 0.552.
- Mode agreement: 115/150 (76.7%), Cohen's kappa = 0.564.
- Both-directional subset: 97/100 (97.0%) same direction.
- 61 unique responses disagree on relation and/or mode.
- Queue: `artifacts/labeling/human_AC_adjudication_pending.csv`.

**Action required:** adjudicate the 61 disagreement cases. Prefer an adjudicator who is blind
to which displayed label came from A or C. A third independent human adjudicator is the
strongest publication-grade option. If the researcher adjudicates, report the final set as
researcher-adjudicated human consensus rather than independent third-party consensus.

After adjudication, preserve the original A and C label sets unchanged and create a separate
final consensus file.

**Sampling limitation:** the 150-response sample deliberately oversamples ambiguous and
evaluator-disagreement strata. Human-human reliability on this sample is valid, but an
unweighted stage-direction estimate is not a population estimate.

**What can proceed now:** evaluator-vs-human-A/C analysis, pair-conditioned evaluator work,
response-mode error analysis, representative-sample design, and cross-lineage replication.
