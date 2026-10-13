# anatomiae continuation handoff — 2026-10-13

This document is the authoritative handoff for a fresh Claude Code session continuing the project.

## 0. Scientific question

`anatomiae` studies where **measured political behavior** in language models comes from by separating three layers:

1. **Formation** — pretraining data/checkpoints, Base → SFT → preference optimization / RL.
2. **Expression** — prompt, language, render/chat format, decoding, truncation, backend/precision.
3. **Measurement** — the evaluator that maps generated text to a stance/outcome.

The working thesis is not “model X is left/right.” It is:

> Post-training can change how political behavior is expressed and how easily it is measured much more strongly than it changes the direction of explicit stances, and response-only evaluators can amplify apparent directional shifts when response form changes.

Use cautious causal language. Cross-family comparisons are associational; within-lineage interventions support stronger attribution.

---

## 1. Repository state at handoff

Expected starting point:

- branch: `main`
- CI fix commit: `ef56e79413777fb1b9cc1be989dd527d963f5604`
- CI run #52: **success**
- lint: success
- tests: success

The CI had been failing since the initial addition of `scripts/analyze_human_validation.py` because of two Ruff violations. Those are fixed. Do not re-open that issue unless a new failure appears.

Start with:

```bash
git status --short
git pull --ff-only
uv sync --extra dev
uv run ruff check .
uv run pytest -q
```

If the worktree contains local scientific artifacts, inspect them before cleaning anything.

---

## 2. Completed research

### Validation gates

Completed:

- Gate A1: Faulborn items/prefixes through the pipeline.
- Gate A2: Faulborn stance-classifier procedure reconstructed. Exact released weights are missing.
- Gate A3: evaluator / elicitation dependence.
- precision sensitivity.
- Gate B: Transformers vs vLLM backend equivalence.
- Gate C: three lineages end-to-end (OLMo 2, Qwen2.5-14B-Instruct, Amber).

### OLMo-2-13B lineage

The full second-pass OLMo chain is complete:

```text
Base → SFT → DPO → RLVR2
```

Design:

- 88 Faulborn items + 88 generated opposites.
- `opinion`, `please_respond`, and `stance_first` prompts.
- raw and native/chat render paths.
- 600-token generation budget.
- vLLM / BF16 / greedy.
- all eight evaluators.
- cached immutable generations.

Read:

- `docs/results/olmo_stages_v2.md`
- `artifacts/cache/olmo_stages_v2.jsonl`
- `artifacts/evaluations/olmo_stages_v2.jsonl`

Key results already established:

- stance-first causes SFT/DPO chat models to state an explicit position in ~99–100% of cases.
- SFT → DPO direction on explicit stance is small: roughly +0.02 to +0.11 depending on evaluator.
- under released free-form prompts, response-only classifiers infer much larger changes: roughly +0.09 to +0.40.
- DPO → RLVR2 explicit-stance direction is approximately -0.08 to +0.02 and no directional evaluator CI excludes zero.
- later post-training changes response mechanics much more visibly than explicit directional stance.
- Base cannot support a strong explicit-stance direction comparison because it rarely states a stance.
- GPT-generated inversion quality is imperfect; the 46 class-A faithful opposites are the strongest inversion subset.

Do **not** convert these into a claim about intrinsic political identity.

---

## 3. Human validation status — COMPLETE

### Frozen independent labels

- `artifacts/labeling/labels_v1_annotator_A_human.csv`
  - Human A = researcher.
  - Labels were chosen by the human. AI only packaged/exported them.
- `artifacts/labeling/labels_v1_annotator_C_human.csv`
  - independent second human.
  - final retained run used clarified French instructions and French statement translations while preserving the original English response.
- `artifacts/labeling/labels_v1_annotator_B_agent_completed.csv`
  - agent auxiliary evaluator only. Never call A↔B human-human IRR.

The first failed C attempt is scientifically excluded because of protocol misunderstanding. Do not reintroduce it.

### A ↔ C reliability

On the retained 150-response set:

- relation exact agreement: 105/150 = 70.0%
- relation Cohen κ ≈ 0.552
- mode exact agreement: 115/150 = 76.7%
- mode Cohen κ ≈ 0.564
- 61 responses disagreed on relation and/or mode.
- among 100 responses both humans considered directional, 97/100 had the **same direction**, κ ≈ 0.94.

Interpretation: most human disagreement is about **stance observability / commitment**, not support-vs-oppose direction.

### Completed adjudication

- `artifacts/labeling/human_AC_adjudication_completed.csv`
- `artifacts/labeling/labels_v1_human_consensus.csv`

This is a **researcher-adjudicated human consensus reference**, not an independent third-rater set.

Final consensus relation counts:

```text
supports                 60
opposes                  57
mixed_or_conditional     18
neutral_or_no_position    8
unclear                   7
```

117/150 are directional.

Final mode counts:

```text
explicit_stance          92
analytical_exposition    40
incomplete               10
voiced_continuation       5
empty                     1
refusal                   1
other                     1
```

Adjudication was not a mechanical preference:
- relation conflicts: A / C / third label = 20 / 20 / 5.
- mode conflicts: A / C / third label = 19 / 13 / 3.

### Sampling warning

The 150 responses were quota-sampled to oversample:
- DPO implicit/free-form cases,
- raw voiced continuation,
- classifier disagreement,
- original/inverse inconsistency,
- stance-first vs released-prompt differences,
- truncated directional outputs,
- explicit controls.

Therefore:

> Do not compute an unweighted stage-level political direction estimate from these 150 rows.

This set is a **measurement-validity sample**.

---

## 4. IMMEDIATE TASK: finish evaluator-vs-human-consensus validation

The script is:

```bash
uv run python scripts/analyze_human_validation.py
```

It now uses:

```text
artifacts/labeling/labels_v1_human_consensus.csv
```

as the primary human reference.

### 4.1 Handle `key_v1.csv` carefully

The script expects:

```text
artifacts/labeling/key_v1.csv
```

This was intentionally gitignored while human annotation was blind.

First check whether it exists locally.

```bash
ls -l artifacts/labeling/key_v1.csv
```

#### If it exists

Do not regenerate it. Run the validation script directly.

#### If it does not exist

Do **not** blindly rerun the sampling script and assume the new key is identical.

Recover the exact frozen key deterministically from the existing frozen sample and generation/evaluation artifacts. The preferred approach is to write a small reconstruction utility that:

1. loads the frozen 150 `sample_id, statement, response` rows from `sheet_v1_annotator_A.csv`;
2. loads the v2 generation cache + evaluations + Faulborn item metadata;
3. matches each frozen row to exactly one original cache record;
4. reconstructs `stage, render, prompt, polarity, item_id, item_lean, truncated, stratum` and all evaluator outcomes;
5. asserts:
   - 150/150 matched;
   - no duplicate match;
   - no ambiguous match;
   - statement and response are exact;
   - sample IDs remain unchanged;
6. writes `key_v1.csv` only after all assertions pass.

If exact reconstruction is impossible, stop and document the blocker rather than fabricating provenance.

Blind annotation is now complete, so the old reason for gitignoring `key_v1.csv` no longer applies. If the reconstructed/local key is proven exact and contains no machine-private information, it is acceptable to:
- remove its gitignore entry,
- version the key,
- document that it was withheld only during annotation blinding.

Do not expose secrets, workstation fingerprints, or private paths.

### 4.2 Required outputs

The validation should produce:

```text
artifacts/tables/human_validation_v1_evaluators.csv
artifacts/tables/human_validation_v1_strata.csv
artifacts/tables/human_validation_v1_human_agreement.csv
docs/results/human_validation_v1.md
```

Inspect the script output and verify denominators before interpreting anything.

Consensus has 7 `unclear` labels, which are currently excluded from evaluator-outcome scoring by `HUMAN_TO_OUTCOME`. Document exact n per metric rather than assuming all metrics use 150.

### 4.3 Diagnostics that matter most

Report, at minimum:

- overall accuracy
- macro-F1
- Cohen κ
- directional exact accuracy
- directional under-read rate
- directional over-read rate
- confusion structure
- results by:
  - human response mode
  - stage
  - render
  - prompt
  - truncation
  - sampling stratum

The **headline diagnostic** is:

> Among human-consensus `mixed_or_conditional` or `neutral_or_no_position` responses, how often does a response-only evaluator force `agreement` or `disagreement`?

Pay special attention to:
- `analytical_exposition`
- `voiced_continuation`
- released/free-form prompts
- DPO chat implicit responses

If over-read is concentrated there, that directly supports the Measurement-layer explanation.

Also compare easy `explicit_stance` cases with analytical prose. If evaluator accuracy is high on explicit stance but poor on analytical/voiced forms, make that distinction explicit.

### 4.4 Interpretation guardrail

Even if evaluator-vs-human results are dramatic, do not claim the 150-sample human consensus gives the population OLMo direction. It does not.

The valid claim is about evaluator validity/error on an intentionally difficult sample.

### 4.5 Finish this phase completely

After validating the numbers:

1. update `docs/results/human_validation_v1.md` with defensible interpretation;
2. update README/current-status text if needed;
3. run:
   ```bash
   uv run ruff check .
   uv run pytest -q
   ```
4. commit generated tables + report + any exact key reconstruction utility/provenance;
5. push;
6. verify GitHub CI is green.

Suggested commit message:

```text
research: finalize human-consensus evaluator validation
```

Do not stop after merely printing metrics.

---

## 5. NEXT TASK: test E(y) vs E(q, y)

Only begin after Section 4 is complete and committed.

### Motivation

A political stance label is conceptually a relation between:
- proposition/question `q`
- generated response `y`

but several existing automatic evaluators effectively score only the response:

```text
E(y)
```

This is vulnerable when the response is:
- analytical exposition,
- voiced continuation,
- hedged,
- truncated,
- or otherwise dependent on the original proposition for interpretation.

The next experiment should compare response-only measurement with a pair-conditioned evaluator:

```text
E(y)  vs  E(q, y)
```

against the 150-row human consensus.

### Before implementation

Create a short experiment protocol under `docs/experiments/` or equivalent that freezes:

- exact research question;
- evaluator(s);
- input formatting;
- label mapping;
- model/revision if a learned evaluator is used;
- decoding (if any);
- primary metrics;
- response-mode slices;
- success/failure criteria.

Do not select the “best” pair-conditioned evaluator after seeing all results without documenting the search.

### Minimum experiment

Use the same frozen 150 rows so the comparison is paired.

Compare:

1. existing response-only evaluator(s);
2. a pair-conditioned evaluator that sees both statement and response;
3. deterministic explicit-stance baseline;
4. human consensus.

Primary question:

> Does conditioning on the proposition reduce systematic over-reading or polarity errors, particularly for analytical exposition and voiced continuation?

At minimum report:
- accuracy / macro-F1 / κ;
- directional over-read / under-read;
- direct support↔oppose reversal rate;
- mode-stratified metrics;
- paired error changes per sample.

Keep generation fixed. This is a **Measurement-layer** experiment; do not regenerate model answers unless scientifically necessary.

### Stronger causal extension: response-form minimal pairs

After the pair-conditioned evaluator pilot, consider a small controlled set where semantic stance is held as constant as practical and response form is manipulated:

- explicit stance
- analytical exposition
- voiced/continuation-like
- hedged
- truncated

This tests whether changing expression form alone changes measured stance.

Document construction and human validation before presenting it as causal evidence.

---

## 6. Backlog after E(q,y)

Do not jump to these before the immediate tasks unless a blocker makes them necessary.

### A. Representative human sample

Draw a separate representative or explicitly weighted human sample from the full OLMo experiment if a human-grounded **population stage-direction estimate** is needed.

Do not reuse the ambiguity-stratified 150 for that purpose.

### B. Cross-lineage replication

Replicate the core measurement gap on a Qwen lineage, preferably a focused design around:

```text
Qwen2.5-14B Base
Qwen2.5-14B-Instruct
DeepSeek-R1-Distill-Qwen-14B
```

Use a reduced but principled item set rather than indiscriminately multiplying generations.

### C. Prompt-family robustness

Use multiple paraphrases of stance-first to show the result is not an accident of one instruction string.

### D. Decoding robustness

On a representative subset, compare greedy with a documented stochastic setup and multiple seeds.

### E. Amber trajectory / multilingual

Useful Formation/Expression extensions, but lower priority than resolving evaluator validity and generalizing the core result.

---

## 7. Frozen scientific rules

Do not:

- overwrite Human A, Human C, or adjudication artifacts;
- relabel agent B as a human;
- call A↔B human-human reliability;
- use the discarded first C attempt;
- claim political identity from measured behavior;
- conflate refusal with stance;
- conflate normative disagreement with misinformation;
- use the 150 ambiguity-stratified sample as an unweighted stage-direction estimator;
- manually edit generated numeric tables to fit a narrative;
- rerun generations merely because a new evaluator is added;
- silently change model revisions, decoding, prompt text, or backend;
- use GPU 0 or both GPUs on the maintainer workstation.

If an observed result contradicts the current thesis, preserve and report it. The goal is decomposition, not proving a predetermined direction.

---

## 8. Files to read first in a fresh session

In order:

1. `CLAUDE.md`
2. this handoff
3. `README.md`
4. `docs/results/human_validation_v1_preliminary.md`
5. `artifacts/labeling/README.md`
6. `artifacts/labeling/precheck_v1.md`
7. `scripts/analyze_human_validation.py`
8. `docs/results/olmo_stages_v2.md`
9. `scripts/make_labeling_sheet.py`
10. `docs/reproducibility.md`

The immediate deliverable is not another broad literature review. It is a finished, reproducible human-consensus evaluator validation, followed by a preregistered pair-conditioned evaluator experiment.
