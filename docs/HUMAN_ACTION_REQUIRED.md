# Human action required

Normally this file should be empty. Each entry documents a blocker that
genuinely requires the human researcher, and what is proceeding in parallel.

---

## 1. Accept gated licenses: Llama 3.1 and Gemma 3

**Resource:**
- `meta-llama/Llama-3.1-8B` and `meta-llama/Llama-3.1-8B-Instruct` (license: `llama3.1`, gated: manual)
- `google/gemma-3-12b-pt` and `google/gemma-3-12b-it` (license: `gemma`, gated: manual)

**Why needed:** These are the current planned control-family models (§10 of the
research spec — Llama/Gemma Base vs Instruct controls). Both require a
human to click "Agree and access repository" on the model page while logged
into a Hugging Face account.

**Action required from researcher:** Visit the following while logged into
the Hugging Face account tied to this workstation and accept the license terms:
- https://huggingface.co/meta-llama/Llama-3.1-8B
- https://huggingface.co/google/gemma-3-12b-pt
(Instruct variants are typically covered by the same collection-level
acceptance, but check the `-Instruct`/`-it` pages too if download still 403s.)

**Why automation cannot perform it:** Accepting a gated model license is an
identity-bound legal action tied to a human account holder.

**What work can continue meanwhile:** Everything else. Llama/Gemma are
replaceable external-validity controls and do not block the locked anchors.

---

## 2. (Optional, non-blocking) Request the missing Faulborn classifier weights from the authors

**Resource:** Faulborn et al.'s fine-tuned BART stance classifier, released
via the "stance detector model files" Google Drive folder linked from
`MaFa211/theory_grounded_pol_bias`'s README.

**Why needed:** that folder contains the step-1750 Trainer checkpoint's
config, tokenizer, scheduler, RNG and trainer state but no weights file
(`model.safetensors` / `pytorch_model.bin`). This is documented in
`docs/results/faulborn_classifier_reproduction.md`.

**Action required from researcher:** optionally contact the paper's authors and ask whether the weights can be shared.

**Scientific impact if unavailable:** Gate A2 remains a validated reconstruction rather than an exact reproduction of the authors' released weights.

---

## 3. Second independent human annotation on frozen v1 sample

**Current state:**

- `labels_v1_annotator_A_human.csv`: complete, 150/150, labeled by the human researcher.
- `labels_v1_annotator_B_agent_completed.csv`: complete, 150/150, produced by an agent and therefore not a second human annotator.
- `human_agent_review_v1.csv`: complete, 38/38 A-vs-agent disagreement cases re-read by the human researcher after seeing both labels. This is a sensitivity artifact, not independent human-human adjudication.
- `adjudication_v1.csv`: reserved for future independent human-human adjudication.

**Why a second human is still useful:** Human A can already serve as the primary
human reference for validating automatic evaluators. However, the paper should not
report A-vs-agent agreement as human inter-rater reliability. A second independent
human allows actual human-human agreement and consensus labels to be estimated.

**Action required:** A second human (`Annotator C`) should label the same frozen
150 responses independently, without seeing A labels, agent-B labels,
`human_agent_review_v1.csv`, model/stage identity, evaluator outputs, or political
coding. Store the compact completed labels as:

`artifacts/labeling/labels_v1_annotator_C_human.csv`

Then run:

```bash
uv run python scripts/analyze_human_validation.py
```

The script uses the local-only frozen `artifacts/labeling/key_v1.csv` to compare
human labels against all automatic evaluators and, when C exists, report human-human
agreement.

**Sampling limitation:** The 150-response sample deliberately oversamples ambiguous
and evaluator-disagreement strata. It is a measurement-validity sample, not a
representative draw from the entire stage experiment. Do not compute an unweighted
population stage-direction estimate from these 150 responses. A human-grounded
population-direction claim requires either a separately representative sample or an
explicitly justified sampling-weight estimator.

**What can proceed now:** evaluator-vs-human-A validation, response-mode error
analysis, and human-agent sensitivity analysis can proceed immediately. Human-human
reliability and consensus human labels remain pending Annotator C.
