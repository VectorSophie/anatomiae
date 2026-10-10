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

## Human annotation status

No human-labeling action is currently required. Human A and Human C are complete,
all 61 disagreements have been researcher-adjudicated, and
`artifacts/labeling/labels_v1_human_consensus.csv` is the final validation reference.

The next blocker is computational: join the consensus labels to the frozen local
`key_v1.csv` and run evaluator validation.