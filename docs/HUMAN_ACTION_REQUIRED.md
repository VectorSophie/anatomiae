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
the Hugging Face account tied to this workstation (see below) and accept the
license terms:
- https://huggingface.co/meta-llama/Llama-3.1-8B
- https://huggingface.co/google/gemma-3-12b-pt
(Instruct variants are typically covered by the same collection-level
acceptance, but check the `-Instruct`/`-it` pages too if download still 403s.)

**Why automation cannot perform it:** Accepting a gated model license is an
identity-bound legal action tied to a human account holder, listed explicitly
as requiring human action in the project's human-intervention policy.

**Existing credential state (verified, not a blocker):** an HF token is
already present on the maintainer workstation and resolves model metadata
for both repos (HTTP 200), but a `HEAD` on `config.json` for both
`meta-llama/Llama-3.1-8B` and `google/gemma-3-12b-pt` returns HTTP 403,
confirming the license has not yet been accepted for that account. No new
token or login is needed — just the license click-through. (Exact
expiry/account detail intentionally kept out of this public document -
see the maintainer-only, gitignored `.local/auth_state.md`.)

**What work can continue meanwhile:** Everything else. All locked model
anchors (OLMo 2, Amber/IFM, Qwen2.5-14B -> DeepSeek-R1-Distill-Qwen-14B,
Qwen3-14B) are ungated (apache-2.0 / mit) and already verified reachable.
Prior-work audit, framework decision, dataset audit, pilot pipeline
implementation, and smoke tests on ungated models all proceed without this.

**Scientific impact if unavailable:** Llama and Gemma are explicitly marked
"replaceable control families" (§10), not locked anchors. If license
acceptance is delayed or declined, the pilot proceeds on the locked anchors
alone and the Llama/Gemma external-validity control arm is deferred or
dropped; this does not block the core lineage/attribution analysis.

*(Resolved when the researcher confirms both licenses are accepted; verify
via `curl -I -H "Authorization: Bearer $TOKEN" https://huggingface.co/meta-llama/Llama-3.1-8B/resolve/main/config.json` returning 200.)*

---

## 2. (Optional, non-blocking) Request the missing Faulborn classifier weights from the authors

**Resource:** Faulborn et al.'s fine-tuned BART stance classifier, released
via the "stance detector model files" Google Drive folder linked from
`MaFa211/theory_grounded_pol_bias`'s README.

**Why needed:** that folder contains the step-1750 Trainer checkpoint's
config, tokenizer, scheduler, RNG and trainer state - but **no weights
file** (no `model.safetensors` / `pytorch_model.bin`). Verified from the raw
Drive folder listing, not just gdown's; no copy exists on the HF Hub, in
GitHub releases/LFS, or other branches (checked 2026-09-23). Hashes of the
files that *are* present are recorded in
`docs/results/faulborn_classifier_reproduction.md`.

**Action required from researcher:** optionally, contact the paper's
authors and ask whether the step-1750 weights file can be shared.

**Why automation cannot perform it:** contacting third-party researchers on
the project's behalf is a person-to-person communication decision, not an
engineering action.

**What work can continue meanwhile:** everything. The classifier is being
reconstructed from their released training data and script, and validated
against their own released test split and reported metrics; the
reconstruction is always labeled as such, never as their weights.

**Scientific impact if unavailable:** Gate A2 can reach "validated
reconstruction" but not "exact reproduction of the released weights."
