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

**Existing credential state (verified, not a blocker):** An HF OAuth token is
already present locally on this workstation (expires 2026-10-13),
presumably the researcher's own account (identity intentionally omitted
from this public document). `whoami` succeeds and unauthenticated model
metadata for both repos resolves (HTTP 200), but a `HEAD` on `config.json`
for both `meta-llama/Llama-3.1-8B` and `google/gemma-3-12b-pt` returns
HTTP 403 under this token, confirming the license has not yet been
accepted for this account. No new token or login is needed — just the
license click-through.

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
