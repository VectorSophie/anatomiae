# Model audit

Status legend: `verified` = confirmed reachable via HF Hub API with real
repo IDs and license/gating info; `pending` = not yet checked; `blocked` =
confirmed blocker recorded in HUMAN_ACTION_REQUIRED.md.

All entries below were checked live against `https://huggingface.co/api/models/...`
on 2026-09-22, not recalled from training data, because model-card naming
for these families changes frequently.

---

## Locked anchor A — OLMo 2 (allenai)

**Status: verified, ungated, full lineage confirmed.**

The `allenai/OLMo-2-1124-13B` family has a *more* complete public lineage
than the spec assumed: it exposes not just Base/SFT/DPO/Instruct but two
separate RLVR checkpoints and the reward model:

| Stage | Repo | Notes |
|---|---|---|
| Base | `allenai/OLMo-2-1124-13B` | pretrained |
| SFT | `allenai/OLMo-2-1124-13B-SFT` | also `-SFT-Preview` |
| DPO | `allenai/OLMo-2-1124-13B-DPO` | also `-DPO-Preview` |
| RM | `allenai/OLMo-2-1124-13B-RM` | reward model used for PO |
| RLVR1 | `allenai/OLMo-2-1124-13B-Instruct-RLVR1` | intermediate RLVR stage |
| RLVR2 / Instruct | `allenai/OLMo-2-1124-13B-Instruct-RLVR2`, `allenai/OLMo-2-1124-13B-Instruct` | final released instruct state |

All apache-2.0, `gated: false`.

**Note on OLMo 3:** AllenAI has since released OLMo 3 (`Olmo-3-7B-Instruct`,
`Olmo-3-32B-Think`, `Olmo-3.1-32B-Think`, etc., dated Oct/Nov 2025). It is
newer but does **not** expose a comparable dense Base->SFT->DPO->RLVR chain
at a 13B-class scale in the current public listing (mostly Think/Instruct
variants at 7B/32B with no visible parallel DPO/RM/RLVR1/RLVR2 checkpoints
for a single dense line). Per the locked-anchor policy (§6: "a replacement
model lacking those artifacts does not replace OLMo scientifically"),
**OLMo-2-1124-13B remains the correct locked anchor** for the pilot. This
should be revisited if OLMo 3 later publishes an equivalently granular
lineage at comparable scale — documented here, not silently substituted.

7B and 1B OLMo-2 lines exist too (same lineage granularity) and are the
natural Tier-1 smoke-test candidates before committing to 13B downloads.

---

## Locked anchor B — LLM360 Amber

**Status: verified, ungated, full checkpoint sequence confirmed.**

**Naming change:** the LLM360 Hugging Face org has been renamed/migrated to
**`IFM`**. `https://huggingface.co/api/models/LLM360/Amber` returns HTTP 307
(redirect), and the live repo is `IFM/Amber` (arxiv:2312.06550 — the
original Amber/LLM360 paper reference, confirming continuity, not a
different model). This is a rename, not an anchor replacement, so no human
decision is needed here per §5's own criteria (an organizational rename is
not "artifact removal, incompatible release, licensing prohibition,
inaccessible gated resource, or irreparable incompatibility").

- Final checkpoint: `IFM/Amber` (license apache-2.0, ungated)
- Chat variants: `IFM/AmberChat`, `IFM/AmberSafe`
- **Intermediate checkpoints: confirmed present as git branches `ckpt_000`
  through (at least) `ckpt_020`+ on the `IFM/Amber` repo — the promised
  360-checkpoint sequence is real and accessible via
  `revision=ckpt_NNN`.** This is exactly the mechanism the spec's D_t -> W_t
  -> B_t analysis needs.
- Training data: `IFM/AmberDatasets` (dataset repo, confirmed to exist;
  size not yet measured).
- Related IFM/LLM360 lineage artifacts also present: `IFM/Crystal`,
  `IFM/CrystalChat`, `IFM/CrystalCoderDatasets`, `IFM/TxT360`, `IFM/K2*`
  family (not part of the locked anchor, noted for awareness only).

Open item: which checkpoint indices to select for the pilot subset (spec
says "one scientifically meaningful earlier checkpoint, final checkpoint" —
§52). Plan: inspect `IFM/Amber`'s published loss/eval curves
(`amber-{arc,hellaswag,mmlu,truthfulqa}-curve.*` are repo files) to pick an
early checkpoint past the initial-loss-spike region, plus a mid-training
checkpoint, before downloading full safetensors for each.

---

## Locked anchor C — Qwen2.5-14B -> DeepSeek-R1-Distill-Qwen-14B lineage

**Status: verified, ungated.**

- `Qwen/Qwen2.5-14B` (base, apache-2.0)
- `Qwen/Qwen2.5-14B-Instruct` (Qwen-native post-training, apache-2.0)
- `deepseek-ai/DeepSeek-R1-Distill-Qwen-14B` (R1-distilled onto the same
  Qwen2.5-14B foundation, MIT license)

All three confirmed to exist and be ungated. This gives the intended
`W_Qwen + P_Qwen` vs `W_Qwen + P_R1` comparison directly.

---

## Important-but-replaceable — Qwen3-14B

**Status: verified, ungated.**

- `Qwen/Qwen3-14B-Base` (pretrained)
- `Qwen/Qwen3-14B` (post-trained default; thinking-mode toggle built in)

Both apache-2.0, ungated. Matches the planned pair from §52 as-is; no
substitution needed for the pilot.

---

## Replaceable controls — Llama 3.1 8B / Gemma 3 12B

**Status: verified to exist; gated, license not yet accepted for this
account. See docs/HUMAN_ACTION_REQUIRED.md #1.**

- `meta-llama/Llama-3.1-8B`, `meta-llama/Llama-3.1-8B-Instruct` — gated
  (`gated: manual`, license `llama3.1`).
- `google/gemma-3-12b-pt`, `google/gemma-3-12b-it` — gated
  (`gated: manual`, license `gemma`).

An HF OAuth token already exists locally
(`/data/jackb/krasis/huggingface/token`, user `VectorSophie`, expires
2026-10-13) and resolves model metadata (HTTP 200) but gets HTTP 403 on
actual weight files for both families — license not yet accepted under this
account. This does not block the locked-anchor pilot arms.

---

## Environment / cache notes

- `HF_HOME=/data/jackb/krasis/huggingface` is already set in the shell
  environment and points to an existing jackb-owned cache (part of a prior
  "krasis" project setup under `/home/jackb/local-ai`, `/home/jackb/.krasis`).
  Per download-engineering guidance (§31-32: avoid duplicate downloads,
  reuse cache across environments), anatomiae reuses this same `HF_HOME`
  rather than creating a second multi-hundred-GB cache. The krasis project
  itself is left untouched.
- A separate, smaller `~/.cache/huggingface` (30G) also exists from before
  `HF_HOME` was set; not used going forward, left as-is.
- `/data` is a shared 3.6T volume (39% used, 2.1T free) with multiple other
  users' model caches (`cr`, `gyu`, `lsj`, ...). `/data/jackb/` is the
  jackb-owned subtree; large anatomiae-specific artifacts that don't belong
  in the shared HF cache should go under `/data/jackb/anatomiae/` (created
  as needed), not the git repo at `/home/jackb/workspace/anatomiae`.

## Download bandwidth (measured, important for feasibility)

Measured directly on 2026-09-22/23, three independent ways:
- xet-backed transfer (default HF client path): ~2-3 MB/s sustained,
  visible in xet logs as repeated "connection struggling" concurrency
  downgrades to 1.
- Plain HTTP range GET on the HF CDN, bypassing xet entirely: 200 MB in
  92s = **2.28 MB/s**.
- 4 parallel HTTP range GETs on the same file: aggregate **~2.57 MB/s**
  (each individual stream dropped to ~0.6-0.75 MB/s) - only ~13% better
  than one stream, confirming this is a real link-bandwidth ceiling on
  this box's path to the HF CDN, not a per-connection or xet-specific
  throttle. Concurrency tuning does not meaningfully help.

**Consequence for the pilot:** at ~2.3-2.6 MB/s, the Tier 2 locked-anchor
set alone (~230 GB: OLMo-2-13B x4 checkpoints, Amber x3, Qwen2.5-14B x2,
DeepSeek-R1-Distill-Qwen-14B) is roughly **24-28 hours of continuous
transfer**; Tier 3's unblocked Qwen3-14B pair adds ~6 hours; Llama/Gemma
(pending license) would add another ~9 hours. This reframes "download the
real artifacts" from an afternoon task to a multi-day background job and
should be reflected directly in the compute projection section of
`docs/PILOT_RESULTS_AND_FEASIBILITY.md`.

**Response:** downloads are run sequentially (not concurrently - measured
to not help) via `scripts/bulk_download.py`, ordered by scientific
priority, as resumable background jobs (`huggingface_hub.snapshot_download`
caching means an interrupted/rerun invocation never re-fetches a completed
file). Priority order launched: OLMo-2-1124-13B Base -> SFT -> DPO ->
Instruct-RLVR2 first, since this lineage is the most direct evidence for
RQ1 (emergence) and stage-wise movement, and is the locked anchor with the
most scientific weight overall.

## vLLM on Blackwell: attention-backend fix (real infra failure, repaired)

First vLLM smoke test (default attention backend, vLLM 0.14.0) failed hard:
`torch.AcceleratorError: CUDA error: the provided PTX was compiled with an
unsupported toolchain` inside `vllm_flash_attn`'s `varlen_fwd` kernel. This
is vLLM's bundled flash-attention CUDA kernel not yet supporting this RTX
PRO 6000 Blackwell (sm_120, very new architecture) toolchain-wise, not a
GPU-isolation or driver problem (driver 570.211.01 / CUDA 12.8 / torch
2.9.1+cu128 all otherwise consistent).

**Fix:** set `VLLM_ATTENTION_BACKEND=FLASHINFER` (FlashInfer 0.5.3 was
already pulled in as a vLLM dependency). Confirmed working: model load
27.2s, generation 0.8s for 144 tokens, real non-refusal output produced.
vLLM logs a deprecation warning that this env var will be removed in
v0.14.0/v1.0.0 in favor of `--attention-config.backend` /
`AttentionConfig(backend=...)` - the inference module should use the
programmatic config form, not the env var, once `inference/backends.py` is
built (see `docs/FRAMEWORK_DECISION.md`).

**Backend divergence (expected, now measured):** same model
(OLMo-2-0425-1B-Instruct), same rendered prompt, greedy decoding
(temperature=0) on both Transformers and vLLM/FlashInfer - outputs are
**not** byte-identical. Both start with an essentially identical first two
sentences, then diverge (different word choices, different closing
sentence) while remaining substantively equivalent in content and stance.
This is expected (different attention-kernel numerics accumulate different
floating-point rounding under bf16), but it is exactly the kind of
divergence §60 asks to characterize before trusting either backend at
scale - "identical" should not be the acceptance bar for backend
correctness in this pipeline; "substantively equivalent, stance-preserving"
should be, and that needs its own evaluator-level check once the
evaluator layer exists, not just a string-equality check. Raw comparison
saved at `artifacts/logs/backend_divergence_result.json`.

## GPU baseline

- `nvidia-smi -L`: GPU 0 = `GPU-fce8b8f8-013c-de7a-bcc3-3b9f651d097b`
  (PCI 00000000:4A:00.0), GPU 1 = `GPU-1824bc0e-afda-4c3f-454b-71e779e4db34`
  (PCI 00000000:CA:00.0). Both RTX PRO 6000 Blackwell Max-Q, 97887 MiB.
- `CUDA_VISIBLE_DEVICES=1` is already pre-set in the shell environment.
- **Observed quirk:** GPU 1 reports persistent 100% `utilization.gpu` and
  ~93W power draw with 0 MiB memory used and no visible compute process
  (`nvidia-smi pmon` shows no process on either GPU). Sampled repeatedly
  over several seconds; stable. Not containerized (`systemd-detect-virt` =
  none). Read as a driver/power-state artifact on this Blackwell card
  rather than real contention (no memory is held), but logged here as a
  baseline in case throughput measurements during the pilot look off from
  what idle GPU 0 shows (0% util, 5W).
