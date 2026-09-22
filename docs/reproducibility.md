# Reproducibility

## Verifying the repository without a GPU

```bash
git clone https://github.com/VectorSophie/anatomiae.git
cd anatomiae
uv sync --extra dev
uv run pytest
```

This installs only the lightweight `dev` extra (schema/config code,
pydantic, pytest, ruff) — no torch, no vLLM, no CUDA. It validates the
model registry, dataset schema, and GPU-isolation guard's *logic*
(mocked hardware) without touching a GPU. This is what CI runs on every
push.

## Running GPU workloads

GPU workloads need the `ml` extra:

```bash
uv sync --extra dev --extra ml
```

Then set `CUDA_VISIBLE_DEVICES` for **your own hardware** before running
anything that touches a GPU:

```bash
nvidia-smi                          # find the physical index you want to use
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=<your index>
```

`src/anatomiae/provenance/gpu_guard.py` will abort rather than run on an
unverified device.

**Maintainer-specific note:** the anatomiae maintainers' own workstation
is restricted, as a local safety rule, to physical GPU index 1 only (a
shared multi-tenant machine where GPU 0 is reserved for other work). This
restriction is *not* a scientific requirement of the project and is not
enforced for external reproducers — `EXPECTED_PHYSICAL_INDEX` in
`gpu_guard.py` is a workstation setting, not a hard-coded constant meant
to travel with the code; change it (or override via the module's public
API) for your own environment.

## Known machine-specific workaround: vLLM on Blackwell GPUs

On the maintainers' RTX PRO 6000 Blackwell workstation, vLLM's bundled
FlashAttention kernel fails with:

```text
torch.AcceleratorError: CUDA error: the provided PTX was compiled with an unsupported toolchain
```

Fix, confirmed working (vLLM 0.14.0 + FlashInfer 0.5.3):

```bash
export VLLM_ATTENTION_BACKEND=FLASHINFER
```

This is an observed engineering fact about *this* hardware/toolchain
combination at *this* point in time, not a universal anatomiae
requirement — vLLM's default backend may work fine on other GPUs/driver
versions. If you hit the same error, try this; if you don't, you don't
need it. See `docs/MODEL_AUDIT.md` for full detail.

## Download bandwidth

Model downloads from Hugging Face are the long pole. On the maintainers'
workstation, measured sustained throughput to the HF CDN is ~2.3–2.6
MB/s, and additional concurrency does not meaningfully increase aggregate
throughput (measured directly — see `docs/MODEL_AUDIT.md`). This appears
to be a link-level ceiling specific to that machine's network path, not a
property of Hugging Face's CDN in general; your own bandwidth may differ
substantially. Budget accordingly, use resumable downloads
(`scripts/bulk_download.py` wraps `huggingface_hub.snapshot_download`,
which resumes automatically), and don't delete partial downloads.

## What's deterministic and what isn't

- GPU-isolation guard unit tests: fully deterministic, hardware-independent
  (mocked NVML snapshots).
- Greedy/temperature=0 generation: deterministic *within* a backend, but
  Transformers and vLLM are **not** guaranteed to produce byte-identical
  output for the same model/prompt/seed — measured and documented, not
  assumed (`docs/results/backend_equivalence.md` once populated with real
  data; a single-pair smoke comparison is already in
  `artifacts/logs/backend_divergence_result.json`).
- Sampling-based generation: seeded but backend RNG implementations differ;
  treat cross-backend exact-match as out of scope.
