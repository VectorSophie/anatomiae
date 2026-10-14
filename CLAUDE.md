# Claude Code instructions for anatomiae

Read `docs/CONTINUATION_HANDOFF_2026-10-13.md` in full before making research or code changes.

## Working style

- Continue the research autonomously. Do not ask the user to restate context already present in the repository or handoff.
- Inspect existing artifacts before creating replacements. Never silently regenerate frozen scientific artifacts.
- Preserve the distinction between **Formation**, **Expression**, and **Measurement**. Do not describe a model as having a political identity; report measured stance, asymmetry, refusal, instability, evaluator dependence, or related observable behavior.
- Human A and Human C independent labels are frozen. The researcher-adjudicated A/C consensus is a separate artifact and must not overwrite either independent set.
- The 150-response human-validation sample intentionally oversamples difficult/ambiguous cases. It is for evaluator validation, not an unweighted population estimate of OLMo stage direction.
- Keep generation and scoring separated. Reuse immutable cached responses whenever possible.
- Before every push run:
  ```bash
  uv run ruff check .
  uv run pytest -q
  ```
- Commit coherent scientific milestones with descriptive messages and push them once validated.
- Current `main` CI is green as of commit `ef56e794` (CI run #52).

## GPU policy

On the maintainer workstation, use **physical GPU 1 only**. Never use GPU 0 or both GPUs.

Before CUDA work:

```bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1
bash scripts/gpu_preflight.sh
```

Respect the repository GPU guard and minimum-power-limit preflight. Use tensor parallel size 1. For the known Blackwell/vLLM PTX issue, the established workaround is `VLLM_ATTENTION_BACKEND=FLASHINFER` when needed.

## Immediate goal

Completed on 2026-10-13: human-consensus evaluator validation, E(y) vs E(q,y), the J-qy re-read of OLMo v2,
and the prospective selective-commitment confirmation (`docs/results/selective_commitment_confirmatory_v1.md`).

2026-10-13 (UTC 09:08-09:15): cross-lineage amendment frozen (`docs/experiments/selective_commitment_crosslineage_v1.md`, `ccabd8c`);
Qwen2.5-14B-Instruct arm replicates (`docs/results/selective_commitment_crosslineage_v1.md`). Commitment-audit
sample frozen (`docs/experiments/commitment_audit_v1.md`); blind sheets in `artifacts/labeling/commitment_audit_v1/`.

2026-10-14: R1-Distill secondary arm complete (+0.34 [0.15, 0.54], no detectable sibling difference). Qwen
self-judging audit frozen (`docs/experiments/commitment_audit_qwen_v1.md`). Consolidated status and recommendation:
`docs/results/research_status_2026-10-14.md`. No further model families by default.

Next (human action first):
1. Human A and C fill `artifacts/labeling/commitment_audit_v1/sheet_{A,C}.csv` and
   `artifacts/labeling/commitment_audit_qwen_v1/sheet_{A,C}.csv` independently. Never edit them afterwards.
2. `uv run python scripts/analyze_commitment_audit.py check --audit olmo` (and `--audit qwen`): validation, A<->C
   agreement, adjudication queue. Researcher fills the queue; then `consensus`, then `analyze` (only step that reads
   the local key). Write `docs/results/commitment_audit_v1.md` / `commitment_audit_qwen_v1.md`.
3. Then decide per section 6 of the status document: manuscript drafting, or the single human-C re-estimation.

Environment note: `uv sync --extra dev` removes the `ml`/`analysis` extras from the venv; on the workstation use
`uv sync --all-extras` (CI installs dev only).
