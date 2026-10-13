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

Next, per that document's handoff:
1. Cross-lineage replication of selective commitment (Qwen2.5-14B-Instruct, DeepSeek-R1-Distill-Qwen-14B). Freeze a
   short protocol amendment naming models/revisions/chat templates first; keep prompts, items, J-qy, metric and
   decision rule identical.
2. In parallel: a human audit of the commitment indicator C on a random, stage x prompt stratified sample of the new
   free-form responses.

Environment note: `uv sync --extra dev` removes the `ml`/`analysis` extras from the venv; on the workstation use
`uv sync --all-extras` (CI installs dev only).
