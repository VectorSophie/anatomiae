# Architecture

## Research decomposition: Formation → Expression → Measurement

```mermaid
flowchart LR
    subgraph F["Formation"]
        D[Pretraining Data] --> P[Pretraining]
        P --> B[Base Model]
        B --> SFT[SFT]
        SFT --> PO[Preference Optimization / RL]
        PO --> A[Post-trained Model]
    end

    subgraph X["Expression"]
        A --> L[Language]
        A --> PR[Prompt / System]
        A --> DE[Decoding]
        L --> Y[Raw Response]
        PR --> Y
        DE --> Y
    end

    subgraph M["Measurement"]
        Y --> E[Evaluator]
        E --> RB[Measured Political Behavior]
    end
```

Source: `figures/source/research_framework.mmd`.

Every claim this project makes about "where political behavior comes
from" is scoped to one of these three groups. A finding that evaluator
choice (Measurement) changes the apparent result is just as valid and
reportable as a finding that pretraining-stage (Formation) does — see
`docs/STATISTICAL_METHOD_AUDIT.md` for how percentages/contributions are
allowed to be phrased.

## Pipeline separation (non-negotiable)

```mermaid
flowchart LR
    DS[Dataset Item] --> PC[Prompt Constructor]
    PC --> GEN[Generator<br/>Transformers / vLLM]
    GEN --> RAW[(Immutable Raw Response)]
    RAW --> PARSE[Parser]
    PARSE --> EA[Evaluator A]
    PARSE --> EB[Evaluator B]
    PARSE --> EC[Evaluator C]
    EA --> AN[Analysis]
    EB --> AN
    EC --> AN
    AN --> TAB[Tables]
    AN --> FIG[Figures]

    style RAW fill:#f5f5f5,stroke:#333,stroke-width:2px
```

Source: `figures/source/experiment_flow.mmd`.

Generation happens exactly once per `(model, prompt, decoding-config)`
triple; the raw response is written once and never overwritten. Every
evaluator reads the same immutable raw response — no evaluator can
influence what was generated, and re-running a new evaluator never
triggers regeneration. This is enforced structurally (see
`src/anatomiae/datasets/schema.py`, `docs/FRAMEWORK_DECISION.md`), not
just by convention.

## Module layout

```text
src/anatomiae/
├── cli/            # entry points (uv run anatomiae ...)
├── datasets/        # canonical item schema, dataset adapters
├── models/           # model registry (configs/models/*.yaml)
├── prompts/          # prompt construction, perturbations
├── inference/         # Transformers / vLLM backends
├── evaluators/        # scoring, one module per evaluator
├── metrics/            # outcome taxonomy, agreement statistics
├── analysis/            # raw records -> analysis frames -> tables
├── provenance/            # GPU isolation guard, generation-record schema
└── utils/
```

See `docs/FRAMEWORK_DECISION.md` for why this is a small bespoke package
rather than a fork of lm-evaluation-harness, Inspect AI, or HELM (patterns
from all three are borrowed; none is the foundation).

## GPU isolation

All GPU work on the maintainers' workstation is restricted to one
physical device, enforced by `src/anatomiae/provenance/gpu_guard.py`
before any CUDA context is created. This is a maintainer safety rule for
a specific shared machine, not a portability requirement — external
reproducers configure `CUDA_VISIBLE_DEVICES` for their own hardware. See
`docs/reproducibility.md`.
