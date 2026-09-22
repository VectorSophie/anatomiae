# Framework decision

**Status: preliminary decision based on architectural knowledge of the
three prior-art systems plus reachability verification; to be revisited
if a deep code read-through of lm-evaluation-harness/Inspect AI surfaces
something that changes the calculus.**

## Candidates considered

### EleutherAI/lm-evaluation-harness
Mature, huge task library, first-class HF/vLLM backend abstraction,
strong caching. Task abstraction is fairly flat (a task = dataset +
prompt template + metric) and oriented around scalar accuracy-style
metrics; multi-evaluator scoring of *cached* raw generations (this
project's non-negotiable separation, §39) is not its native shape — it
tends to couple generation and scoring per task. Extending it to support
"generate once, score N ways with independent evaluators, keep raw
outputs as first-class artifacts across languages/perturbations" is
possible but works against the grain.

### Inspect AI
Clean `Task -> Solver -> Scorer` composition that *does* separate
generation from scoring, good eval-log format, growing ecosystem. Newer
and more opinionated about its own eval-log/sample model; adapting its
sample/state model to carry this project's provenance fields (chat
template, decoding regime, language, perturbation type, translation
provenance, GPU identity) is plausible but means working inside someone
else's state machine for a research pipeline whose exact record shape is
still evolving (§44's provenance schema is itself a moving target this
month).

### Stanford HELM
Best-matched conceptual model for this project: scenario abstraction,
built-in perturbations, multidimensional evaluation, standardized run
configuration. However HELM entered maintenance mode in 2026 (per the
spec's own framing, confirmed still the operative status) — not a
foundation to build new work on, only architectural prior art. Its
scenario/perturbation split is worth copying conceptually.

## Decision: small bespoke pipeline, patterns borrowed, no fork

Build a small, purpose-specific package (`src/anatomiae/`) rather than
extending or forking any of the three. Reasons:

1. **The non-negotiable separation (§39) is the whole point.** `Dataset
   Item -> Prompt Constructor -> Generator -> Immutable Raw Output ->
   Evaluator A/B/C -> Analysis` needs to be structurally impossible to
   violate (no evaluator can see generation-time state; raw JSONL records
   are append-only and evaluators only ever read them back). None of the
   three frameworks make this the *central* organizing structure — it's
   achievable in each but not what they optimize for. A ~10-module bespoke
   package can make it the literal file/class boundary.
2. **Multilingual + perturbation + lineage metadata is unusually rich**
   compared to a typical eval-harness task (model lineage tree, checkpoint
   revision, translation provenance, semantic-equivalence status, GPU
   provenance, backend-divergence records). Pydantic/dataclass schemas
   designed for this from scratch are simpler than mapping onto another
   system's sample/state object.
3. **Small team, single paper, single research question family.** This is
   explicitly not "a distributed banking platform" (§38's own framing).
   The generality lm-eval-harness/Inspect provide (hundreds of tasks,
   plugin ecosystems, many backends) is mostly unused surface area here.
4. **vLLM/Transformers dual-backend requirement (§60) is simple to own
   directly** — a thin `inference/` module with two backends and a
   divergence-comparison harness, versus adapting either framework's
   backend abstraction to guarantee bit-for-bit-comparable golden-set runs
   across both.

## What is borrowed (not reinvented)

- **HELM's scenario/perturbation split** — dataset items are perturbation-
  agnostic; perturbations (paraphrase, framing, language) are applied as a
  separate transform layer with provenance, not baked into the dataset.
- **Inspect AI's Solver/Scorer separation** as the conceptual model for
  `inference/` vs `evaluators/`, even though the concrete implementation
  is bespoke.
- **lm-eval-harness's backend abstraction pattern** (a thin common
  interface over HF Transformers and vLLM) for `inference/backends.py`.
- Caching discipline from all three: content-addressed / hash-keyed raw
  generation cache so a rerun never regenerates a completed (model,
  prompt-hash, decoding-config) triple.

## Scoring against requirements (qualitative)

| Requirement | bespoke | lm-eval-harness | Inspect AI | HELM |
|---|---|---|---|---|
| Raw-output preservation as first-class artifact | native | bolt-on | native-ish | native |
| Multiple independent evaluators over cached output | native | bolt-on | good fit | good fit |
| Multilingual + translation provenance metadata | native | bolt-on | bolt-on | bolt-on |
| vLLM throughput | direct | native | native | unclear (maintenance mode) |
| Base-model (non-chat) support | direct | native | good | good |
| Experiment manifests (§64) | native | partial | good (eval logs) | good |
| Reproducibility | direct | good | good | good |
| Extensibility for this project's specific needs | high | medium | medium-high | low (maintenance mode) |
| Complexity added | low | medium-high | medium | high (unmaintained) |

## Open follow-up

A closer code read of lm-evaluation-harness's caching layer and Inspect
AI's eval-log schema is still worth doing before finalizing
`src/anatomiae/provenance/` — reusing a well-tested serialization format
(e.g. Inspect's `.eval` log structure) for interoperability with existing
tooling (e.g. viewing logs in Inspect's log viewer) may be worth a thin
export/import adapter even without adopting Inspect's runtime.
