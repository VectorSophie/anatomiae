# anatomiae

Causal Anatomy of Political Behavior in Language Models.

Research-engineering pipeline separating political *behavior* (measured, model-conditional
output) from political *identity* (not attributed). See `docs/` for the research audit,
model/dataset audits, framework decision, and pilot protocol/results as they are produced.

GPU policy: this project is restricted to physical GPU index 1 only (see
`src/anatomiae/provenance/gpu_guard.py`). GPU 0 must never be used.
