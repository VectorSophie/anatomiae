# Figures

`source/` holds the canonical Mermaid source for every diagram used in
`README.md` and `docs/`. GitHub renders Mermaid code blocks natively in
Markdown, so the source files are consumed directly (copy-pasted into the
relevant `.md` file's fenced ```mermaid block) rather than requiring a
separate render step for normal viewing.

`generated/` is reserved for static (`.svg`/`.pdf`/`.png`) exports of
these diagrams for contexts that don't render Mermaid (e.g. a future
manuscript). A `mermaid-cli` render attempt during initial setup did not
produce usable output on this workstation; regenerating this directory
with a working `mmdc` invocation (or an alternative renderer) is a known
follow-up, not yet done. Do not assume files exist here without checking.

| Source | Used in |
|---|---|
| `source/research_framework.mmd` | `README.md`, `docs/architecture.md` |
| `source/model_lineages.mmd` | `README.md`, `docs/model_lineages.md` |
| `source/experiment_flow.mmd` | `docs/architecture.md` |
