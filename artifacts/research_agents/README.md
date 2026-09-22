# Research-agent write isolation

Policy, adopted after a real incident: a parallel research subagent
(assigned to deep-read one locked paper) wrote directly to
`docs/PRIOR_WORK_MATRIX.md` with a full-file overwrite instead of
reporting its findings back to the orchestrator, clobbering a section
another subagent had already appended. It was recovered manually by
re-reading and re-appending. A second subagent, told the same thing,
also wrote directly to `CITATIONS.bib` and `docs/DATASET_AUDIT.md` -
harmlessly in that case (the content was correct and, incidentally,
caught an error in the orchestrator's own first-pass notes), but the
pattern is not reliable: a third subagent assigned to a different paper
skipped its task entirely and returned an unrelated status summary
instead. **"Don't write files, just report back" is not effective
concurrency control against a subagent's own judgment.**

## Structural fix

Research subagents (fetching/reading papers, repos, external sources)
write findings only to their own isolated path:

```text
artifacts/research_agents/<task-id-or-slug>.md
```

or, for a subagent with multiple sub-tasks:

```text
artifacts/research_agents/<task-id-or-slug>/<subtask>.md
```

Never to a shared canonical file under `docs/` or to `CITATIONS.bib`
directly. This directory is many-writer-safe: each subagent owns a
distinct path, so there is no clobber risk regardless of how many run
concurrently.

The orchestrating agent (or a human) is the **single writer** for
canonical documents:

- `docs/PRIOR_WORK_MATRIX.md`
- `docs/RESEARCH_AUDIT.md`
- `docs/DATASET_AUDIT.md`
- `docs/MODEL_AUDIT.md`
- `CITATIONS.bib`

It reads each isolated research-agent file, verifies/reconciles the
content, and merges it into the canonical doc itself - one writer,
sequential edits, no race.

Files in this directory are working notes, not a second copy of the
canonical record. Once merged, they may be left in place as a raw
provenance trail (which subagent produced which finding) rather than
deleted - they are cheap, text-only, and small.
