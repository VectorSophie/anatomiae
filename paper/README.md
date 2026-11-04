# Manuscript draft

- `manuscript.md` is the working draft (Markdown with pandoc citations against `../CITATIONS.bib`). Convert it
  with pandoc to a venue template; nothing venue-specific is in the text.
- `tables/` and `figures/` are **generated**. Do not edit them; run
  `uv run python scripts/make_paper_artifacts.py`. Audit-dependent artifacts are written as explicit
  "pending" stubs, and they fill in automatically once the source tables exist.
- Markers in the draft:
  - `[PENDING-AUDIT]`: depends on the human commitment audits;
  - `[PENDING-REESTIMATE]`: depends on the frozen Qwen re-estimate;
  - `[CITE NEEDED]`: needs a verified reference that is not yet in `CITATIONS.bib`.
