# Local labeling UI for the commitment audits

This is a convenience layer for the two frozen commitment audits. It
runs entirely on this computer: a Python standard-library server bound to
`127.0.0.1`, plus one static page. There are no external assets, network
requests, models, suggestions or analytics.

## Launch

From the repository root:

```bash
uv run python scripts/labeling_ui/server.py
```

Then open http://127.0.0.1:8765/ and choose your audit and annotator letter.
Use `--port N` for another port.

(`python3 scripts/labeling_ui/server.py` also works; it needs only the
standard library.)

## What it does and does not do

- **What it reads.** It reads only the four blind sheets in
  `artifacts/labeling/commitment_audit_v1/` and
  `artifacts/labeling/commitment_audit_qwen_v1/` (`sheet_A.csv` and
  `sheet_C.csv`). It never opens any `key.csv`.
- **What it shows.** One row at a time, in the sheet's own order: the
  statement and the response. It shows no stage, prompt, judge output,
  stance-first pole, item coding or other annotator's labels.
- **What it saves.** Progress autosaves to
  `artifacts/labeling/_ui_drafts/<audit>_<annotator>.json`, a separate file
  per audit and annotator. The draft directory is gitignored. The frozen
  sheets are never written.
- **Export.** **Export completed CSV** validates the labels:
  - same rows, sample IDs and order;
  - unchanged statement and response text;
  - every relation, mode and confidence filled;
  - only taxonomy values.

  If they pass, it writes
  `artifacts/labeling/_ui_exports/<audit>_sheet_<annotator>.completed.csv`
  with exactly the columns
  `sample_id,statement,response,relation,mode,confidence_1to3,notes`.

**Submitting a sheet is your explicit, manual step.** After checking the
export, copy it over your own frozen sheet with the `cp` command the UI
prints. For example:

```bash
cp artifacts/labeling/_ui_exports/olmo_sheet_A.completed.csv artifacts/labeling/commitment_audit_v1/sheet_A.csv
```

Then run:

```bash
uv run python scripts/analyze_commitment_audit.py check --audit olmo
```

## Offline single files: one per annotator

`artifacts/labeling/standalone/annotator_A.html` and `annotator_C.html`
are self-contained pages. Send each annotator **only their own file**; they
open it by double-clicking. No server and no internet are needed.

- **What each file holds.** Only that annotator's two blind sheets (Audit 1
  and Audit 2): sample IDs, statements and responses, in the sheets' own
  row order. There are no keys, no hidden metadata and no other annotator's
  data.
- **No network.** A Content-Security-Policy inside the file allows only its
  own (hash-pinned) inline script and style, and sets `connect-src 'none'`.
  The page cannot make network requests.
- **Saving.** Progress is saved in that browser's local storage, separately
  per audit. **Download draft backup** and **Load draft backup** give a
  file copy, for example to move to another computer.
- **Export.** **Export completed CSV** validates completeness and taxonomy
  values, then downloads `<audit>_sheet_<annotator>.completed.csv`. The
  annotator sends that file to the researcher.

The researcher checks each returned file against the frozen sheet
(read-only), then replaces the sheet manually:

```bash
uv run python scripts/labeling_ui/verify_export.py olmo A path/to/olmo_sheet_A.completed.csv
cp path/to/olmo_sheet_A.completed.csv artifacts/labeling/commitment_audit_v1/sheet_A.csv   # only if VALID
```

(Use `qwen` and `artifacts/labeling/commitment_audit_qwen_v1/` for Audit 2.)

Rebuild the two files (deterministic; a test fails if the committed copies
are stale) with:

```bash
uv run python scripts/labeling_ui/build_standalone.py
```

## Keyboard shortcuts

| key | action |
|---|---|
| Q W E R T | relation: supports, opposes, mixed_or_conditional, neutral_or_no_position, unclear |
| A S D F G H J | mode: explicit_stance, analytical_exposition, voiced_continuation, refusal, empty, incomplete, other |
| 1 2 3 | confidence |
| ← → | previous / next row |
| Enter | next incomplete row |
| N | focus notes |
| Esc | leave notes |
