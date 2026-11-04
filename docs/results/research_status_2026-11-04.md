# anatomiae — consolidated research status (2026-11-04)

This supersedes `research_status_2026-10-14.md` for current state. The older
document is kept unchanged as the record of the decision rule (its §6) that
triggered the Qwen human re-estimation.

**Branch:** `research/human-commitment-audit-closeout`.

## 0. What changed since 2026-10-14

1. **The human audits' completed labels are not yet on the analysis
   workstation.** Searched for and not found (2026-11-04: `/`, `/data`,
   `/mnt/usb`):
   - `olmo_sheet_A.completed.csv`
   - `olmo_sheet_C.completed.csv`
   - `qwen_sheet_A.completed.csv`
   - `qwen_sheet_C.completed.csv`
   - `adjudication_queue_olmo.completed.csv`
   - `adjudication_queue_qwen.completed.csv`

   The in-repo `sheet_{A,C}.csv` files are the blank frozen sheets: 150/150
   and 100/100 relations empty. **No audit result is reported here.** The keys
   (`key.csv`, local-only) are present.
2. **The audit pipeline is ready** (`0a807f9`):
   - non-destructive import of completed exports;
   - a tolerant but strict adjudication import;
   - consensus;
   - extended keyed analysis: pole contrasts with counts, Newcombe CI, Fisher
     exact p and item-cluster bootstrap; the evaluator measurement effect;
     A-only and C-only sensitivity;
   - the Qwen-vs-OLMo comparison.

   `check` and `consensus` are tested never to open the key.
3. **The Qwen human selection-gap re-estimation is frozen** (`9776f7f`) before
   any new label: `docs/experiments/selective_commitment_human_reestimate_qwen_v1.md`.
4. **A manuscript draft exists**, with generated tables and figures: `paper/`.

## 1. Erratum: two-decimal rounding of headline numbers

The full-precision values were recomputed from the caches with the frozen code
(`selective_commitment_v1.gap`). Some previously documented two-decimal values
were rounded twice (via the 4-dp tables), e.g. 0.2346 → 0.235 → 0.24.

| quantity | documented | correct (full precision) |
|---|---|---|
| OLMo-SFT gap, class A | +0.24 | **+0.23** (0.2346) [0.05, 0.41] |
| R1-Distill gap CI lower bound, class A | 0.15 | **0.14** (0.1449) |
| Qwen-Instruct gap, all items | +0.37 | **+0.36** (0.3647) |
| Qwen-Instruct gap CI upper bound, class A | 0.49 | 0.49 (0.48501; correct) |

**No decision changes.** The frozen results documents are left as committed;
this erratum is the correction record. The paper tables round once, half-up,
from the stored values (`scripts/make_paper_artifacts.py`, tested).

## 2. Confirmed / replicated (J-qy measurement)

Selection gap: class A, 5 free-form prompts pooled, 95% item bootstrap.

| checkpoint | lineage | gap [95% CI] | commit rate | evidence class |
|---|---|---|---:|---|
| OLMo-2-13B-SFT | OLMo 2 | +0.23 [0.05, 0.41] | 0.28 | confirmatory (preregistered) |
| OLMo-2-13B-DPO | OLMo 2 | +0.33 [0.15, 0.52] | 0.30 | confirmatory (preregistered) |
| OLMo-2-13B-RLVR2 | OLMo 2 | +0.50 [0.27, 0.72] | 0.31 | confirmatory (preregistered) |
| Qwen2.5-14B-Instruct | Qwen2.5 | +0.32 [0.16, 0.49] | 0.34 | prospective replication (primary) |
| DeepSeek-R1-Distill-Qwen-14B | Qwen2.5 | +0.34 [0.14, 0.54] | 0.39 | prospective replication (secondary) |

These are **five checkpoints across two lineages, not five independent model
families.** Source: `paper/tables/table1_selective_commitment.md`, generated.

## 3. Human validation: what is known, and what is pending

**Known (difficult 150-sample, `human_validation_v1.md`, `eqy_v1.md`):**
- Human A↔C disagreement is about commitment. Among responses both call
  directional, they agree on direction 97/100.
- J-qy, shown the statement, has 0/117 reversals on human-directional
  responses, but over-reads 19/26 human non-directional ones. The residual
  over-read sits where one human also saw that direction.

**Pending (representative audits; labels exist but are not yet imported):**
- the OLMo commitment audit (150);
- the Qwen self-judging audit (100).

The prompt for this session quoted a provisional pre-analysis from outside the
repository: human-directional 37/150 (OLMo) and 38/100 (Qwen); J-qy accuracy
≈ 0.85 / 0.83; κ ≈ 0.62 / 0.64; pole-dependent error in Qwen. **These are not
reproduced here and are not results.** The keyed analysis is authoritative.

**Interpretation, conditional:**
- The status-document wording is fixed only after `analyze`.
- If the keyed results match the pre-analysis:
  - evaluator error is mainly at the commitment boundary;
  - polarity is stable once both agree a response is directional;
  - the OLMo asymmetry survives human inspection;
  - the Qwen J-qy magnitude stays measurement-qualified until §4 returns.

## 4. The one remaining follow-up: Qwen human re-estimation (frozen)

| | |
|---|---|
| verified population (class A) | 79 left / 10 right / 3 non-directional item-polarities |
| sample | all 10 right + all 3 non-directional + 20 left (seed 20261103), × 5 prompts |
| clusters / rows | 33 / 165 |
| reused labels (by `cache_key`) | 23 rows: A, C and consensus from the Qwen audit |
| **new labels needed** | **142 rows × 2 annotators** |
| label-free design check (J-qy, same units) | +0.26 [0.09, 0.49] (population +0.32) |

Annotator files:
- Annotator A: `artifacts/labeling/standalone/selection_gap_qwen_annotator_A.html`
- Annotator C: `artifacts/labeling/standalone/selection_gap_qwen_annotator_C.html`

  **Before sending page C**, the researcher reviews the 33 drafted French
  statement translations in
  `artifacts/labeling/selection_gap_human_qwen_v1/statements_fr.csv`, which are
  unvalidated.

## 5. Order of operations from here

1. Import the six completed files:
   ```bash
   uv run python scripts/labeling_ui/verify_export.py olmo A <file>
   uv run python scripts/analyze_commitment_audit.py import-sheet --audit olmo --annotator A <file>
   # likewise for olmo C, qwen A, qwen C
   uv run python scripts/analyze_commitment_audit.py check --audit olmo   # and --audit qwen
   uv run python scripts/analyze_commitment_audit.py import-adjudication --audit olmo adjudication_queue_olmo.completed.csv
   uv run python scripts/analyze_commitment_audit.py consensus --audit olmo   # and qwen
   uv run python scripts/analyze_commitment_audit.py analyze --audit olmo     # and qwen
   uv run python scripts/analyze_commitment_audit.py compare
   uv run python scripts/make_paper_artifacts.py
   ```
   Then:
   - check that `check` reports 95 (OLMo) and 78 (Qwen) disagreement rows;
   - write `docs/results/commitment_audit_v1.md` and
     `commitment_audit_qwen_v1.md`, including the frozen Qwen reading rule.
2. In parallel, send the follow-up pages to A and C (142 rows each). Then:
   ```bash
   uv run python scripts/analyze_commitment_audit.py import-sheet --audit qwen_gap --annotator A <file>
   # likewise for C; then check, adjudicate, consensus --audit qwen_gap
   uv run python scripts/selection_gap_human_qwen_v1.py analyze
   ```
   Write `docs/results/selective_commitment_human_reestimate_qwen_v1.md`.
3. Finalize the manuscript's §9–10, the abstract and the headline claim
   according to the frozen reading rules.

## 6. Recommendation

**No further model or GPU experiment.** None of the open questions is about
generation. Both concern human measurement of `C`, and they are answered by
labels on existing outputs.

Adding model families would increase the count of J-qy-measured checkpoints.
It would not address the one reviewer-critical question: whether the
commitment variable's errors depend on the pole.

Stopping rule: finish the human-grounded Qwen re-estimate, then finalize the
manuscript according to what that estimate supports.
