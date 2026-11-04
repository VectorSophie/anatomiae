# anatomiae — consolidated research status (2026-11-04)

This supersedes `research_status_2026-10-14.md` for current state. The older
document is kept unchanged as the record of the decision rule (its §6) that
triggered the Qwen human re-estimation.

**Branch:** `research/human-commitment-audit-closeout`.

## 0. What changed since 2026-10-14

1. **Both human commitment audits are closed** (`114ac04`).
   - The six completed files arrived in `anatomiae_human_audit_handoff_2026-11-04.zip`.
     They were not committed; the canonical CSVs were imported through the
     pipeline.
   - Every file matched the supplied manifest SHA-256 and passed `verify_export`.
   - The import was non-destructive: the blank frozen sheets are unchanged.
   - The adjudication queues had 95 and 78 rows, as expected.
   - Only `analyze` opened the local keys, which stay gitignored.
   - Results: `commitment_audit_v1.md` and `commitment_audit_qwen_v1.md`.

   An earlier pass of this session found none of the six files on the
   workstation. That gap is now closed.
2. **Audit pipeline** (`0a807f9`, plus a CLI fix in `114ac04`):
   - non-destructive import, adjudication import and consensus;
   - extended keyed analysis;
   - the cross-audit comparison.

   `check` and `consensus` are tested never to open the key.
3. **The Qwen human selection-gap re-estimation is frozen** (`9776f7f`). It was
   frozen before any new label and before the audit analysis:
   `docs/experiments/selective_commitment_human_reestimate_qwen_v1.md`. The
   design is unchanged.
4. **The manuscript integrates the audit results** (`dc8753e`): `paper/`.

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

## 3. Human validation

**Difficult 150-response sample** (`human_validation_v1.md`, `eqy_v1.md`):
- Human A↔C disagreement is about commitment. Among responses both call
  directional, they agree on direction 97/100.
- J-qy, shown the statement, makes 0/117 reversals.

**Representative audits** (closed 2026-11-04; consensus reference; Wilson 95% CIs):

| | OLMo (150) | Qwen-Instruct (100) |
|---|---|---|
| A↔C commitment agreement, κ | 117/150, 0.33 | 61/100, 0.17 |
| A↔C relation κ / mode κ | 0.32 / 0.22 | 0.17 / −0.06 |
| A↔C direction, both directional | 14/14 | 13/14 |
| human / J-qy commitment | 37 / 46 | 38 / 39 |
| accuracy, κ | 0.847, 0.619 | 0.830, 0.641 |
| positive / negative agreement | 0.72 / 0.89 | 0.78 / 0.86 |
| over-read | 16/113 [0.09, 0.22] | 9/62 [0.08, 0.25] |
| under-read | 7/37 [0.09, 0.34] | 8/38 [0.11, 0.36] |
| reversals / both directional | 0/30 | 1/30 |

**Pole-conditioned results** (left vs right; Newcombe 95% CI):

| | OLMo | Qwen-Instruct |
|---|---|---|
| over-read | 13/76 vs 3/32: +0.08 [−0.09, 0.19] | 9/48 vs **0/11**: +0.19 [−0.09, 0.32] |
| under-read | 5/35 vs 2/2 (n = 2, not interpreted) | 7/34 vs 1/4 (n = 4, not interpreted) |
| human commitment, left − right | 35/111 vs 2/34: **+0.26 [0.10, 0.36]** | 34/82 vs 4/15: +0.15 [−0.12, 0.34] |
| J-qy commitment, left − right | +0.30 [0.13, 0.41] | +0.24 [−0.03, 0.41] |
| J-qy − human asymmetry (bootstrap) | +0.04 [−0.08, 0.15] | +0.09 [−0.05, 0.29] |

**Interpretation:**
- **Evaluator error is on the commitment boundary.** Direction is stable once
  both the human reference and J-qy see a side: 0/30 and 1/30 reversals.
- **OLMo.** The selective-commitment asymmetry is present under human labels:
  consensus, A alone (+0.16) and C alone (+0.15), all CIs above 0. J-qy
  preserves it. There is no detectable pole-dependent over-read.
- **Qwen.** No pole-conditioned error difference is statistically detectable.
  But J-qy over-reads only on left-pole responses (0/11 on the right pole), and
  it widens the asymmetry relative to every human reference (+0.09 / +0.18 /
  +0.22). Under human labels the asymmetry is not distinguishable from 0.

  **Frozen reading rule:**
  - condition 1 (error CIs overlap OLMo's) is met;
  - condition 2 (no pole-dependent error large enough to account for the gap)
    is not established.

  So the Qwen +0.32 is **measurement-qualified**. The frozen re-estimation
  decides its magnitude.
- **Human references.** Inter-human commitment agreement is low (κ 0.17–0.33).
  The consensus is mainly researcher adjudication, so A-only and C-only results
  are reported throughout.

The provisional pre-analysis quoted in this session's instructions is matched
on every headline value: 37/150, 38/100, accuracy 0.847 / 0.830, κ 0.619 /
0.641. The keyed pipeline is the source.

## 4. The one remaining follow-up: Qwen human re-estimation (frozen, `9776f7f`)

The design is unchanged after the audit results.

| | |
|---|---|
| verified population (class A) | 79 left / 10 right / 3 non-directional item-polarities |
| sample | all 10 right + all 3 non-directional + 20 left (seed 20261103), × 5 prompts |
| clusters / rows | 33 / 165 |
| reused labels (by `cache_key`, verified against the audit key) | 23 rows (13 left, 8 right, 2 non-directional): A, C and consensus from the Qwen audit |
| **new labels needed** | **142 from A and 142 from C**, then adjudication of their disagreements |
| label-free design check (J-qy, same units) | +0.26 [0.09, 0.49]. The population +0.32 is the original reference, not a paired comparator. |

**Primary comparison:** the weighted human gap minus the weighted J-qy gap on
the same sampled units.

**Pipeline readiness.** A test runs `selection_gap_human_qwen_v1.py analyze` end
to end with synthetic placeholder labels, in a temporary directory. It
reproduces the +0.262 J-qy comparator and the +0.316 census.

**Files to send:**
- **Annotator A:** `artifacts/labeling/standalone/selection_gap_qwen_annotator_A.html`
  (142 rows, English).
- **Annotator C:** `artifacts/labeling/standalone/selection_gap_qwen_annotator_C.html`
  (142 rows, French UI).
  - 14/33 statement translations are reused verbatim from the earlier human-C
    French rerun (`prior_C_translation_reuse_reference.csv`; provenance
    `reused_from_human_C_retry150_FR_completed`).
  - **19/33 are drafts that need researcher review before sending**
    (provenance `draft_by_claude_2026-11-04_unvalidated_needs_review` in
    `statements_fr.csv`).
  - Translation reuse is display-only. The rerun is not an annotation source
    for any follow-up row.

## 5. Order of operations from here

1. Review the 19 draft translations. If any is changed, rebuild with
   `uv run python scripts/labeling_ui/build_standalone.py`, run the tests and
   commit **before** C starts.
2. Send the two pages. When the exports return:
   ```bash
   uv run python scripts/labeling_ui/verify_export.py qwen_gap A <file>    # and C
   uv run python scripts/analyze_commitment_audit.py import-sheet --audit qwen_gap --annotator A <file>   # and C
   uv run python scripts/analyze_commitment_audit.py check --audit qwen_gap
   # researcher fills artifacts/labeling/selection_gap_human_qwen_v1/adjudication_queue.csv (a copy)
   uv run python scripts/analyze_commitment_audit.py import-adjudication --audit qwen_gap <completed queue>
   uv run python scripts/analyze_commitment_audit.py consensus --audit qwen_gap
   uv run python scripts/selection_gap_human_qwen_v1.py analyze
   uv run python scripts/make_paper_artifacts.py
   ```
3. Write `docs/results/selective_commitment_human_reestimate_qwen_v1.md` and
   apply the frozen reading rule. Finalize the manuscript's §10, the abstract
   and the headline.

## 6. Recommendation

**No further model or GPU experiment.** None of the open questions is about
generation. Both concern human measurement of `C`, and they are answered by
labels on existing outputs.

Adding model families would increase the count of J-qy-measured checkpoints.
It would not address the one reviewer-critical question: whether the
commitment variable's errors depend on the pole.

Stopping rule: finish the human-grounded Qwen re-estimate, then finalize the
manuscript according to what that estimate supports.
