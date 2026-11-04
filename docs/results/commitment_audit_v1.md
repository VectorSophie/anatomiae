# Human audit of the commitment indicator C — OLMo-2-13B (v1)

**Status:** complete (2026-11-04). Measurement layer, post-result validation.

**Protocol:** `docs/experiments/commitment_audit_v1.md`. The sample was frozen at
`ccabd8c` (seed label `20261014`).

**Data:** 150 OLMo-2-13B free-form responses from the confirmatory run, 10 random
per stage × prompt cell. The sample is representative of the 2,640 free-form
responses: all 88 items, 82 of the 150 rows class A.

**What this is:** a validation of J-qy's commitment variable `C`.

**What this is not:** a re-estimate of the selection gap.

## Provenance

The completed files were delivered in `anatomiae_human_audit_handoff_2026-11-04.zip`.
- The SHA-256 of every file matched the supplied manifest before import.
- Each sheet passed `verify_export.py` against the frozen blind sheet.
- Imported non-destructively (`imports.json` records the hashes):
  - `artifacts/labeling/commitment_audit_v1/sheet_{A,C}.completed.csv`;
  - `adjudication_queue.completed.csv`.
- The blank frozen sheets are unchanged.

**Pipeline:** `scripts/analyze_commitment_audit.py` `check` → `import-adjudication` →
`consensus` → `analyze`. Only `analyze` opened the local key (`key.csv`,
gitignored).

**Tables:**
- `artifacts/tables/commitment_audit_olmo_{human_agreement,metrics,pole_contrasts,measurement_effect}.csv`;
- `commitment_audit_olmo_report.md`.

**Consensus:**
- 55 rows where A and C agreed on relation and mode;
- 95 researcher-adjudicated rows, as expected. On the 69 relation conflicts
  the adjudication took A's label 26 times, C's 38 times and a third label
  5 times.
- Consensus relation: mixed 59, neutral 54, opposes 24, supports 13, unclear 0.

**Definition, as frozen:** `C_human = 1` iff the relation ∈ {supports, opposes}.

## 1. Human A ↔ Human C

| dimension | n | agree | exact | Cohen κ |
|---|---:|---:|---:|---:|
| relation (5 classes) | 150 | 81 | 0.540 | 0.323 |
| binary commitment | 150 | 117 | 0.780 | 0.327 |
| response mode | 150 | 106 | 0.707 | 0.218 |
| direction, when both are directional | 14 | 14 | 1.000 | 1.000 |

**The two humans differ mostly on commitment.**
- A labels 36/150 directional and C 25/150. Only 14 responses are directional
  for both.
- They disagree on whether a response commits in 33/150 cases.
- When both see a side, they never disagree on which side.

Agreement is lower than on the difficult v1 validation sample (relation κ 0.55).
Representative free-form OLMo responses are mostly analytical prose (consensus
mode: analytical exposition 121/150), where the commitment boundary is
contested.

## 2. J-qy `C` vs adjudicated human consensus

| measure | value |
|---|---|
| human commitment | 37/150 = 0.247 |
| J-qy commitment | 46/150 = 0.307 |
| accuracy | 0.847 |
| Cohen κ | 0.619 |
| positive / negative agreement | 0.723 / 0.894 |
| over-read (J-qy commits, human does not) | 16/113 = 0.142 [0.089, 0.218] |
| under-read (human commits, J-qy does not) | 7/37 = 0.189 [0.095, 0.342] |
| supports ↔ opposes reversals, both directional | **0/30** |

Intervals are 95% Wilson CIs. J-qy commits slightly more often than the
consensus: −0.060 [−0.118, 0.000] (human minus J-qy, item-cluster bootstrap).
Its errors are on the commitment boundary. **Direction is never reversed.**

## 3. The audit's key question: errors by stance-first pole

The stance-first pole of each response's item-polarity, from the confirmatory
run (local key):

| pole | n | human commit | J-qy commit | over-read | under-read |
|---|---:|---:|---:|---:|---:|
| left | 111 | 35 (0.315) | 43 (0.387) | 13/76 = 0.171 [0.103, 0.271] | 5/35 = 0.143 [0.063, 0.294] |
| right | 34 | 2 (0.059) | 3 (0.088) | 3/32 = 0.094 [0.032, 0.242] | 2/2 = 1.000 [0.342, 1.000] |
| non-directional | 5 | 0 | 0 | 0/5 | — (0/0) |

**Left minus right.** Each contrast is reported three ways:
- a Newcombe hybrid-score 95% CI (Wilson-based, primary);
- a Fisher exact p (descriptive);
- an item-cluster bootstrap (2,000, seed 0).

| contrast | difference | Newcombe 95% CI | Fisher p | bootstrap 95% CI |
|---|---:|---|---:|---|
| over-read | +0.077 | [−0.086, 0.195] | 0.38 | [−0.063, 0.206] |
| under-read | −0.857 | [−0.937, −0.182] | 0.03 | undefined (right cell n = 2) |
| human commitment | +0.256 | [0.103, 0.357] | 0.002 | [0.129, 0.384] |
| J-qy commitment | +0.299 | [0.134, 0.409] | 0.001 | [0.143, 0.450] |

**Reading under the frozen rule.** The gap would be an evaluator artefact only if
J-qy over-read more on left-pole items or under-read more on right-pole ones.

- **Over-read:** slightly higher on the left pole (0.17 vs 0.09). The difference
  is not detectable.
- **Under-read on the right pole:** 2/2. Both right-pole responses the consensus
  calls committed were missed by J-qy. That is the artefact direction, but the
  cell has 2 rows and is not interpretable.
- **The decisive point:** the selective pattern is present under **human**
  measurement.
  - Humans see a side in 35/111 left-pole responses but 2/34 right-pole
    responses: +0.26 [0.10, 0.36].
  - J-qy's asymmetry (+0.30) is close to it.

### Does the evaluator preserve, attenuate or amplify selective commitment?

| quantity (item-cluster bootstrap 95% CI) | consensus | A only | C only |
|---|---|---|---|
| human commitment asymmetry, left − right | +0.256 [0.129, 0.384] | +0.162 [0.009, 0.297] | +0.148 [0.015, 0.272] |
| J-qy commitment asymmetry, left − right | +0.299 [0.143, 0.450] | (same) | (same) |
| J-qy − human asymmetry | +0.043 [−0.083, 0.146] | +0.138 [−0.006, 0.294] | +0.151 [−0.005, 0.312] |
| human − J-qy commitment rate | −0.060 [−0.118, 0.000] | −0.067 [−0.155, 0.021] | −0.140 [−0.206, −0.078] |

**J-qy preserves the asymmetry relative to the consensus.** Relative to either
single annotator it may amplify it, by about +0.14 to +0.15, with CIs touching
0. Under every human reference, the asymmetry is still positive with its CI
above 0.

## 4. Annotator sensitivity (A-only, C-only; not competing ground truths)

| reference | human commit | accuracy | κ | over-read | under-read | reversals |
|---|---:|---:|---:|---:|---:|---:|
| consensus | 37 | 0.847 | 0.619 | 16/113 | 7/37 | 0/30 |
| A only | 36 | 0.733 | 0.332 | 25/114 | 15/36 | 0/21 |
| C only | 25 | 0.807 | 0.479 | 25/125 | 4/25 | 0/21 |

Over-read by pole is left > right under each annotator:
- A: 23/80 vs 2/30, Newcombe [0.05, 0.34];
- C: 22/88 vs 3/32, Newcombe [−0.01, 0.27].

Each annotator therefore suggests somewhat more left-pole over-read than the
consensus does. The human commitment asymmetry stays positive under each
(§3).

**Caveat:** J-qy's agreement with the consensus (κ 0.62) exceeds the
humans' agreement with each other on commitment (κ 0.33). Most of the
consensus on contested rows is
researcher adjudication. The adjudication queue carries no J-qy output or key
field, but this repository cannot verify what else the adjudicator saw. The
consensus agreement figures should be read alongside the A-only and C-only rows.

## 5. Descriptive slices (consensus)

| slice | n | human / J-qy commit | accuracy | κ | over-read | under-read |
|---|---:|---|---:|---:|---:|---:|
| SFT | 50 | 12 / 16 | 0.80 | 0.51 | 7/38 | 3/12 |
| DPO | 50 | 14 / 16 | 0.88 | 0.72 | 4/36 | 2/14 |
| RLVR2 | 50 | 11 / 14 | 0.86 | 0.63 | 5/39 | 2/11 |
| ff_assessment | 30 | 10 / 16 | 0.67 | 0.35 | 8/20 | 2/10 |
| ff_comment | 30 | 10 / 9 | 0.90 | 0.77 | 1/20 | 2/10 |
| ff_make_of | 30 | 5 / 9 | 0.80 | 0.45 | 5/25 | 1/5 |
| ff_thoughts | 30 | 10 / 9 | 0.90 | 0.77 | 1/20 | 2/10 |
| ff_where_stand | 30 | 2 / 3 | 0.97 | 0.78 | 1/28 | 0/2 |

Over-read concentrates in `ff_assessment` (8/20). `ff_where_stand` elicits
almost no commitment under either measure (2 and 3 of 30), consistent with the
confirmatory coverage table.

## 6. Conclusion

- **J-qy is a reasonable commitment instrument for OLMo free-form outputs:**
  accuracy 0.85, κ 0.62, no direction reversals. Its errors sit on the
  commitment boundary, where the two humans themselves disagree.
- **There is no detectable pole-dependent over-read.**
- **The selective-commitment pattern is present under human labels.**
  Right-pole responses are rarely judged committed by humans (2/34) compared
  with left-pole ones (35/111). That holds under the consensus, under A alone
  and under C alone.
- **Within this audit's precision, the OLMo selection result is not an
  evaluator artefact.**
- **What remains open:** the right-pole under-read cell (2/2), with n = 2. It
  is too small to evaluate, and only a pole-stratified sample could.

This representative audit does not re-estimate the OLMo selection gaps.
