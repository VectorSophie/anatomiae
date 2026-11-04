# Human audit of commitment C on Qwen2.5-14B-Instruct outputs (self-judging check) — v1

**Status:** complete (2026-11-04). Measurement layer, **post-result
validation**. The Qwen selective-commitment result (`e14a823`) was known when
this sample was frozen.

**Protocol:** `docs/experiments/commitment_audit_qwen_v1.md` (seed `7101`).

**Data:** 100 Qwen2.5-14B-Instruct free-form responses, 20 random per prompt.
All 88 items are in scope; 57 of the 100 rows are class A. This sample
validates J-qy's commitment variable on its own model's outputs. It is **not**
a re-estimate of the selection gap. That is the separate, frozen follow-up
`selective_commitment_human_reestimate_qwen_v1`.

## Provenance

The provenance and pipeline are the same as the OLMo audit (`commitment_audit_v1.md`):
- every manifest SHA-256 matched;
- `verify_export.py` passed;
- the import was non-destructive;
- only `analyze` read the local key.

**Tables:**
- `artifacts/tables/commitment_audit_qwen_{human_agreement,metrics,pole_contrasts,measurement_effect}.csv`;
- `commitment_audit_qwen_report.md`;
- `commitment_audit_qwen_vs_olmo.csv`.

**Consensus:**
- 22 rows where A and C agreed;
- **78 researcher-adjudicated rows**, as expected. On the 62 relation
  conflicts the adjudication took A's label 30 times, C's 26 times and a
  third label 6 times.
- Consensus relation: mixed 37, neutral 25, opposes 22, supports 16,
  unclear 0.

## 1. Human A ↔ Human C

| dimension | n | agree | exact | Cohen κ |
|---|---:|---:|---:|---:|
| relation (5 classes) | 100 | 38 | 0.380 | 0.168 |
| binary commitment | 100 | 61 | 0.610 | 0.174 |
| response mode | 100 | 73 | 0.730 | −0.056 |
| direction, when both are directional | 14 | 13 | 0.929 | 0.837 |

**Human agreement on Qwen commitment is poor.**
- A labels 45/100 directional and C 22/100.
- They disagree on whether a response commits in 39/100 cases.
- When both see a side, they agree on direction 13/14 times.

The reference for this audit is therefore mostly researcher adjudication. The
A-only and C-only results (§4) bound how much the conclusions depend on it.

## 2. J-qy `C` vs adjudicated human consensus

| measure | Qwen (this audit) | OLMo audit, for comparison |
|---|---|---|
| human commitment | 38/100 = 0.380 | 37/150 = 0.247 |
| J-qy commitment | 39/100 = 0.390 | 46/150 = 0.307 |
| accuracy | 0.830 | 0.847 |
| Cohen κ | 0.641 | 0.619 |
| positive / negative agreement | 0.779 / 0.862 | 0.723 / 0.894 |
| over-read | 9/62 = 0.145 [0.078, 0.253] | 16/113 = 0.142 [0.089, 0.218] |
| under-read | 8/38 = 0.211 [0.111, 0.363] | 7/37 = 0.189 [0.095, 0.342] |
| supports ↔ opposes reversals, both directional | 1/30 | 0/30 |

Overall, J-qy is about as accurate on its own model's outputs as on OLMo's. Its
total commitment rate matches the consensus (−0.010 [−0.084, 0.073]). The one
reversal is a right-pole response.

## 3. Errors by stance-first pole

| pole | n | human commit | J-qy commit | over-read | under-read |
|---|---:|---:|---:|---:|---:|
| left | 82 | 34 (0.415) | 36 (0.439) | 9/48 = 0.188 [0.102, 0.319] | 7/34 = 0.206 [0.103, 0.368] |
| right | 15 | 4 (0.267) | 3 (0.200) | **0/11** = 0.000 [0.000, 0.259] | 1/4 = 0.250 [0.046, 0.699] |
| non-directional | 3 | 0 | 0 | 0/3 | — (0/0) |

**Left minus right:**

| contrast | difference | Newcombe 95% CI | Fisher p | bootstrap 95% CI |
|---|---:|---|---:|---|
| over-read | +0.188 | [−0.085, 0.319] | 0.19 | [0.077, 0.306]* |
| under-read | −0.044 | [−0.505, 0.217] | 1.00 | [−0.848, 0.310] |
| human commitment | +0.148 | [−0.124, 0.339] | 0.39 | [−0.182, 0.429] |
| J-qy commitment | +0.239 | [−0.033, 0.408] | 0.095 | [−0.064, 0.495] |

\* The right-pole over-read cell is 0/11. That rate is 0 in every bootstrap
replicate, so the bootstrap interval reflects only left-pole variability. The
Wilson-based Newcombe interval is the protocol-consistent one, and it includes
0.

The protocol says right-pole cells are not interpreted unless n ≥ 10:
- the right-pole over-read cell (n = 11) qualifies;
- the right-pole under-read cell (n = 4) does not.

### Evaluator measurement effect

| quantity (item-cluster bootstrap 95% CI) | consensus | A only | C only |
|---|---|---|---|
| human commitment asymmetry, left − right | +0.148 [−0.182, 0.429] | +0.063 [−0.273, 0.390] | +0.020 [−0.240, 0.226] |
| J-qy commitment asymmetry, left − right | +0.239 [−0.064, 0.494] | (same) | (same) |
| J-qy − human asymmetry | +0.091 [−0.051, 0.286] | +0.176 [−0.142, 0.541] | +0.220 [−0.002, 0.463] |
| human − J-qy commitment rate | −0.010 [−0.084, 0.073] | +0.060 [−0.030, 0.155] | −0.170 [−0.291, −0.058] |

## 4. Annotator sensitivity

| reference | human commit | accuracy | κ | over-read | under-read | over-read left vs right |
|---|---:|---:|---:|---:|---:|---|
| consensus | 38 | 0.830 | 0.641 | 9/62 | 8/38 | 9/48 vs 0/11 |
| A only | 45 | 0.760 | 0.509 | 9/55 | 15/45 | 8/44 vs 1/9 |
| C only | 22 | 0.730 | 0.384 | 22/78 | 5/22 | 21/64 vs 1/12 |

Under every reference, J-qy over-reads more on left-pole responses than on
right-pole ones. No Newcombe interval excludes 0.

The human commitment asymmetry shrinks as the reference changes: +0.15 under
the consensus, +0.06 under A and +0.02 under C. J-qy's asymmetry is +0.24.

## 5. The frozen reading rule, applied

The rule, from `commitment_audit_qwen_v1.md`: the Qwen measurement counts as
"not detectably inflated by self-judging" if **both** of these hold:
1. its over-read and under-read CIs overlap the OLMo audit's;
2. there is no pole-dependent error pattern large enough to account for a gap
   of the observed size.

Otherwise, the Qwen replication is reported as measurement-qualified.

**Condition 1: met.**
- Over-read: [0.078, 0.253] against OLMo's [0.089, 0.218].
- Under-read: [0.111, 0.363] against [0.095, 0.342].

Overall, the self-judge is no less accurate than on OLMo.

**Condition 2: not met.**
- The over-read pattern points in the direction that would inflate the gap:
  9/48 left-pole against 0/11 right-pole. J-qy also widens the commitment
  asymmetry relative to every human reference: +0.09, +0.18 and +0.22.
- None of these differences is statistically detectable at n = 100: the
  Newcombe CI is [−0.09, 0.32] and Fisher p = 0.19.
- But their upper bounds are large. The over-read difference could reach about
  0.3, and the J-qy-minus-human asymmetry about 0.29–0.54.
- Under human labels, the left−right commitment asymmetry in this sample is not
  distinguishable from 0 under any reference.

The audit therefore cannot show the absence of a pole-dependent error large
enough to account for a +0.32 gap. The rule requires that absence.

### Answer

> **Is the Qwen self-judge measurement detectably pole-dependent?** No, not
> detectably. By the protocol's Wilson-based contrasts, no pole-conditioned
> error difference excludes 0. The observed pattern is directionally
> consistent with inflation, however: left-pole over-read 0.19 vs right-pole
> 0/11. It is too imprecise to rule out an effect of the size that matters.
>
> **Is the original +0.32 therefore measurement-qualified?** **Yes.** Under the
> frozen rule, condition 2 is not established, so the Qwen2.5-14B-Instruct
> replication is reported as **measurement-qualified**. Its sign and its CI
> above 0 were obtained under J-qy measurement. Whether the magnitude survives
> human-labelled commitment is the question of the frozen re-estimation
> (`docs/experiments/selective_commitment_human_reestimate_qwen_v1.md`, `9776f7f`).
> This audit does not answer it.

## 6. Descriptive slices (consensus)

| prompt | n | human / J-qy commit | accuracy | κ | over-read | under-read | reversals |
|---|---:|---|---:|---:|---:|---:|---:|
| ff_assessment | 20 | 9 / 9 | 0.80 | 0.60 | 2/11 | 2/9 | 0 |
| ff_comment | 20 | 5 / 6 | 0.95 | 0.88 | 1/15 | 0/5 | 1 |
| ff_make_of | 20 | 6 / 8 | 0.80 | 0.57 | 3/14 | 1/6 | 0 |
| ff_thoughts | 20 | 13 / 11 | 0.80 | 0.59 | 1/7 | 3/13 | 0 |
| ff_where_stand | 20 | 5 / 5 | 0.80 | 0.47 | 2/15 | 2/5 | 0 |

## 7. Limitations

- **Poor human agreement on commitment** (κ 0.17). The consensus is mainly
  researcher adjudication. The queue carries no J-qy or key field, but what the
  adjudicator saw otherwise cannot be verified here.
- **Coarse right-pole cells:** 15 rows, 4 committed by the consensus.
- **Not class A only:** the population is all 88 items, so the pole composition
  differs from the primary analysis.
