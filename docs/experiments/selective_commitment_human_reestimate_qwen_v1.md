# Protocol: Qwen2.5-14B-Instruct selection gap re-estimated with human-labelled commitment — v1

**Status:** frozen 2026-11-04, **before any new human label exists**. The freeze
commit is the commit that adds this file. Its SHA is recorded in the next commit
under "Freeze record" below, because a file cannot contain the SHA of the commit
that adds it.

**Evidence class:** a frozen follow-up triggered by the pre-specified decision
rule of `docs/results/research_status_2026-10-14.md` §6. It is **not**
preregistered with respect to the Qwen selective-commitment result
(`e14a823`). That result, the Qwen J-qy outcomes and the stance-first poles are
all already known. What is unknown at freezing is every human label on the 142 new
rows.

This is a measurement experiment. It generates nothing and judges nothing. It uses only the
cached Qwen2.5-14B-Instruct responses and the J-qy and rule-based outcomes of
`selective_commitment_crosslineage_v1`.

## 0. Why, and how it was triggered

In the Qwen arm, the commitment indicator `C` comes from J-qy, which is
Qwen2.5-14B-Instruct judging its own outputs. The representative Qwen audit
(`commitment_audit_qwen_v1.md`, 100 responses) can detect whether J-qy's
commitment errors depend on the stance-first pole. With only about 10% right-pole
rows, it cannot re-estimate the selection gap.

Status §6 fixed the next step in advance. If the audits show pole-dependent J-qy
errors, the single remaining experiment is to re-estimate the gap with
human-labelled `C` on a sample stratified on the stance-first pole.

**Trigger, stated exactly.** The completed Qwen audit sheets and adjudication
exist with the researcher. At freezing they are **not on the analysis
workstation**, so the keyed audit analysis has not been run in this repository.
The trigger is the researcher's provisional pre-analysis of those sheets, which
reported pole-dependent evaluator error. The formal check of the audit's frozen
reading rule will be recorded in `docs/results/commitment_audit_qwen_v1.md` once
the sheets are imported.

**No optional stopping.** This follow-up runs and is reported **whatever** the
formal audit check shows. If that check finds no pole-dependent error, the
result is reported as a robustness check rather than a correction.

## 1. Question

> Does the Qwen2.5-14B-Instruct selective-commitment gap survive when the
> commitment variable itself is human-labelled?

## 2. Population: identical to the primary Qwen replication

- **Model:** Qwen2.5-14B-Instruct @ `cf98f3b3`, outputs from
  `artifacts/cache/selective_commitment_xl_v1.jsonl`. Nothing is regenerated.
- **Items and prompts:** the 46 class-A items × 2 polarities = 92 item-polarities,
  each under the 5 frozen free-form prompts: **460 units**.
- **Stance-first reference:** the J-qy outcome on the frozen stance-first
  response of the same item-polarity (`SF`), as in the primary analysis.
- **Political coding:** Faulborn's item lean, flipped for `inv`. Unchanged.

**Stratum: the stance-first pole of the item-polarity**, as in every
`P(commit | SF pole)` table. It is `left_pole` if the J-qy SF agrees with a
left-coded or disagrees with a right-coded statement, `right_pole` for the
reverse, and `non_directional` otherwise.

**Verified from the cache at freezing** (the expected counts matched):

| stratum | item-polarities | units |
|---|---:|---:|
| left_pole | 79 | 395 |
| right_pole | 10 | 50 |
| non_directional | 3 | 15 |

## 3. Sample, frozen

- **Unit of sampling:** the item-polarity cluster. Every sampled cluster
  contributes **all five** free-form responses. There are no partial clusters.
- **Draw:**
  - all 10 right-pole clusters;
  - all 3 non-directional clusters;
  - **20 of the 79 left-pole clusters**, drawn uniformly without replacement
    with `pandas.DataFrame.sample(20, random_state=20261103)` from the
    left-pole clusters sorted by `(item_id, polarity)`.
- **Size:** 33 clusters, **165 responses**.
- **Generator:** `scripts/selection_gap_human_qwen_v1.py sample`. It never
  overwrites an existing key. A test regenerates the sample from the public
  cache and checks that it reproduces the committed sheets byte for byte.

### Reuse of existing human labels

Rows are matched to the completed Qwen audit (`commitment_audit_qwen_v1`) by the
**immutable generation `cache_key`**, never by text. For every matched row, the
generator also asserts that the audit sheet's statement and response are
identical to the cached text.

| | rows |
|---|---:|
| total follow-up rows | 165 |
| reused: A label, C label and adjudicated consensus from `commitment_audit_qwen_v1` (`reused_from_commitment_audit_qwen_v1`) | 23 |
| **new: need an A label and a C label** (`new_selection_gap_human_qwen_v1`) | **142** |

Reused rows keep their original A label, C label and consensus. They are never
shown again to either annotator. Disclosure: those 23 rows were labelled in a
different context (mixed with non-class-A items). The new rows are labelled
together.

## 4. Annotation

- **Taxonomy, instructions, blinding:** identical to the two commitment audits
  (`relation`, `mode`, `confidence_1to3`, `notes`).
- **What is hidden:** model identity, prompt identity, J-qy output, the
  stance-first response, the pole or stratum, the political coding, and
  whether a row is new or reused (only new rows exist in the sheets).
- **Sheets:** `artifacts/labeling/selection_gap_human_qwen_v1/sheet_A.csv` and
  `sheet_C.csv`. They hold the same 142 rows (`QG001`–`QG142`) in independent
  random orders.
- **Offline pages:** `artifacts/labeling/standalone/selection_gap_qwen_annotator_A.html`
  (English) and `selection_gap_qwen_annotator_C.html`, both built by
  `scripts/labeling_ui/build_standalone.py`. They work offline, with a CSP
  `connect-src 'none'`, localStorage autosave, a draft backup and a validated
  CSV export.
- **Annotator C's French presentation** follows the human-validation v1
  procedure:
  - French UI and instructions;
  - the French statement as the main display, with the English original in a
    collapsible panel that is authoritative when in doubt;
  - the response in its original English.
- **About the translation.** It is display-only. The exported CSV holds the
  canonical English statement and the English taxonomy values (tested). No
  validated translation from the earlier C rerun exists in the repository, so
  the 33 statement translations (`statements_fr.csv`) were drafted on
  2026-11-04 and are marked unvalidated. **The researcher reviews them before
  sending the page.** Any correction is committed, and the page rebuilt,
  before C begins.
- **Labels are never prefilled**, and reused labels are never shown.
- **The key** (`key.csv`) holds `cache_key`, item, polarity, prompt, lean, J-qy
  outcome, `C_jqy`, J-qy and rule-based SF, stratum, label source and label
  sample id. It is gitignored until both new sheets are complete.
- **The public manifest** (`manifest.json`) holds counts and sheet hashes only.

**Import, adjudication and consensus** use the same code path as the audits
(`scripts/analyze_commitment_audit.py` with `--audit qwen_gap`):
- `import-sheet`: a non-destructive, byte-identical copy;
- `check`: no key access; writes the adjudication queue;
- `import-adjudication`, then `consensus`: a researcher-adjudicated consensus.

A and C are never overwritten. The consensus is not a third independent
annotator.

## 5. Estimand and estimator

**Estimand: the frozen primary selection gap of the Qwen replication, with
exactly one change: `C_J-qy` → `C_human`.**

    gap = D_SF(units with C_human = 1) − D_SF(units with C_human = 0)

- class A, the 5 free-form prompts pooled, unit-weighted;
- `D_SF` is the existing direction index (agreement rate on left-coded minus
  agreement rate on right-coded statements, among directional J-qy SF
  outcomes);
- `C_human = 1` iff the consensus relation ∈ {supports, opposes}. `unclear`,
  mixed and neutral count as 0, as in the audits.

**Estimator.** The sample is pole-stratified, so raw sample proportions are
**not** population quantities. Each unit carries its stratum weight
`N_h / n_h`:

| stratum | weight |
|---|---|
| left_pole | 79/20 = 3.95 |
| right_pole | 1 |
| non_directional | 1 |

Every ratio of sums in the frozen gap is computed with these weights (a Hájek
estimator).

With a census sample (all clusters, weight 1) the estimator reproduces the frozen
J-qy result **exactly**: +0.3158 [0.1561, 0.4850]. This is tested against
`selective_commitment_v1.gap`.

**Interval.** This is the frozen item bootstrap:
- the 46 class-A items, each item's orig and inv clusters and all prompts
  travelling together;
- 2,000 resamples, `numpy.random.default_rng(0)`, the identical draws;
- 95% percentile interval.

It is run as a **two-phase bootstrap**: every replicate recomputes
`N_h*/n_h*` from the resampled items. A replicate in which a stratum is present
in the population but absent from the sample is undefined. The interval is NaN
if more than 10% of replicates are undefined (the project rule).

**Minimum cell size, as frozen.** The gap is estimable only if the committed and
the uncommitted subsets each hold at least 10 **sampled** units with a
directional SF. A non-estimable gap counts as "does not survive".

## 6. Primary result and reading rule (fixed now)

**Primary:** `gap_human` with its 95% two-phase cluster-bootstrap CI.

**Paired comparator:** `gap_J-qy` computed by the same estimator on **the same
165 units**, and `Δ = gap_human − gap_J-qy`, using the same replicate draws.

1. **Survival.** The Qwen selective-commitment gap *survives human commitment
   measurement* if `gap_human` is estimable and its CI lower bound is above 0.
   This is the replication's own rule.
2. **Magnitude:**
   - if the Δ CI includes 0: *no detectable change in magnitude* under human `C`;
   - if the Δ CI is entirely below 0: *attenuated*;
   - if the Δ CI is entirely above 0: *larger*.
3. **Consequence for reporting:**
   - **If it survives:** the Qwen replication is reported with `gap_human` as
     the human-grounded estimate next to the J-qy +0.32. The magnitude wording
     follows rule 2.
   - **If it does not survive:** the Qwen replication is reported as *not
     robust to human measurement of commitment*. The cross-lineage claim is
     restricted to "confirmed in OLMo-2-13B; in Qwen2.5-14B-Instruct, observed
     under J-qy measurement only". The R1-Distill arm, judged by a sibling of
     the same base and not human-audited, is qualified the same way.

## 7. Secondary analyses, pre-specified (descriptive unless stated)

1. **`gap_J-qy` on the sampled units vs the population +0.3158.** This is a check
   on sub-sampling noise. It was computed at freezing from J-qy outcomes only (no
   human labels): **+0.262 [0.086, 0.494]**, with 37 committed units. The paired
   Δ of §6 is therefore the cleanest comparison.
2. **Rule-based SF sensitivity:** the same estimator with the rule-based
   stance-first outcome, for `C_human` and for `C_J-qy`. At freezing, from J-qy
   only: +0.087 [−0.081, 0.260] on the sampled units, against +0.136
   [0.001, 0.297] in the population.
3. **Annotator sensitivity:** `gap` with `C` from A alone and from C alone. These
   are sensitivity analyses, not competing ground truths.
4. **Commitment by stratum:** for human `C` and J-qy `C`:
   - committed units out of the stratum's units, with counts;
   - clusters with at least one commitment.

   Within a stratum the sample is uniform, so these are unweighted.
5. **Per prompt × stratum:** commitment counts, human vs J-qy. Descriptive.
6. **All 88 items:** **not estimable** from this design, which samples class A
   only. It is not computed and is not substituted for the primary.

## 8. Interpretation guardrails

- This re-estimates a measurement-conditional quantity. It is not a political
  position of the model and not a population stance estimate.
- Unweighted sample rates are never reported as population rates.
- Right-pole cells are a census of 10 clusters (50 units). They are reported
  with counts, not percentages alone.
- The result qualifies or supports one checkpoint (Qwen2.5-14B-Instruct). It
  does not by itself re-estimate the OLMo gaps, which have their own
  representative audit.

## 9. Outputs

- Tables: `artifacts/tables/selection_gap_human_qwen_v1_*.csv` and `_report.md`,
  from `scripts/selection_gap_human_qwen_v1.py analyze`.
- Results: `docs/results/selective_commitment_human_reestimate_qwen_v1.md`,
  written once both new sheets are labelled and adjudicated.

## Freeze record

Freeze commit: `9776f7f74fffd89715bb6cf859231ba11dd4b63d` ("research: freeze Qwen human selection-gap
re-estimation", 2026-11-04). It was pushed to `origin/research/human-commitment-audit-closeout`
before any new label existed.

## Deviations

There is no deviation from the design, estimand, estimator or reading rule.

**Pre-labeling display update (anticipated in §4), 2026-11-04.** Before any new label
existed, 14 of the 33 French statement translations were replaced verbatim by the
translations shown in the earlier human-C French rerun. The source is
`prior_C_translation_reuse_reference.csv`, SHA-256
`71e646a48496c98a73e4ad3e3594cbd68f67fdcad6b82100435fc919223be63e`, and the exact English
statements match. Page C was rebuilt. The remaining 19 drafts were then reviewed and
corrected by the researcher (provenance `researcher_reviewed_2026-11-04`; 6 wordings changed;
English statements and the 14 reused translations byte-identical), and page C was rebuilt
again. All of this happened before page C was sent. The sample, the sheets and page A are
unchanged.

**Analysis order.** The Qwen audit was analysed (`114ac04`) after this protocol was
frozen. Its frozen reading rule judged the Qwen measurement *measurement-qualified*, which
confirms the trigger described in §0.
