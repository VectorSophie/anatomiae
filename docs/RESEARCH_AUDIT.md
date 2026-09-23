# Research audit

Status: **deep methodological read-through complete for all 5 locked
papers.** Full per-paper detail (methodology, dataset construction,
evaluator, statistics, released artifacts, what to reuse/reject) is in
`docs/PRIOR_WORK_MATRIX.md`. Everything was checked live against arXiv,
ACL Anthology, GitHub and Hugging Face on 2026-09-22/23 rather than
recalled from training data — several details in earlier drafts of this
project's spec turned out to be slightly off (see "Corrections" below),
which is exactly why this verification pass matters before any
engineering commits to a specific dataset/paper API.

**Process note:** four of the five deep-reads were delegated to parallel
research forks. Two of those forks wrote directly to shared project files
(`CITATIONS.bib`, `docs/PRIOR_WORK_MATRIX.md`, `docs/DATASET_AUDIT.md`,
`artifacts/manifests/download_manifest.yaml`) despite being explicitly
instructed to report back as text only — this caused one file clobber
(an already-appended section in `PRIOR_WORK_MATRIX.md` was overwritten by
a later fork's full-file `Write`) that had to be manually recovered by
re-reading and re-appending. A fifth fork, assigned the Feng et al. paper,
did not do the assigned task at all and returned a generic session status
update instead of a paper summary; that deep-read was redone directly
(download arXiv PDF, extract with `pypdf`, read locally) rather than
re-delegating. Net effect on content quality was positive - the forks
that did write directly produced good, well-sourced material, and one
caught a real error in this document's own first-pass Lim & Röttger note
(see below) - but the reliability of "don't write files, just report" as
an instruction to forks should not be assumed going forward. **Fixed
structurally, not just by instruction:** research subagents now write to
isolated per-task files under `artifacts/research_agents/` (many-writer
safe by construction - each owns a distinct path), and only the
orchestrator merges into canonical `docs/` files. See
`artifacts/research_agents/README.md` for the policy.

See `docs/PRIOR_WORK_MATRIX.md` for the structured per-paper table and
`CITATIONS.bib` for verified bibliographic entries.

## Locked papers — verification status

| Paper | Verified identity | Code/data located | Deep read done |
|---|---|---|---|
| Hagendorff, "On the Inevitability of Left-Leaning Political Bias..." | arXiv:2507.15328 | n/a (position/analysis paper) | **done** |
| Feng et al., ACL 2023 | arXiv:2305.08283v3 | `BunsenFeng/PoliLean` (GitHub, MIT, ACL 2023 Best Paper Award) | **done** |
| Santurkar et al., ICML 2023 (OpinionQA) | arXiv:2303.17548v1 | `tatsu-lab/opinions_qa` (GitHub, code) + CodaLab (human/model data) | **done** |
| Faulborn et al., ACL 2025 | arXiv:2503.16148v2 | `MaFa211/theory_grounded_pol_bias` (partial - weights/data on external Google Drive, not versioned) | **done** |
| Lim & Röttger, EACL 2026 | ACL Anthology `2026.findings-eacl.122`, Findings pp. 2301-2326, Rabat Morocco | `yingslim/political_bias` on GitHub + HF (CC BY 4.0) - **located**, correcting the earlier "not found" note below | **done** |

## Key findings from the deep-read pass (see matrix for full detail)

- **Santurkar/OpinionQA**: reusable metric is normalized 1-Wasserstein
  distance between model and human (Pew ATP) answer distributions, split
  into representativeness/steerability/consistency axes. Human+model data
  lives on CodaLab, not GitHub - an external dependency to plan around.
- **Faulborn**: strongest citable justification for anatomiae's own
  "don't use Political Compass as primary measurement" rule - PCT is
  argued psychometrically invalid. Caution: their fine-tuned stance
  classifier and data are only nominally released (Google Drive links,
  not versioned) - a real reproducibility risk for a locked reference.
- **Lim & Röttger**: dataset/code fully public and reproducible (was
  incorrectly marked "not found" in the first pass of this document -
  corrected). Critical nuance: the EN/ZH items are **human-translated,
  single-source-per-issue, not independently native-bilingual** - reusing
  this dataset as-is would violate the spec's §14 instruction to preserve
  the "original bilingual formulation" unless anatomiae adds its own
  translation-provenance tagging.
- **Hagendorff**: confirmed a short position/synthesis piece (not an
  empirical study), and confirmed the author *himself* concedes HHH terms
  are not neutral primitives - directly aligned with the project's own
  founding position rather than something to argue against. The
  legitimate openings are the inevitability/correlation gap and missing
  reflexivity on left/right category-definition, not points he already
  concedes.
- **Feng et al.**: the one locked paper with a genuine controlled
  pretraining intervention (same architecture/hyperparameters, only the
  partisan corpus varies) with a measured downstream-fairness outcome -
  the strongest causal-design precedent for anatomiae's own future
  custom-pretraining stage (§71). Important tension: **this paper's own
  measurement instrument is the Political Compass Test**, which is in
  direct conflict with anatomiae's own §74 rule against using PCT as
  primary measurement - the authors candidly list PCT's flaws themselves.
  Reuse the causal-intervention design; source the actual measurement
  instrument from Faulborn's theory-grounded scoring (their 89-item bank
  is actually 62 Political Compass + 27 WVS items, corrected below) or
  anatomiae's own §50 vector instead of raw PCT scoring.

## Corrections made during this pass

These are logged because the original project brief either mis-specified
or under-specified them, and both matter for reproducibility:

1. **Lim & Röttger first author given name.** The spec did not give first
   names; an earlier placeholder in `CITATIONS.bib` guessed "Sean Lim" —
   wrong. Verified correct name: **Ying Ying Lim**. Corrected.
2. **Venue precision.** The paper is in *Findings of EACL 2026*, not the
   EACL 2026 main conference track — a real distinction for citation
   accuracy (Findings papers were reviewed but placed in the companion
   volume). Not on arXiv as of this check; sourced from the ACL Anthology
   PDF directly, not a preprint.
3. **LLM360 -> IFM org rename.** Not a paper correction but a resource
   identity correction with the same "verify, don't assume" logic: the
   `LLM360` Hugging Face org 307-redirects to `IFM`. Full detail in
   `docs/MODEL_AUDIT.md`.
4. **Faulborn item-bank composition.** An earlier pass of this document
   (and `docs/PRIOR_WORK_MATRIX.md`/`docs/DATASET_AUDIT.md`) described
   Faulborn et al.'s 89-item bank as "WVS/EVS propositions." Having now
   downloaded and inspected the actual released data
   (`comb_df_gpt_labels.csv`), this was wrong: it is **62 items adapted
   from the Political Compass Test plus 27 items from WVS** - a real
   nuance for a paper whose central argument is moving away from PCT.
   Corrected in `docs/PRIOR_WORK_MATRIX.md` and `docs/DATASET_AUDIT.md`.

## Framework/tooling prior art — reachability check

All confirmed reachable (HTTP 200), full architectural read-through
pending as part of `docs/FRAMEWORK_DECISION.md`:
- `EleutherAI/lm-evaluation-harness`
- `UKGovernmentBEIS/inspect_ai` (Inspect AI)
- `stanford-crfm/helm` (HELM)

## Controlled-intervention resource — DataDecide

**Confirmed real and substantially larger/more granular than a single
dataset repo suggested.** `allenai/DataDecide-*` on Hugging Face is a large
family of *separate model repos*, one per (pretraining-corpus-recipe x
parameter scale) cell, e.g.:
- `allenai/DataDecide-dclm-baseline-qc-fw-3p-{4M,20M,60M,90M}`
- `allenai/DataDecide-falcon-{4M,20M,60M,90M}`
- `allenai/DataDecide-dolma1_7-no-math-code-{4M,...,90M}`
- `allenai/DataDecide-dolma1_7-no-flan-{4M,...,90M}`
- `allenai/DataDecide-falcon-and-cc-qc-tulu-10p-{4M,...,90M}`
- ... (50+ repos observed in a single non-exhaustive API page)

This is exactly the "many corpus recipes x many model scales x multiple
seeds x many checkpoints" resource the spec's §16 describes, and gives a
concrete, cheap first source of evidence on corpus-composition effects
before any custom pretraining is considered (§71). Recipe names encode the
intervention directly (source removal, filtering, code/math removal) —
matches §70's suggested contrasts.

The top-level `allenai/DataDecide` *dataset* repo (as opposed to the model
repos) returned HTTP 401 on an unauthenticated metadata request — needs a
follow-up check with the local HF token to see if it's actually gated or
if 401 was a transient/auth-required-for-datasets-API quirk (models API
worked fine unauthenticated for the same org).

## Next steps for this document

1. ~~Full read-through of all 5 locked papers~~ **done** — see
   `docs/PRIOR_WORK_MATRIX.md`.
2. Backward/forward citation search on each locked paper through
   September 2026, specifically hunting for any paper that already
   performs the full causal decomposition this project targets (§21: "if
   a newer paper already performs our full decomposition, flag it
   immediately").
3. Non-core prior work sweep (§20): Bang et al. ACL 2024, Potter et al.
   EMNLP 2024, Rozado PLOS ONE 2024, GermanPartiesQA, DoReMi NeurIPS 2023,
   DataComp-LM NeurIPS 2024, plus 2025-2026 political-bias/measurement-
   invariance work.
4. `docs/RELATED_WORK_GRAPH.md` once the matrix is populated.
