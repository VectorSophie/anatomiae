# Datasets

Full audit detail (access status, license, reproducibility risk) lives in
`docs/DATASET_AUDIT.md`; this page gives the scientific role each dataset
plays, condensed for a first-time reader.

| Dataset | Scientific role | Language(s) | License | Redistribution notes |
|---|---|---|---|---|
| Faulborn et al. (WVS/EVS-grounded) | Measurement methodology — theory-grounded political construct, reproduced as a correctness check (`docs/FAULBORN_REPRODUCTION.md`) | English | CC-BY-4.0 (code) | Full ~88K-response dataset only on external Google Drive, not versioned — a real reproducibility risk, documented not hidden. |
| Lim & Röttger | EN/ZH prompt-language and model-origin robustness | English + Chinese | CC BY 4.0 | **Items are human-translated, single-source-per-issue, not independently native-bilingual.** Reused with explicit translation-provenance tagging (`translation_origin`, `independently_authored` on every item) — never presented as parallel-native data. |
| OpinionQA | Human-population grounding — model output vs. real Pew survey response distributions | English | Code: unspecified; underlying Pew microdata has its own terms | Data hosted on CodaLab, not GitHub; license unverified for the derived data — do not redistribute without checking Pew's original terms. |
| Dolma / OLMo training data | Transparent pretraining-corpus attribution for the OLMo lineage arm | English (primarily) | Apache-2.0-family (per AI2) | Needed only for the pretraining-attribution arm, not the core pilot. |
| Amber training-data sequence (`IFM/AmberDatasets`) | Checkpoint-level longitudinal pretraining attribution | English (primarily) | Apache-2.0 | Confirmed to exist; size/format not yet fully characterized. |
| DataDecide (`allenai/DataDecide-*`) | Controlled pretraining-data-mixture intervention (source removal, filtering, code/math removal) across scale x seed | English | Apache-2.0-family (per AI2) | A model grid (corpus recipe x scale), not a single dataset — ~50+ repos observed. |
| WVS / CHES / Manifesto Project | External human/political reference systems, used as separate calibration points, never blended into one "ground truth ideology" table | Multiple | Per-source, not yet fully audited | Licensing must be checked per source before any redistribution of derived artifacts. |

## Why these, together

No single dataset here does everything: Faulborn gives measurement rigor
without causal leverage over model formation; OpinionQA gives human
grounding without locating the pipeline stage responsible; Lim & Röttger
gives language/origin contrast but with translation-dependent parallel
items; Dolma/Amber/DataDecide give training-time transparency that
political-bias studies typically lack entirely. `anatomiae` combines
these complementary strengths rather than treating any one as sufficient
on its own — see `docs/PRIOR_WORK_MATRIX.md` for the full per-paper
tension/complementarity analysis.

## Bilingual/multilingual provenance

Every dataset item carries explicit translation provenance
(`source_language`, `translation_origin`, `translator_type`,
`independently_authored`, `parallel_item_id` — see
`src/anatomiae/datasets/schema.py`). This is not optional metadata: it is
the only way to distinguish "two languages independently produced
different responses" from "one response was translated and compared to
itself," and collapsing that distinction would misattribute translation
artifacts to genuine cross-lingual variation.
