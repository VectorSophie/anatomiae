# Dataset audit

Status: initial identity/access pass. Full schema/license/contamination
detail to be filled in as each dataset is actually pulled and inspected
(not just confirmed to exist).

| Dataset | Canonical source | Access status | Locked? | Notes |
|---|---|---|---|---|
| Lim & Röttger EN/ZH bilingual political-issue prompts | Code: `github.com/yingslim/political_bias` (6-stage pipeline: scrape -> cluster issues -> generate prompts -> query models -> LLM-judge eval -> analysis). Data: `huggingface.co/datasets/yingslim/political_bias`, **CC BY 4.0**, 36k prompts / 144k precomputed responses across 4 models | verified_located | Locked (§14) | Deep-read complete (fork, 2026-09-23; corrected after a closer pass). **CORRECTION to an earlier note in this file:** items are **single-source-per-issue and human-translated into the second language**, not independently native-authored in both languages ("we manually translated China-related issues from U.S. media into Chinese and U.S.-related issues from Chinese media into English"). This does **not** satisfy the project spec's §14 preference to "preserve its original bilingual formulation where possible" at face value — if reused, anatomiae must add its own translation-provenance tagging (source language per issue) rather than presenting it as parallel-native data. Their own scoring uses a single LLM-judge (Gemini 2.5 Flash, benchmarked against a 500-item human gold set); per anatomiae's §57 (no single LLM judge defines political truth), treat `model_generated_responses`/judge scores as one input evaluator among several, and re-score the released `prompts` with anatomiae's own multi-evaluator pipeline. |
| Faulborn et al. theory-grounded political bias | `MaFa211/theory_grounded_pol_bias` (GitHub, CC-BY-4.0). **89 items (62 Political Compass Test items + 27 WVS items, corrected 2026-09-23 after downloading and inspecting the actual data - not purely "WVS/EVS" as an earlier pass claimed)** + 30 prompt variants each (10 prefixes x 3 paraphrase/inversion forms); ~88K responses + 1,320 hand-labeled examples now downloaded to `/data/jackb/anatomiae/external/faulborn/` (outside git, per the large-external-data convention) | verified_substantive, data_downloaded | Locked (§13) | Deep-read complete (fork, 2026-09-23); external data actually downloaded and inspected (not just located) 2026-09-23. Repo is real and functional: `analysis/` (paper-figure notebooks), `stance_detector/` (BART-MNLI fine-tune code), `dataset/` (prep scripts). `comb_df_gpt_labels.csv` has all 89 items with human political/topic labels plus GPT-generated paraphrase and position-inverted variants; `all_labels.csv` has the 1,320 hand-labeled (response, stance) pairs used to train their classifier. |
| OpinionQA | Code: `tatsu-lab/opinions_qa` (GitHub, analysis notebooks only). **Actual data: CodaLab worksheet `0x6fb693719477478aac73fc07db333f69`** (`model_input`=1,498 Pew ATP questions, `human_resp`=individual Pew responses across 60 demographic groups, `runs`=precomputed 2023-era OpenAI/AI21 outputs via HELM) | verified_located, license_unverified | Locked (§12) | Deep-read complete (fork, 2026-09-23). No LICENSE file in the GitHub repo; underlying Pew microdata likely carries its own redistribution terms — do not assume it's freely reusable, check before redistributing any derived artifact. Generation reproduction is coupled to 2023-era HELM run-specs querying OpenAI/AI21 endpoints that may no longer be queryable as-was; the analysis layer (representativeness/steerability/consistency notebooks) is highly reproducible against the precomputed `runs`. |
| DataDecide | `allenai/DataDecide-*` (Hugging Face, ~50+ model repos observed) | verified_reachable | Locked (§16) | See docs/RESEARCH_AUDIT.md — this is a *model* grid (corpus recipe x scale), not a single dataset; top-level `allenai/DataDecide` dataset repo returned HTTP 401 unauthenticated, needs re-check with local HF token. |
| Dolma / OLMo training data | Part of `allenai/*` HF org | not yet checked | Locked (§17) | Needed only for the OLMo attribution arm; not required for Tier 1-2 pilot generation. |
| Amber training data sequence | `IFM/AmberDatasets` (Hugging Face) | verified_reachable | Locked (§18) | Confirmed to exist; size/format not yet measured. |
| OpinionQA / Pew, WVS, CHES, Manifesto Project | various | not yet checked | Core human-reference (§19) | Licensing must be checked per-source before any redistribution of derived artifacts; keep separate, do not blend into one "ground truth" table. |

## Next steps

1. Clone/inspect Faulborn and OpinionQA repos for real (file listing,
   README, license file, data format) rather than existence-only checks.
2. Locate the actual Lim & Röttger data release — check the ACL Anthology
   page's supplementary materials link and the paper PDF's data
   availability statement.
3. Re-check `allenai/DataDecide` dataset-repo 401 with the local HF token
   to determine gated vs. transient-auth-quirk.
4. Contamination pass (§65) once pilot prompt sets are finalized: check
   publication dates against each candidate model's training cutoff.
