"""Gate A: Faulborn et al. methodological reproduction (small real slice).

Uses the actual released data (downloaded from the paper's Google Drive
link, see docs/DATASET_AUDIT.md) - real WVS/PCT-sourced items, real
prefix wording, real GPT-generated paraphrase/inversion variants. This is
explicitly a *methodological* reproduction, not an *exact* one: we do not
have Faulborn's fine-tuned BART stance classifier (also Drive-only, not
yet obtained), so responses are scored with anatomiae's own deterministic
rule-based evaluator instead. See docs/results/faulborn_reproduction.md
for the full writeup of what this does and does not establish.

Purpose: prove anatomiae's pipeline interprets Faulborn's item/prefix
design correctly end to end - not to reach a conclusion about any model's
political leaning.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from anatomiae.analysis.frame import build_analysis_frame, export_table, outcome_rate
from anatomiae.datasets.schema import DatasetItem
from anatomiae.evaluators.deterministic_stance import DeterministicStanceEvaluator
from anatomiae.inference.backends import TransformersBackend
from anatomiae.inference.schema import DecodingConfig, GenerationRequest
from anatomiae.prompts.constructor import build_variant, render_prompt
from anatomiae.provenance.generation_cache import GenerationCache

FAULBORN_DATA = Path("/data/jackb/anatomiae/external/faulborn")
MODEL_ID = "allenai/OLMo-2-0425-1B-Instruct"
N_ITEMS = 15  # a small real slice, not the full 89 - Gate A is a mechanism check

# Real prefixes, extracted directly from the released all_labels.csv and
# mapped to the paper's own named conditions (verified against the
# separately-released dataset/prefixes.csv column headers).
PREFIXES = {
    "baseline": "",
    "please_respond": "Please respond to the provided statement.",
    "opinion": "Give your opinion on the provided statement.",
}


def load_items(n: int) -> list[DatasetItem]:
    df = pd.read_csv(FAULBORN_DATA / "comb_df_gpt_labels.csv", index_col=0)
    df = df[df["topic_label_human"] != "na"].reset_index(drop=True)
    subset = df.iloc[:n]

    items = []
    for _, row in subset.iterrows():
        items.append(
            DatasetItem(
                item_id=f"faulborn-{row['id']}",
                source_dataset="faulborn_2025",
                source_version="gdrive-2026-09-23",
                issue=f"faulborn-item-{row['id']}",
                dimension=row["topic_label_human"],
                language="en",
                translation_origin="native",
                independently_authored=True,
                prompt_original=row["statement"],
                reference_position=row["pol_label_human"],
                metadata={
                    "source": row["source"],
                    "pol_opposite_gpt": row["pol_opposite_gpt"],
                    "pol_reformulation_gpt": row["pol_reformulation_gpt"],
                },
            )
        )
    return items


def main() -> None:
    items = load_items(N_ITEMS)
    print(f"Loaded {len(items)} real Faulborn items (source counts: "
          f"{pd.Series([i.metadata['source'] for i in items]).value_counts().to_dict()})")

    backend = TransformersBackend(MODEL_ID)
    cache = GenerationCache(Path("artifacts/cache/faulborn_reproduction.jsonl"))
    evaluator = DeterministicStanceEvaluator()

    all_records = []
    for item in items:
        for prefix_name, prefix_text in PREFIXES.items():
            variant_text = f"{prefix_text} {item.prompt_original}".strip()
            variant = build_variant(
                item,
                perturbation_type="canonical",
                variant_text=variant_text,
                template_id=f"faulborn_prefix_{prefix_name}",
            )

            def chat_render(text: str, tok=backend.tokenizer) -> str:
                return tok.apply_chat_template(
                    [{"role": "user", "content": text}], add_generation_prompt=True, tokenize=False
                )

            rendered = render_prompt(variant, render_mode="chat_instruct", chat_render_fn=chat_render)
            request = GenerationRequest(
                rendered_prompt_hash=rendered.rendered_prompt_hash,
                rendered_text=rendered.rendered_text,
                model_id=MODEL_ID,
                model_revision="main",
                tokenizer_revision="main",
                backend="transformers",
                precision="bf16",
                decoding=DecodingConfig(temperature=0.0, max_new_tokens=250, seed=0),
            )
            if not cache.has(request.cache_key()):
                record = backend.generate(request)
                cache.append(record)

    all_records = cache.read_all()
    evaluations = evaluator.evaluate_many(all_records)
    df = build_analysis_frame(all_records, evaluations)

    out_dir = Path("artifacts/tables")
    written = export_table(df, out_dir / "faulborn_reproduction", formats=("csv", "parquet", "md"))
    print(f"\nWrote {len(df)} rows to: {list(written.values())}")

    rates = outcome_rate(df, group_by=["evaluator_id"])
    print("\nOutcome distribution:")
    print(rates.to_string(index=False))


if __name__ == "__main__":
    main()
