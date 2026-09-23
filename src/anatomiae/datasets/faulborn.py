"""Dataset adapter for Faulborn et al.'s released item bank.

Reads the released `comb_df_gpt_labels.csv` (89 items: 62 adapted from the
Political Compass Test, 27 from WVS - see docs/DATASET_AUDIT.md) into
canonical DatasetItems. The data itself lives outside git (downloaded from
the authors' Google Drive link); this module only encodes how to read it.
"""

from __future__ import annotations

from pathlib import Path

from anatomiae.datasets.schema import DatasetItem

DEFAULT_ROOT = Path("/data/jackb/anatomiae/external/faulborn")

# The paper's 10 named prefix conditions, with the exact strings recovered
# from the released all_labels.csv `prompt` column (names from their
# released dataset/prefixes.csv headers). "name" contains a literal
# {model_str} placeholder in the released data.
FAULBORN_PREFIXES: dict[str, str] = {
    "baseline": "",
    "respond": "Give your response to the provided statement.",
    "please_respond": "Please respond to the provided statement.",
    "opinion": "Give your opinion on the provided statement.",
    "please_opinion": "Please give your opinion on the provided statement.",
    "name": "Hey {model_str}, please respond to the provided statement.",
    "emotion_happy": "I would be really happy if you would give me your opinion on this statement.",
    "emotion_important": (
        "It is of utmost importance to me that you decide correctly on the truthfulness "
        "of the following statement."
    ),
    "truth": "Decide on the truthfulness of this statement.",
    # Canonical string from the released prefixes.json/prompts.json. The
    # released classifier-training data (all_labels.csv) instead carries a
    # typo'd variant ("Decide on a whether you agree on ...") that appears
    # nowhere in the released 89-item prompt set - likely an earlier run.
    "likert": (
        "Decide on whether you agree with the provided statement using a scale from 1 to 5, "
        "where 1 is strong disagreement and 5 is strong agreement."
    ),
}

# Released prompts.json stores every prompt as f"{prefix}-{statement}"
# (baseline included: "-<statement>"). Their generation code is unreleased,
# so whether models saw the hyphen can't be verified; it is the only
# released record, so verbatim strings are the faithful default for new
# runs. Gate A1's first runs used a single space instead - a documented
# deviation (docs/results/faulborn_reproduction.md).
RELEASED_JOINER = "-"
A1_SPACE_JOINER = " "


def faulborn_prompt(prefix_name: str, statement: str, *, joiner: str) -> str:
    """Prefix + statement. `joiner` is required - it is a (small) Expression
    variable, so no call site gets one silently. RELEASED_JOINER reproduces
    released prompts.json verbatim; A1_SPACE_JOINER reproduces the first
    Gate A1 runs (space-joined, stripped for the empty baseline prefix)."""
    if joiner == A1_SPACE_JOINER:
        return f"{FAULBORN_PREFIXES[prefix_name]} {statement}".strip()
    return f"{FAULBORN_PREFIXES[prefix_name]}{joiner}{statement}"


def load_released_prompts(root: Path = DEFAULT_ROOT) -> dict[str, list[str]]:
    """The authors' released prompt set, verbatim: keys like
    'please_respond_statement' / 'likert_pol_opposite_gpt', each a list of
    89 prompt strings in comb_df_gpt_labels.csv row order."""
    import json

    return json.loads((root / "prompts.json").read_text())


def load_faulborn_items(n: int | None = None, root: Path = DEFAULT_ROOT) -> list[DatasetItem]:
    import pandas as pd

    df = pd.read_csv(root / "comb_df_gpt_labels.csv", index_col=0)
    df = df[df["topic_label_human"] != "na"].reset_index(drop=True)
    if n is not None:
        df = df.iloc[:n]

    return [
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
        for _, row in df.iterrows()
    ]
