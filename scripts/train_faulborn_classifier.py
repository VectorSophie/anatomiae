"""Reconstruct Faulborn et al.'s fine-tuned BART stance classifier.

Their released checkpoint (step 1750) is missing its weights file - see
docs/results/faulborn_classifier_reproduction.md. This retrains following
their released `stance_detector/finetune-bart.py` so the classifier path
can be validated against their own released test split and reported
metrics. The output is an anatomiae *reconstruction*, never "their
weights".

Two data-flow variants, identical in every other respect (and, by
construction, in training-set size: 2,112 NLI pairs each):

  as_written    - replicates the released script's data flow: NLI pairs are
                  built from ALL 1,320 labeled responses (the script maps the
                  full shuffled set, not its own saved train split) and then
                  re-split 80/20 at the pair level, so responses from the
                  released 264-row test split contribute training pairs.
  leakage_free  - NLI pairs built only from the released 1,056-row train
                  split; the released test split never contributes training
                  pairs.

Kept identical to finetune-bart.py: base facebook/bart-large-mnli; premise =
response text only; hypothesis "The stance of the statement is {label}";
one entailment pair (label 2) with the gold stance plus one contradiction
pair (label 0) with a random other stance per response; lr 2e-5,
max_steps 1750, per-device batch 4, warmup 500, weight_decay 0.2, fp32.

Deliberate, documented deviations: dynamic padding instead of
padding="max_length" (padding is attention-masked, so this changes compute,
not outputs); the contradiction label is drawn from a seeded RNG (the
released script uses unseeded `random.choice`, so its pairs are not
reproducible); no intermediate pair-level evaluation or wandb logging.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from anatomiae.provenance.gpu_power import full_gpu_preflight

FAULBORN_ROOT = Path("/data/jackb/anatomiae/external/faulborn")
SPLITS = FAULBORN_ROOT / "stance_detector_eval"
OUT_ROOT = Path("/data/jackb/anatomiae/models/faulborn_classifier_reconstruction")
BASE_MODEL = "facebook/bart-large-mnli"
TEMPLATE = "The stance of the statement is {}"
LABELS = ["disagree", "agree", "neutral", "unrelated"]


def sha256_dir(path: Path) -> str:
    h = hashlib.sha256()
    for f in sorted(path.rglob("*")):
        if f.is_file():
            h.update(f.name.encode())
            h.update(f.read_bytes())
    return h.hexdigest()


def nli_pairs(rows: list[tuple[str, str]], rng: random.Random) -> list[dict]:
    pairs = []
    for text, label in rows:
        contradiction = rng.choice([x for x in LABELS if x != label])
        pairs.append({"premise": text, "hypothesis": TEMPLATE.format(label), "labels": 2})
        pairs.append({"premise": text, "hypothesis": TEMPLATE.format(contradiction), "labels": 0})
    return pairs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["as_written", "leakage_free"], required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    preflight = full_gpu_preflight()

    from datasets import Dataset, load_from_disk
    from transformers import (
        AutoTokenizer,
        BartForSequenceClassification,
        DataCollatorWithPadding,
        Trainer,
        TrainingArguments,
    )

    train_rows = load_from_disk(str(SPLITS / "train" / "train_set")).to_pandas()
    test_rows = load_from_disk(str(SPLITS / "test" / "test_set")).to_pandas()
    rng = random.Random(args.seed)

    if args.variant == "as_written":
        pool = list(zip(train_rows.text, train_rows.label)) + list(
            zip(test_rows.text, test_rows.label)
        )
        pairs = nli_pairs(pool, rng)
        rng.shuffle(pairs)
        pairs = pairs[: int(0.8 * len(pairs))]
    else:
        pairs = nli_pairs(list(zip(train_rows.text, train_rows.label)), rng)

    test_texts = set(test_rows.text)
    n_pairs_from_test = sum(p["premise"] in test_texts for p in pairs)
    print(f"variant={args.variant} seed={args.seed} train_pairs={len(pairs)} "
          f"pairs_whose_premise_is_a_test_response={n_pairs_from_test}")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    ds = Dataset.from_list(pairs).map(
        lambda b: tokenizer(b["premise"], b["hypothesis"], truncation=True),
        batched=True,
        remove_columns=["premise", "hypothesis"],
    )
    model = BartForSequenceClassification.from_pretrained(BASE_MODEL)

    out_dir = OUT_ROOT / f"{args.variant}_seed{args.seed}"
    training_args = TrainingArguments(
        output_dir=str(out_dir / "trainer_tmp"),
        max_steps=1750,
        per_device_train_batch_size=4,
        learning_rate=2e-5,
        warmup_steps=500,
        weight_decay=0.2,
        logging_steps=50,
        save_strategy="no",
        report_to="none",
        seed=args.seed,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds,
        data_collator=DataCollatorWithPadding(tokenizer),
    )
    t0 = time.time()
    train_result = trainer.train()
    elapsed = time.time() - t0

    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))

    git_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    ).stdout.strip()
    provenance = {
        "note": "anatomiae reconstruction of Faulborn et al.'s stance classifier - NOT the authors' weights",
        "variant": args.variant,
        "seed": args.seed,
        "base_model": BASE_MODEL,
        "train_pairs": len(pairs),
        "train_pairs_whose_premise_is_a_released_test_response": n_pairs_from_test,
        "released_train_split_sha256": sha256_dir(SPLITS / "train" / "train_set"),
        "released_test_split_sha256": sha256_dir(SPLITS / "test" / "test_set"),
        "hyperparameters": {
            "max_steps": 1750, "per_device_train_batch_size": 4, "learning_rate": 2e-5,
            "warmup_steps": 500, "weight_decay": 0.2, "precision": "fp32",
        },
        "final_train_loss": train_result.training_loss,
        "train_seconds": elapsed,
        "gpu": {
            "physical_index": preflight.gpu.physical_index,
            "power_limit_w": preflight.power.applied_power_limit_w,
        },
        "anatomiae_git_commit": git_commit,
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    (out_dir / "anatomiae_provenance.json").write_text(json.dumps(provenance, indent=2))
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
