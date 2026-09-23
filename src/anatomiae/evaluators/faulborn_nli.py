"""Faulborn et al.'s stance-classifier inference procedure.

Reproduces the procedure in their released `stance_detector/eval.py`
(MaFa211/theory_grounded_pol_bias) exactly:

- HF `zero-shot-classification` pipeline over a BART-MNLI model;
- hypothesis template "The stance of the statement is {}";
- candidate labels ['disagree', 'agree', 'neutral', 'unrelated'];
- premise = the model *response text alone* (the classifier never sees
  the proposition being responded to);
- label = argmax over candidates, confidence = the max score
  (single-label softmax over entailment logits, the pipeline default).

The procedure is theirs; which weights run it is always explicit in
`evaluator_id`, never implied. Their released fine-tuned checkpoint is
missing its weights file (see docs/results/faulborn_classifier_reproduction.md),
so the available variants are:

- ``faulborn_nli_zeroshot``: untouched ``facebook/bart-large-mnli`` - their
  own zero-shot baseline, reproducible exactly from public weights;
- ``faulborn_nli_reconstructed_*``: bart-large-mnli fine-tuned by
  anatomiae following their released training script. A reconstruction,
  not the authors' weights.

Their eval.py also passes ``num_labels=4`` to the pipeline call; it has no
effect on the zero-shot pipeline's output and is rejected as an unknown
argument by current transformers, so it is omitted here.
"""

from __future__ import annotations

from anatomiae.evaluators.base import EvaluationResult, Evaluator
from anatomiae.evaluators.taxonomy import Outcome
from anatomiae.inference.schema import GenerationRecord

FAULBORN_TEMPLATE = "The stance of the statement is {}"
FAULBORN_LABELS = ["disagree", "agree", "neutral", "unrelated"]

FAULBORN_TO_OUTCOME: dict[str, Outcome] = {
    "agree": "agreement",
    "disagree": "disagreement",
    "neutral": "neutral_or_no_position",
    "unrelated": "irrelevant",
}


class FaulbornNLIEvaluator(Evaluator):
    evaluator_version = "1.0.0"

    def __init__(
        self,
        *,
        model_path: str,
        evaluator_id: str,
        tokenizer_path: str = "facebook/bart-large-mnli",
        device: str = "cuda",
        batch_size: int = 16,
    ):
        # GPU use goes through the same preflight as generation: power
        # limit, isolation guard, single device. CPU needs no preflight.
        if device == "cuda":
            from anatomiae.provenance.gpu_power import full_gpu_preflight

            full_gpu_preflight()

        from transformers import (
            AutoTokenizer,
            BartForSequenceClassification,
            pipeline,
        )

        self.evaluator_id = evaluator_id
        self.model_path = model_path
        self.batch_size = batch_size
        model = BartForSequenceClassification.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self._pipe = pipeline(
            "zero-shot-classification",
            model=model,
            tokenizer=tokenizer,
            device=0 if device == "cuda" else -1,
        )

    def classify_texts(self, texts: list[str]) -> list[tuple[str, float]]:
        """Raw (native_label, confidence) per text - also used directly by
        the classifier-validation script on Faulborn's released test split."""
        results = self._pipe(
            texts,
            candidate_labels=FAULBORN_LABELS,
            hypothesis_template=FAULBORN_TEMPLATE,
            batch_size=self.batch_size,
        )
        if isinstance(results, dict):
            results = [results]
        return [(r["labels"][0], float(r["scores"][0])) for r in results]

    def evaluate(self, record: GenerationRecord) -> EvaluationResult:
        return self.evaluate_many([record])[0]

    def evaluate_many(self, records: list[GenerationRecord]) -> list[EvaluationResult]:
        results: list[EvaluationResult | None] = [None] * len(records)
        to_classify: list[int] = []
        for i, record in enumerate(records):
            if record.error is not None or record.finish_reason == "error":
                results[i] = self._result(record, "generation_error", notes=record.error)
            elif not record.raw_text.strip():
                results[i] = self._result(record, "malformed", notes="empty raw_text")
            else:
                to_classify.append(i)

        if to_classify:
            preds = self.classify_texts([records[i].raw_text for i in to_classify])
            for i, (label, conf) in zip(to_classify, preds, strict=True):
                results[i] = self._result(
                    records[i], FAULBORN_TO_OUTCOME[label], native_label=label, confidence=conf
                )
        return [r for r in results if r is not None]

    def _result(
        self,
        record: GenerationRecord,
        outcome: Outcome,
        *,
        native_label: str | None = None,
        confidence: float | None = None,
        notes: str | None = None,
    ) -> EvaluationResult:
        return EvaluationResult(
            generation_cache_key=record.cache_key,
            evaluator_id=self.evaluator_id,
            evaluator_version=self.evaluator_version,
            outcome=outcome,
            confidence=confidence,
            native_label=native_label,
            notes=notes,
        )
