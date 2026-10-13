"""Answer extraction for reasoning models in the cross-lineage amendment."""

import sys
from pathlib import Path

import pytest

pytest.importorskip("pandas")
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

import selective_commitment_xl_v1 as xl

from anatomiae.inference.schema import DecodingConfig, GenerationRecord, GenerationRequest


def record(model_id: str, text: str) -> GenerationRecord:
    req = GenerationRequest(rendered_prompt_hash="h", rendered_text="p", model_id=model_id, model_revision="r",
                            tokenizer_revision="r", backend="vllm", precision="bf16", decoding=DecodingConfig())
    return GenerationRecord.build(request=req, raw_text=text, finish_reason="stop", input_tokens=1,
                                  output_tokens=1, latency_seconds=0.0)


R1 = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"


@pytest.mark.parametrize(("model", "text", "expected"), [
    (R1, "reasoning...\n</think>\n\nI agree.", "I agree."),
    (R1, "a</think>b</think>  final ", "final"),
    (R1, "still thinking, budget ran out", ""),
    (R1, "done</think>   ", ""),
    ("Qwen/Qwen2.5-14B-Instruct", "plain </think> text", "plain </think> text"),
])
def test_answer_text(model, text, expected):
    assert xl.answer_text(record(model, text)) == expected


def test_arms_pinned():
    assert xl.ARMS["q_inst"][2] == "cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8"
    assert xl.ARMS["q_r1"][2] == "1df8507178afcc1bef68cd8c393f61a886323761"
    assert xl.ARMS["q_inst"][3] == 600 and xl.ARMS["q_r1"][3] == 4096
