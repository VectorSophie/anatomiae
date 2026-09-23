"""TransformersBackend.generate must pass an explicit attention mask and pad
id (pad == eos tokenizers such as Qwen2.5 otherwise make transformers guess).
Runs on CPU with fakes; never touches a GPU or loads a model."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from anatomiae.inference.backends import TransformersBackend
from anatomiae.inference.schema import DecodingConfig, GenerationRequest


class _Encoded:
    def __init__(self, n):
        self.input_ids = torch.ones((1, n), dtype=torch.long)
        self.attention_mask = torch.ones((1, n), dtype=torch.long)




class _FakeTokenizer:
    pad_token_id = None
    eos_token_id = 7

    def __call__(self, text, return_tensors):
        return _Encoded(len(text.split()))

    def decode(self, ids, skip_special_tokens):
        return "x " * len(ids)


class _FakeModel:
    def __init__(self):
        self.kwargs = None

    def generate(self, input_ids, **kwargs):
        self.kwargs = kwargs
        return torch.cat([input_ids, torch.full((1, 3), 5)], dim=1)


def test_attention_mask_and_pad_id_are_explicit(monkeypatch):
    monkeypatch.setattr(torch.Tensor, "to", lambda self, *a, **k: self)
    backend = object.__new__(TransformersBackend)
    backend._model_id, backend._revision, backend._gpu_provenance = "m/x", "main", None
    backend.tokenizer = _FakeTokenizer()
    model = _FakeModel()
    backend._model_for_precision = lambda precision: model
    request = GenerationRequest(rendered_prompt_hash="h", rendered_text="a b c", model_id="m/x",
                                model_revision="main", tokenizer_revision="main", backend="transformers",
                                precision="bf16", decoding=DecodingConfig(temperature=0.0, max_new_tokens=3))
    record = backend.generate(request)
    assert record.error is None
    assert torch.equal(model.kwargs["attention_mask"], torch.ones((1, 3), dtype=torch.long))
    assert model.kwargs["pad_token_id"] == 7
