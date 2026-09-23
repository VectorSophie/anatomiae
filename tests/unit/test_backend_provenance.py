"""Backend/request provenance invariant tests.

TransformersBackend/VLLMBackend.__init__ needs real GPU hardware and
model weights (full_gpu_preflight, actual model/tokenizer loads), so these
tests construct bare instances via object.__new__() and set only the
attributes _validate_request() actually reads - deliberately never
calling __init__, never touching a GPU, never loading a model. This is a
pure-logic test of "does the validator correctly compare request fields
against backend identity," not an integration test of the backend itself.
"""

from __future__ import annotations

import pytest

from anatomiae.inference.backends import (
    RequestProvenanceMismatchError,
    TransformersBackend,
    VLLMBackend,
)
from anatomiae.inference.schema import DecodingConfig, GenerationRequest


def _bare_transformers_backend(model_id="model/x", revision="main"):
    backend = object.__new__(TransformersBackend)
    backend._model_id = model_id
    backend._revision = revision
    return backend


def _bare_vllm_backend(model_id="model/x", revision="main", precision="bf16"):
    backend = object.__new__(VLLMBackend)
    backend._model_id = model_id
    backend._revision = revision
    backend._precision = precision
    return backend


def _request(**overrides):
    defaults = {
        "rendered_prompt_hash": "hash",
        "rendered_text": "prompt",
        "model_id": "model/x",
        "model_revision": "main",
        "tokenizer_revision": "main",
        "backend": "transformers",
        "precision": "bf16",
        "decoding": DecodingConfig(),
    }
    defaults.update(overrides)
    return GenerationRequest(**defaults)


class TestTransformersValidation:
    def test_matching_request_passes(self):
        backend = _bare_transformers_backend()
        backend._validate_request(_request())  # no raise

    def test_wrong_backend_field_rejected(self):
        backend = _bare_transformers_backend()
        with pytest.raises(RequestProvenanceMismatchError, match="backend"):
            backend._validate_request(_request(backend="vllm"))

    def test_wrong_model_id_rejected(self):
        backend = _bare_transformers_backend(model_id="model/x")
        with pytest.raises(RequestProvenanceMismatchError, match="model_id"):
            backend._validate_request(_request(model_id="model/y"))

    def test_wrong_model_revision_rejected(self):
        backend = _bare_transformers_backend(revision="main")
        with pytest.raises(RequestProvenanceMismatchError, match="model_revision"):
            backend._validate_request(_request(model_revision="ckpt_180"))

    def test_wrong_tokenizer_revision_rejected(self):
        backend = _bare_transformers_backend(revision="main")
        with pytest.raises(RequestProvenanceMismatchError, match="tokenizer_revision"):
            backend._validate_request(_request(tokenizer_revision="other"))

    def test_multiple_mismatches_all_reported(self):
        backend = _bare_transformers_backend(model_id="model/x", revision="main")
        with pytest.raises(RequestProvenanceMismatchError) as exc_info:
            backend._validate_request(_request(model_id="model/y", model_revision="other"))
        msg = str(exc_info.value)
        assert "model_id" in msg
        assert "model_revision" in msg


class TestVLLMValidation:
    def test_matching_request_passes(self):
        backend = _bare_vllm_backend(precision="bf16")
        backend._validate_request(_request(backend="vllm", precision="bf16"))  # no raise

    def test_wrong_backend_field_rejected(self):
        backend = _bare_vllm_backend()
        with pytest.raises(RequestProvenanceMismatchError, match="backend"):
            backend._validate_request(_request(backend="transformers"))

    def test_precision_mismatch_rejected(self):
        """The core regression case: an engine built with dtype=bf16 must
        never accept (and silently honor) a request claiming fp32 - that
        would write a record claiming FP32 produced a BF16 result."""
        backend = _bare_vllm_backend(precision="bf16")
        with pytest.raises(RequestProvenanceMismatchError, match="precision"):
            backend._validate_request(_request(backend="vllm", precision="fp32"))

    def test_wrong_model_id_rejected(self):
        backend = _bare_vllm_backend(model_id="model/x", precision="bf16")
        with pytest.raises(RequestProvenanceMismatchError, match="model_id"):
            backend._validate_request(_request(backend="vllm", model_id="model/y", precision="bf16"))
