"""Transformers and vLLM generation backends behind a shared interface.

Both backends call `full_gpu_preflight()` before creating any CUDA context
- no model load, no engine construction, until the preflight (power limit
-> isolation guard -> tensor-parallel check) returns successfully.

vLLM is configured via the current programmatic `AttentionConfig(backend=...)`
field (see docs/reproducibility.md) rather than the deprecated
`VLLM_ATTENTION_BACKEND` env var - FlashInfer is required on the
maintainer workstation's Blackwell GPU (see docs/MODEL_AUDIT.md for the
underlying flash-attn/PTX-toolchain incompatibility this works around).
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod

from anatomiae.inference.schema import (
    FinishReason,
    GenerationRecord,
    GenerationRequest,
    GPURecordProvenance,
)
from anatomiae.provenance.gpu_power import FullPreflightResult, full_gpu_preflight

_DTYPE_MAP = {"fp32": "float32", "bf16": "bfloat16", "fp16": "float16"}


class RequestProvenanceMismatchError(RuntimeError):
    """Raised when a GenerationRequest's declared identity does not match
    the backend/model/precision actually loaded. The experimental record
    must never claim a different condition from the one that actually
    produced it - this is checked and raised *before* any model call, so
    no misleading GenerationRecord is ever written for a mismatched
    request."""


class GenerationBackend(ABC):
    @abstractmethod
    def generate(self, request: GenerationRequest) -> GenerationRecord: ...


def _gpu_record_provenance(preflight: FullPreflightResult) -> GPURecordProvenance:
    return GPURecordProvenance(
        physical_index=preflight.gpu.physical_index,
        uuid=preflight.gpu.uuid,
        driver_version=preflight.gpu.driver_version,
        power_limit_w=preflight.power.applied_power_limit_w,
    )


class TransformersBackend(GenerationBackend):
    """Loads one model per (precision) on first use - needed for the
    FP32-vs-BF16 precision-sensitivity comparison (docs/results/
    precision_sensitivity.md), which must run the *same* checkpoint under
    two dtypes without reloading between every single request."""

    def __init__(self, model_id: str, revision: str = "main", *, tensor_parallel_size: int = 1):
        preflight = full_gpu_preflight(tensor_parallel_size=tensor_parallel_size)
        self._gpu_provenance = _gpu_record_provenance(preflight)
        self._model_id = model_id
        self._revision = revision
        self._models: dict[str, object] = {}

        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)

    def _validate_request(self, request: GenerationRequest) -> None:
        problems = []
        if request.backend != "transformers":
            problems.append(f"request.backend={request.backend!r} but this is TransformersBackend")
        if request.model_id != self._model_id:
            problems.append(f"request.model_id={request.model_id!r} != loaded {self._model_id!r}")
        if request.model_revision != self._revision:
            problems.append(
                f"request.model_revision={request.model_revision!r} != loaded {self._revision!r}"
            )
        if request.tokenizer_revision != self._revision:
            problems.append(
                f"request.tokenizer_revision={request.tokenizer_revision!r} != loaded {self._revision!r}"
            )
        if request.precision not in _DTYPE_MAP:
            problems.append(f"request.precision={request.precision!r} is not a supported dtype")
        if problems:
            raise RequestProvenanceMismatchError(
                "Refusing to generate - request identity does not match this backend instance: "
                + "; ".join(problems)
            )

    def _model_for_precision(self, precision: str):
        if precision not in self._models:
            import torch
            from transformers import AutoModelForCausalLM

            dtype = getattr(torch, _DTYPE_MAP[precision])
            model = AutoModelForCausalLM.from_pretrained(
                self._model_id, revision=self._revision, dtype=dtype
            ).to("cuda")
            self._models[precision] = model
        return self._models[precision]

    def generate(self, request: GenerationRequest) -> GenerationRecord:
        self._validate_request(request)

        import torch

        model = self._model_for_precision(request.precision)
        input_ids = self.tokenizer(request.rendered_text, return_tensors="pt").input_ids.to("cuda")
        t0 = time.time()
        try:
            gen_kwargs: dict = {"max_new_tokens": request.decoding.max_new_tokens}
            if request.decoding.temperature and request.decoding.temperature > 0:
                # Stochastic decoding: `request.decoding.seed` is part of
                # generation identity (GenerationRequest.cache_key()), so it
                # must actually control the sampling RNG, not just be
                # recorded decoratively.
                #
                # A per-call `torch.Generator` passed as `generate(...,
                # generator=g)` was tried first and does NOT work on this
                # transformers version (4.57.6): `generate()`'s **kwargs are
                # merged into GenerationConfig fields only, `generator` is
                # not one, and it raises `ValueError: The following
                # model_kwargs are not used by the model: ['generator']`
                # rather than silently ignoring it - caught by the real
                # integration check in
                # scripts/seed_reproducibility_check.py, not assumed to
                # work. Global seeding immediately before the call is the
                # mechanism this library version actually supports; it is
                # deterministic and reproducible given (model, prompt,
                # seed), which is the cache-key invariant that matters here.
                # It is not isolated against concurrent calls, but this
                # backend only ever runs one generate() call at a time
                # (synchronous), so that is not a practical risk.
                if request.decoding.seed is not None:
                    torch.manual_seed(request.decoding.seed)
                    torch.cuda.manual_seed_all(request.decoding.seed)
                gen_kwargs.update(
                    do_sample=True,
                    temperature=request.decoding.temperature,
                    top_p=request.decoding.top_p,
                )
            else:
                # Greedy decoding is deterministic given the model/prompt;
                # seed is scientifically irrelevant here by construction -
                # intentionally not passed to `generate()`.
                gen_kwargs.update(do_sample=False, temperature=None, top_p=None)

            with torch.no_grad():
                output_ids = model.generate(input_ids, **gen_kwargs)
            latency = time.time() - t0

            completion_ids = output_ids[0][input_ids.shape[1] :]
            raw_text = self.tokenizer.decode(completion_ids, skip_special_tokens=True)
            finish_reason: FinishReason = (
                "length" if completion_ids.shape[0] >= request.decoding.max_new_tokens else "stop"
            )
            return GenerationRecord.build(
                request=request,
                raw_text=raw_text,
                finish_reason=finish_reason,
                input_tokens=int(input_ids.shape[1]),
                output_tokens=int(completion_ids.shape[0]),
                latency_seconds=latency,
                gpu=self._gpu_provenance,
            )
        except Exception as e:  # noqa: BLE001 - captured as a first-class outcome, not swallowed
            return GenerationRecord.build(
                request=request,
                raw_text="",
                finish_reason="error",
                input_tokens=int(input_ids.shape[1]),
                output_tokens=0,
                latency_seconds=time.time() - t0,
                gpu=self._gpu_provenance,
                error=repr(e),
            )


class VLLMBackend(GenerationBackend):
    def __init__(
        self,
        model_id: str,
        revision: str = "main",
        *,
        precision: str = "bf16",
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.5,
    ):
        preflight = full_gpu_preflight(tensor_parallel_size=tensor_parallel_size)
        self._gpu_provenance = _gpu_record_provenance(preflight)

        from vllm import LLM
        from vllm.config import AttentionConfig
        from vllm.v1.attention.backends.registry import AttentionBackendEnum

        self._llm = LLM(
            model=model_id,
            revision=revision,
            dtype=_DTYPE_MAP[precision],
            gpu_memory_utilization=gpu_memory_utilization,
            enforce_eager=True,
            attention_config=AttentionConfig(backend=AttentionBackendEnum.FLASHINFER),
        )
        self._model_id = model_id
        self._revision = revision
        # The engine's dtype is fixed at construction and cannot change
        # per-request - unlike TransformersBackend, which loads a model per
        # precision on demand, a vLLM engine built with dtype=bf16 must
        # never accept a request claiming precision=fp32 and silently
        # generate with (and then record) the wrong condition.
        self._precision = precision

    def _validate_request(self, request: GenerationRequest) -> None:
        problems = []
        if request.backend != "vllm":
            problems.append(f"request.backend={request.backend!r} but this is VLLMBackend")
        if request.model_id != self._model_id:
            problems.append(f"request.model_id={request.model_id!r} != loaded {self._model_id!r}")
        if request.model_revision != self._revision:
            problems.append(
                f"request.model_revision={request.model_revision!r} != loaded {self._revision!r}"
            )
        if request.precision != self._precision:
            problems.append(
                f"request.precision={request.precision!r} != engine dtype {self._precision!r} "
                "(the vLLM engine's dtype is fixed at construction, not per-request)"
            )
        if problems:
            raise RequestProvenanceMismatchError(
                "Refusing to generate - request identity does not match this backend instance: "
                + "; ".join(problems)
            )

    def generate(self, request: GenerationRequest) -> GenerationRecord:
        self._validate_request(request)

        from vllm import SamplingParams

        sampling_params = SamplingParams(
            temperature=request.decoding.temperature,
            top_p=request.decoding.top_p,
            max_tokens=request.decoding.max_new_tokens,
            seed=request.decoding.seed,
        )
        t0 = time.time()
        try:
            outputs = self._llm.generate([request.rendered_text], sampling_params)
            latency = time.time() - t0
            completion = outputs[0].outputs[0]
            finish_reason: FinishReason = "length" if completion.finish_reason == "length" else "stop"
            return GenerationRecord.build(
                request=request,
                raw_text=completion.text,
                finish_reason=finish_reason,
                input_tokens=len(outputs[0].prompt_token_ids or []),
                output_tokens=len(completion.token_ids),
                latency_seconds=latency,
                gpu=self._gpu_provenance,
            )
        except Exception as e:  # noqa: BLE001
            return GenerationRecord.build(
                request=request,
                raw_text="",
                finish_reason="error",
                input_tokens=0,
                output_tokens=0,
                latency_seconds=time.time() - t0,
                gpu=self._gpu_provenance,
                error=repr(e),
            )
