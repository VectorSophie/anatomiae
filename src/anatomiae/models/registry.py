"""Model registry schema (project spec §43).

Loads the YAML configs under configs/models/ into a typed record. Model
lineage (parent_checkpoint) must be explicit so stage-wise analyses (Base
-> SFT -> DPO -> RLVR, or checkpoint-t -> checkpoint-(t+1)) can walk the
graph without re-deriving it from naming conventions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict

CONFIGS_DIR = Path(__file__).resolve().parents[3] / "configs" / "models"


class TransparencyInfo(BaseModel):
    model_config = ConfigDict(frozen=True)

    tier: str
    pretraining_data_reconstructable: bool
    training_code_available: bool
    intermediate_checkpoints_available: bool


class LicenseInfo(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    gated: bool = False


class InferenceConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    backend: Literal["vllm", "transformers"] = "vllm"
    dtype: str = "bfloat16"
    tensor_parallel_size: int = 1


class ModelEntry(BaseModel):
    model_config = ConfigDict(frozen=True)

    family: str
    provider: str
    checkpoint: str
    revision: str
    parameters_total: str | None = None
    parameters_active: str | None = None
    architecture: str = "dense"
    training_stage: Literal["base", "sft", "dpo", "rm", "rlvr", "instruct", "unknown"]
    parent_checkpoint: str | None = None

    transparency: TransparencyInfo
    license: LicenseInfo
    inference: InferenceConfig = InferenceConfig()

    locked_anchor: bool = False
    scientific_role: str = ""


def load_registry(configs_dir: Path = CONFIGS_DIR) -> dict[str, ModelEntry]:
    entries: dict[str, ModelEntry] = {}
    for path in sorted(configs_dir.glob("*.yaml")):
        data = yaml.safe_load(path.read_text())
        entries[path.stem] = ModelEntry.model_validate(data)
    return entries
