"""GPU minimum-power-limit preflight.

HARD RULE: before anatomiae runs any CUDA workload on the maintainer
workstation's physical GPU 1, that card must first be set to its
device-reported *minimum* supported power limit (queried live, never
assumed - e.g. do not hard-code 100 W). This is a throughput-affecting
experimental variable, not a cosmetic setting: every maintainer throughput
number must be read together with the applied power limit (see
`docs/reproducibility.md`).

This module only handles the power-limit step. Combine it with
`anatomiae.provenance.gpu_guard.verify_gpu_isolation` /
`assert_no_tensor_parallel` via `full_gpu_preflight()` below for the
complete ordered sequence:

    physical GPU exists
    -> minimum power limit discovered
    -> minimum power limit applied (or already there)
    -> applied power limit verified
    -> CUDA_VISIBLE_DEVICES exposes only the target physical GPU
    -> UUID/PCI identity guard passes
    -> tensor_parallel_size == 1
    -> CUDA workload may start

No CUDA context may be created before `full_gpu_preflight()` returns
successfully.
"""

from __future__ import annotations

import dataclasses
import re
import subprocess
from collections.abc import Callable

from anatomiae.provenance.gpu_guard import (
    DEFAULT_EXPECTED_PHYSICAL_INDEX,
    GPUProvenance,
    assert_no_tensor_parallel,
    verify_gpu_isolation,
)

Runner = Callable[[list[str]], "subprocess.CompletedProcess[str]"]


class PowerLimitError(RuntimeError):
    """Base class for power-preflight failures. Callers must abort, not
    fall back to an unverified power state."""


class HumanActionRequiredError(PowerLimitError):
    """Raised when the current power limit is not at the device minimum
    and non-interactive sudo is unavailable to fix it. This is a genuine
    human-only blocker (see docs/HUMAN_ACTION_REQUIRED.md) - never store
    or prompt for a sudo password from project code."""


@dataclasses.dataclass(frozen=True)
class PowerLimitState:
    physical_index: int
    min_power_limit_w: float
    max_power_limit_w: float
    default_power_limit_w: float
    current_power_limit_w: float
    applied_power_limit_w: float
    action_taken: str  # "already_at_minimum" | "applied_via_sudo"


def _default_runner(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, capture_output=True, text=True, check=False)


_POWER_FIELDS = {
    "current": "Current Power Limit",
    "default": "Default Power Limit",
    "min": "Min Power Limit",
    "max": "Max Power Limit",
}


def query_power_info(physical_index: int, *, runner: Runner = _default_runner) -> dict[str, float]:
    """Parse `nvidia-smi -q -i <idx> -d POWER` for the *top-level* GPU Power
    Readings block. Only ever queries the single requested index - this
    cannot itself observe "more than one GPU", by construction; that check
    belongs to the CUDA_VISIBLE_DEVICES stage in gpu_guard."""
    result = runner(["nvidia-smi", "-q", "-i", str(physical_index), "-d", "POWER"])
    if result.returncode != 0:
        raise PowerLimitError(
            f"nvidia-smi power query failed for physical GPU {physical_index} "
            f"(exit {result.returncode}): {result.stderr.strip() or result.stdout.strip()}"
        )

    # Only the first "GPU Power Readings" block (top-level card, not the
    # "Module Power Readings" sub-block, which reports N/A on this hardware
    # and would otherwise silently shadow the real values via a greedy regex).
    block_match = re.search(
        r"GPU Power Readings(.*?)(?:Power Samples|Module Power Readings)", result.stdout, re.DOTALL
    )
    block = block_match.group(1) if block_match else result.stdout

    values: dict[str, float] = {}
    for key, label in _POWER_FIELDS.items():
        m = re.search(rf"{re.escape(label)}\s*:\s*([\d.]+)\s*W", block)
        if not m:
            raise PowerLimitError(
                f"Could not parse {label!r} for physical GPU {physical_index} from nvidia-smi "
                "output - refusing to guess a power limit. Raw output follows.\n" + result.stdout
            )
        values[key] = float(m.group(1))
    return values


def ensure_minimum_power_limit(
    *,
    physical_index: int | None = None,
    runner: Runner = _default_runner,
) -> PowerLimitState:
    """Query and, if needed, apply the device-reported minimum power limit
    for the target physical GPU. Never assumes a specific wattage."""
    idx = DEFAULT_EXPECTED_PHYSICAL_INDEX if physical_index is None else physical_index

    info = query_power_info(idx, runner=runner)
    minimum, current = info["min"], info["current"]

    if abs(current - minimum) < 0.01:
        return PowerLimitState(
            physical_index=idx,
            min_power_limit_w=minimum,
            max_power_limit_w=info["max"],
            default_power_limit_w=info["default"],
            current_power_limit_w=current,
            applied_power_limit_w=current,
            action_taken="already_at_minimum",
        )

    set_result = runner(["sudo", "-n", "nvidia-smi", "-i", str(idx), "-pl", str(int(minimum))])
    if set_result.returncode != 0:
        raise HumanActionRequiredError(
            f"Physical GPU {idx} power limit is {current}W; device minimum is {minimum}W. "
            "Non-interactive sudo is unavailable to apply it (this project never stores or "
            "prompts for a sudo password). A human must run:\n"
            f"  sudo nvidia-smi -i {idx} -pl {int(minimum)}\n"
            "then re-run the preflight. See docs/HUMAN_ACTION_REQUIRED.md."
        )

    verify = query_power_info(idx, runner=runner)
    if abs(verify["current"] - minimum) > 0.01:
        raise PowerLimitError(
            f"Applied power limit did not take effect on physical GPU {idx}: "
            f"requested {minimum}W, nvidia-smi now reports {verify['current']}W."
        )

    return PowerLimitState(
        physical_index=idx,
        min_power_limit_w=minimum,
        max_power_limit_w=info["max"],
        default_power_limit_w=info["default"],
        current_power_limit_w=current,
        applied_power_limit_w=verify["current"],
        action_taken="applied_via_sudo",
    )


@dataclasses.dataclass(frozen=True)
class FullPreflightResult:
    power: PowerLimitState
    gpu: GPUProvenance


def full_gpu_preflight(
    *,
    physical_index: int | None = None,
    tensor_parallel_size: int = 1,
    power_runner: Runner = _default_runner,
    nvml_snapshot=None,
) -> FullPreflightResult:
    """The complete ordered preflight. Raises and aborts on any violation;
    callers must not catch these and fall back to running anyway.

    Order matters: power limit is applied before the isolation guard runs,
    so a workload can never start under an unverified power state even if
    the isolation check would otherwise pass.
    """
    idx = DEFAULT_EXPECTED_PHYSICAL_INDEX if physical_index is None else physical_index

    power_state = ensure_minimum_power_limit(physical_index=idx, runner=power_runner)
    gpu_state = verify_gpu_isolation(expected_physical_index=idx, nvml_snapshot=nvml_snapshot)
    assert_no_tensor_parallel(tensor_parallel_size)

    return FullPreflightResult(power=power_state, gpu=gpu_state)


if __name__ == "__main__":
    import dataclasses as _dc
    import json

    result = full_gpu_preflight()
    print(
        json.dumps(
            {"power": _dc.asdict(result.power), "gpu": _dc.asdict(result.gpu)},
            indent=2,
        )
    )
