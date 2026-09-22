"""GPU isolation preflight guard.

ABSOLUTE RULE (anatomiae research spec, GPU isolation): only the physical GPU
reported as index 1 by `nvidia-smi` may be used. Physical GPU 0 must never be
touched or exposed to a CUDA process, and no job may request more than one
device. This module must be called, and must succeed, before any CUDA context
is created (before `.cuda()`, before constructing a vLLM engine).

On any violation this raises GPUIsolationError. Callers must let the job
abort rather than catch this and fall back to a different device.
"""

from __future__ import annotations

import dataclasses
import os

EXPECTED_PHYSICAL_INDEX = 1


class GPUIsolationError(RuntimeError):
    pass


@dataclasses.dataclass(frozen=True)
class GPUProvenance:
    physical_index: int
    uuid: str
    pci_bus_id: str
    name: str
    driver_version: str
    cuda_runtime_version: str
    total_memory_mib: int
    logical_index_in_process: int


def _nvml_snapshot() -> dict[int, dict]:
    import pynvml

    pynvml.nvmlInit()
    try:
        driver_version = pynvml.nvmlSystemGetDriverVersion()
        if isinstance(driver_version, bytes):
            driver_version = driver_version.decode()
        count = pynvml.nvmlDeviceGetCount()
        out: dict[int, dict] = {}
        for i in range(count):
            h = pynvml.nvmlDeviceGetHandleByIndex(i)
            uuid = pynvml.nvmlDeviceGetUUID(h)
            if isinstance(uuid, bytes):
                uuid = uuid.decode()
            pci = pynvml.nvmlDeviceGetPciInfo(h)
            bus_id = pci.busId
            if isinstance(bus_id, bytes):
                bus_id = bus_id.decode()
            name = pynvml.nvmlDeviceGetName(h)
            if isinstance(name, bytes):
                name = name.decode()
            mem = pynvml.nvmlDeviceGetMemoryInfo(h)
            out[i] = {
                "uuid": uuid,
                "pci_bus_id": bus_id,
                "name": name,
                "driver_version": driver_version,
                "total_memory_mib": mem.total // (1024 * 1024),
            }
        return out
    finally:
        pynvml.nvmlShutdown()


def verify_gpu_isolation(
    *, max_devices: int = 1, nvml_snapshot=None
) -> GPUProvenance:
    """Verify the process is scoped to exactly the approved physical GPU.

    `nvml_snapshot` is an injection point for tests; production callers
    should leave it as None (real NVML query).
    """
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not cvd:
        raise GPUIsolationError(
            "CUDA_VISIBLE_DEVICES is unset - refusing to guess which GPU to use. "
            f"Set CUDA_VISIBLE_DEVICES={EXPECTED_PHYSICAL_INDEX} before launching."
        )
    ids = [x.strip() for x in cvd.split(",") if x.strip()]
    if len(ids) > max_devices:
        raise GPUIsolationError(
            f"CUDA_VISIBLE_DEVICES={cvd!r} exposes {len(ids)} devices; "
            f"this project permits at most {max_devices} (single-GPU only)."
        )

    system = (nvml_snapshot or _nvml_snapshot)()

    token = ids[0]
    if token.upper().startswith(("GPU-", "MIG-")):
        matches = [idx for idx, info in system.items() if info["uuid"] == token]
        if not matches:
            raise GPUIsolationError(f"CUDA_VISIBLE_DEVICES UUID {token!r} not found on this host.")
        physical_index = matches[0]
    else:
        try:
            physical_index = int(token)
        except ValueError as e:
            raise GPUIsolationError(f"Cannot parse CUDA_VISIBLE_DEVICES entry {token!r}") from e
        if physical_index not in system:
            raise GPUIsolationError(f"Physical GPU index {physical_index} not present on this host.")

    if physical_index != EXPECTED_PHYSICAL_INDEX:
        raise GPUIsolationError(
            f"CUDA_VISIBLE_DEVICES resolves to physical GPU {physical_index}, but this "
            f"project is locked to physical GPU {EXPECTED_PHYSICAL_INDEX}. Aborting."
        )

    info = system[physical_index]

    cuda_runtime_version = "unknown"
    logical_index = 0
    try:
        import torch

        if torch.cuda.is_available():
            if torch.cuda.device_count() != 1:
                raise GPUIsolationError(
                    f"torch reports {torch.cuda.device_count()} visible CUDA devices; expected exactly 1."
                )
            cuda_runtime_version = torch.version.cuda or "unknown"
            props = torch.cuda.get_device_properties(0)
            # UUID is the authoritative cross-check: NVML's reported total
            # memory and CUDA runtime's total_memory differ by driver-
            # reserved amounts (observed ~629 MiB on 96 GB Blackwell cards),
            # so memory size alone is too strict/unreliable for identity.
            seen_uuid = f"GPU-{props.uuid}"
            if seen_uuid != info["uuid"]:
                raise GPUIsolationError(
                    "torch-visible device UUID does not match the expected physical GPU "
                    f"{EXPECTED_PHYSICAL_INDEX} (expected {info['uuid']}, saw {seen_uuid}). "
                    "Aborting rather than trusting the mapping."
                )
            seen_total_mib = props.total_memory // (1024 * 1024)
            if abs(seen_total_mib - info["total_memory_mib"]) > 4096:
                raise GPUIsolationError(
                    "torch-visible device memory is wildly different from NVML's report for "
                    f"physical GPU {EXPECTED_PHYSICAL_INDEX} (expected ~{info['total_memory_mib']} "
                    f"MiB, saw {seen_total_mib} MiB) despite matching UUID - aborting out of caution."
                )
    except ImportError:
        pass

    return GPUProvenance(
        physical_index=physical_index,
        uuid=info["uuid"],
        pci_bus_id=info["pci_bus_id"],
        name=info["name"],
        driver_version=info["driver_version"],
        cuda_runtime_version=cuda_runtime_version,
        total_memory_mib=info["total_memory_mib"],
        logical_index_in_process=logical_index,
    )


def assert_no_tensor_parallel(tensor_parallel_size: int) -> None:
    if tensor_parallel_size > 1:
        raise GPUIsolationError(
            f"tensor_parallel_size={tensor_parallel_size} requested; this project has exactly "
            "one usable physical GPU and must never request multi-device tensor parallelism."
        )


if __name__ == "__main__":
    import json

    prov = verify_gpu_isolation()
    assert_no_tensor_parallel(1)
    print(json.dumps(dataclasses.asdict(prov), indent=2))
