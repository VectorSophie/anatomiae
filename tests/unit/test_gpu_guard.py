import importlib.util

import pytest

from anatomiae.provenance.gpu_guard import (
    EXPECTED_PHYSICAL_INDEX,
    GPUIsolationError,
    assert_no_tensor_parallel,
    verify_gpu_isolation,
)

FAKE_SYSTEM = {
    0: {
        "uuid": "GPU-fce8b8f8-013c-de7a-bcc3-3b9f651d097b",
        "pci_bus_id": "00000000:4A:00.0",
        "name": "NVIDIA RTX PRO 6000 Blackwell",
        "driver_version": "570.211.01",
        "total_memory_mib": 97887,
    },
    1: {
        "uuid": "GPU-1824bc0e-afda-4c3f-454b-71e779e4db34",
        "pci_bus_id": "00000000:CA:00.0",
        "name": "NVIDIA RTX PRO 6000 Blackwell",
        "driver_version": "570.211.01",
        "total_memory_mib": 97887,
    },
}


def _snapshot():
    return FAKE_SYSTEM


def test_missing_cuda_visible_devices_aborts(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    with pytest.raises(GPUIsolationError, match="unset"):
        verify_gpu_isolation(nvml_snapshot=_snapshot)


def test_multiple_devices_aborts(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    with pytest.raises(GPUIsolationError, match="at most"):
        verify_gpu_isolation(nvml_snapshot=_snapshot)


def test_physical_gpu_0_aborts(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    with pytest.raises(GPUIsolationError, match="configured for physical GPU 1"):
        verify_gpu_isolation(nvml_snapshot=_snapshot)


def test_physical_gpu_1_by_index_succeeds(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    prov = verify_gpu_isolation(nvml_snapshot=_snapshot)
    assert prov.physical_index == EXPECTED_PHYSICAL_INDEX
    assert prov.uuid == FAKE_SYSTEM[1]["uuid"]
    assert prov.pci_bus_id == FAKE_SYSTEM[1]["pci_bus_id"]


def test_physical_gpu_1_by_uuid_succeeds(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", FAKE_SYSTEM[1]["uuid"])
    prov = verify_gpu_isolation(nvml_snapshot=_snapshot)
    assert prov.physical_index == 1


def test_unknown_index_aborts(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "7")
    with pytest.raises(GPUIsolationError, match="not present"):
        verify_gpu_isolation(nvml_snapshot=_snapshot)


def test_garbage_token_aborts(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "not-a-gpu")
    with pytest.raises(GPUIsolationError, match="Cannot parse"):
        verify_gpu_isolation(nvml_snapshot=_snapshot)


# Deliberately fictitious - unrelated to any real hardware on the machine
# running these tests, so an override test can never accidentally resolve
# to (and thus, via the torch cross-check, transiently touch) a real GPU
# index the guard is supposed to forbid.
OTHER_HARDWARE_SYSTEM = {
    0: {
        "uuid": "GPU-00000000-0000-0000-0000-000000000000",
        "pci_bus_id": "00000000:01:00.0",
        "name": "Fictitious Test GPU 0",
        "driver_version": "000.00.00",
        "total_memory_mib": 24576,
    },
}


def _other_hardware_snapshot():
    return OTHER_HARDWARE_SYSTEM


def test_expected_index_is_overridable_for_other_hardware(monkeypatch):
    """The maintainers' physical-GPU-1 rule is a local workstation setting,
    not a portable scientific requirement - external reproducers must be
    able to point the guard at their own hardware. Uses a fictitious NVML
    snapshot (not the real machine's) and, when torch is installed,
    disables its CUDA cross-check so this test can never touch real
    hardware, including physical GPU 0 on the machine actually running the
    test suite. torch is optional (see pyproject.toml's `ml` extra - CI
    runs without it), so only patch it if it's actually importable;
    verify_gpu_isolation itself already skips the torch cross-check via
    its own `except ImportError` when torch is absent."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    if importlib.util.find_spec("torch") is not None:
        monkeypatch.setattr("torch.cuda.is_available", lambda: False, raising=False)
    prov = verify_gpu_isolation(
        nvml_snapshot=_other_hardware_snapshot, expected_physical_index=0
    )
    assert prov.physical_index == 0


def test_expected_index_env_var_override():
    """ANATOMIAE_EXPECTED_GPU_INDEX is read once at module import time, so
    this must run in a fresh subprocess rather than via importlib.reload()
    in-process: reload() re-executes the module against its *same*
    __dict__, which rebinds module-level classes (GPUIsolationError etc.)
    to new objects while other already-imported references in this test
    file keep pointing at the old ones - a real bug hit once already,
    where a reload here silently broke `pytest.raises(GPUIsolationError)`
    in an unrelated, later-running test via an isinstance mismatch between
    the pre- and post-reload exception classes."""
    import subprocess
    import sys
    from pathlib import Path

    code = (
        "import os; "
        "os.environ['ANATOMIAE_EXPECTED_GPU_INDEX'] = '0'; "
        "from anatomiae.provenance.gpu_guard import DEFAULT_EXPECTED_PHYSICAL_INDEX; "
        "assert DEFAULT_EXPECTED_PHYSICAL_INDEX == 0, DEFAULT_EXPECTED_PHYSICAL_INDEX; "
        "print('OK')"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).resolve().parents[2]),
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout


def test_tensor_parallel_rejected():
    with pytest.raises(GPUIsolationError, match="tensor_parallel_size"):
        assert_no_tensor_parallel(2)


def test_tensor_parallel_1_ok():
    assert_no_tensor_parallel(1)
