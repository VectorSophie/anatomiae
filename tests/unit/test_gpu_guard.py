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
    with pytest.raises(GPUIsolationError, match="locked to physical GPU 1"):
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


def test_tensor_parallel_rejected():
    with pytest.raises(GPUIsolationError, match="tensor_parallel_size"):
        assert_no_tensor_parallel(2)


def test_tensor_parallel_1_ok():
    assert_no_tensor_parallel(1)
