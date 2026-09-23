"""Unit tests for the GPU minimum-power-limit preflight.

All nvidia-smi/sudo calls are mocked via an injected `runner`. These tests
must never invoke a real `sudo nvidia-smi -pl` command - that would either
fail loudly (no passwordless sudo in CI/dev, which is fine) or, far worse,
actually mutate real GPU hardware state from a test run. Every test here
asserts that never happens by using a fake runner and, for the "verify
sudo never runs" cases, asserting the mock was not called with `sudo`.
"""

from __future__ import annotations

import pytest

from anatomiae.provenance.gpu_power import (
    HumanActionRequiredError,
    PowerLimitError,
    ensure_minimum_power_limit,
    full_gpu_preflight,
    query_power_info,
)


def _power_block(current: float, default: float, minimum: float, maximum: float) -> str:
    return f"""
    GPU Power Readings
        Average Power Draw                : 13.48 W
        Instantaneous Power Draw          : 13.45 W
        Current Power Limit               : {current:.2f} W
        Requested Power Limit             : {current:.2f} W
        Default Power Limit               : {default:.2f} W
        Min Power Limit                   : {minimum:.2f} W
        Max Power Limit                   : {maximum:.2f} W
    Power Samples
        Duration                          : 1.00 sec
    Module Power Readings
        Average Power Draw                : N/A
        Current Power Limit               : N/A
        Min Power Limit                   : N/A
    """


class FakeRunner:
    """Records calls and returns scripted responses in order."""

    def __init__(self, responses: list[tuple[int, str, str]]):
        self.responses = list(responses)
        self.calls: list[list[str]] = []

    def __call__(self, cmd: list[str]):
        self.calls.append(cmd)
        returncode, stdout, stderr = self.responses.pop(0)

        class _Result:
            pass

        r = _Result()
        r.returncode = returncode
        r.stdout = stdout
        r.stderr = stderr
        return r

    def used_sudo(self) -> bool:
        return any(c and c[0] == "sudo" for c in self.calls)


@pytest.mark.parametrize("minimum", [100.0, 80.0, 120.0])
def test_minimum_variants_are_read_from_device_not_assumed(minimum):
    """Regression guard: the code must never hard-code a wattage - it must
    apply whatever the device itself reports as its minimum."""
    current = minimum + 50  # not already at minimum, forces a set attempt
    runner = FakeRunner(
        [
            (0, _power_block(current, 300.0, minimum, 325.0), ""),  # initial query
            (0, "", ""),  # sudo -n nvidia-smi -pl succeeds
            (0, _power_block(minimum, 300.0, minimum, 325.0), ""),  # verify query
        ]
    )
    state = ensure_minimum_power_limit(physical_index=1, runner=runner)
    assert state.applied_power_limit_w == minimum
    assert state.min_power_limit_w == minimum
    assert state.action_taken == "applied_via_sudo"
    # the exact minimum was passed to nvidia-smi -pl, not a hard-coded value
    set_call = runner.calls[1]
    assert set_call[:2] == ["sudo", "-n"]
    assert str(int(minimum)) in set_call


def test_already_at_minimum_does_not_invoke_sudo():
    runner = FakeRunner([(0, _power_block(250.0, 300.0, 250.0, 325.0), "")])
    state = ensure_minimum_power_limit(physical_index=1, runner=runner)
    assert state.action_taken == "already_at_minimum"
    assert state.applied_power_limit_w == 250.0
    assert not runner.used_sudo()
    assert len(runner.calls) == 1  # only the initial query - no set, no re-verify


def test_minimum_unavailable_fails_closed():
    """nvidia-smi output missing the Min Power Limit field specifically -
    refuse to guess, abort (other fields present so the failure is
    unambiguously about the missing minimum)."""
    broken_output = (
        "GPU Power Readings\n"
        "    Current Power Limit               : 250.00 W\n"
        "    Default Power Limit               : 300.00 W\n"
        "    Max Power Limit                   : 325.00 W\n"
    )
    runner = FakeRunner([(0, broken_output, "")])
    with pytest.raises(PowerLimitError, match="Min Power Limit"):
        ensure_minimum_power_limit(physical_index=1, runner=runner)


def test_nvidia_smi_query_failure_fails_closed():
    runner = FakeRunner([(1, "", "No devices were found")])
    with pytest.raises(PowerLimitError, match="power query failed"):
        query_power_info(1, runner=runner)


def test_sudo_unavailable_raises_human_action_required():
    runner = FakeRunner(
        [
            (0, _power_block(250.0, 300.0, 100.0, 325.0), ""),  # not at minimum
            (1, "", "sudo: a password is required"),  # sudo -n fails
        ]
    )
    with pytest.raises(HumanActionRequiredError, match="sudo nvidia-smi -i 1 -pl 100"):
        ensure_minimum_power_limit(physical_index=1, runner=runner)


def test_applied_limit_not_taking_effect_fails_closed():
    """sudo reports success but a re-query shows the limit didn't actually
    change - do not trust the exit code alone."""
    runner = FakeRunner(
        [
            (0, _power_block(250.0, 300.0, 100.0, 325.0), ""),
            (0, "", ""),  # sudo claims success
            (0, _power_block(250.0, 300.0, 100.0, 325.0), ""),  # but limit is still 250
        ]
    )
    with pytest.raises(PowerLimitError, match="did not take effect"):
        ensure_minimum_power_limit(physical_index=1, runner=runner)


# ---- Combined orchestrator (full_gpu_preflight): power + isolation guard ----

FAKE_NVML_SYSTEM = {
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


def _nvml_snapshot():
    return FAKE_NVML_SYSTEM


def test_full_preflight_happy_path(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    runner = FakeRunner([(0, _power_block(250.0, 300.0, 250.0, 325.0), "")])
    result = full_gpu_preflight(physical_index=1, power_runner=runner, nvml_snapshot=_nvml_snapshot)
    assert result.power.action_taken == "already_at_minimum"
    assert result.gpu.physical_index == 1


def test_full_preflight_aborts_on_wrong_gpu(monkeypatch):
    """CUDA_VISIBLE_DEVICES resolves to physical GPU 0, not the requested 1 -
    must abort even though the power step for GPU 1 alone would succeed."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    runner = FakeRunner([(0, _power_block(250.0, 300.0, 250.0, 325.0), "")])
    from anatomiae.provenance.gpu_guard import GPUIsolationError

    with pytest.raises(GPUIsolationError, match="configured for physical GPU 1"):
        full_gpu_preflight(physical_index=1, power_runner=runner, nvml_snapshot=_nvml_snapshot)


def test_full_preflight_aborts_on_multiple_visible_gpus(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    runner = FakeRunner([(0, _power_block(250.0, 300.0, 250.0, 325.0), "")])
    from anatomiae.provenance.gpu_guard import GPUIsolationError

    with pytest.raises(GPUIsolationError, match="at most"):
        full_gpu_preflight(physical_index=1, power_runner=runner, nvml_snapshot=_nvml_snapshot)


def test_full_preflight_rejects_tensor_parallel(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    runner = FakeRunner([(0, _power_block(250.0, 300.0, 250.0, 325.0), "")])
    from anatomiae.provenance.gpu_guard import GPUIsolationError

    with pytest.raises(GPUIsolationError, match="tensor_parallel_size"):
        full_gpu_preflight(
            physical_index=1,
            tensor_parallel_size=2,
            power_runner=runner,
            nvml_snapshot=_nvml_snapshot,
        )


def test_full_preflight_never_touches_gpu0_power(monkeypatch):
    """The power step must only ever be invoked for the target index (1) -
    never GPU 0, even implicitly."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    runner = FakeRunner([(0, _power_block(250.0, 300.0, 250.0, 325.0), "")])
    full_gpu_preflight(physical_index=1, power_runner=runner, nvml_snapshot=_nvml_snapshot)
    for call in runner.calls:
        assert "-i" in call
        assert call[call.index("-i") + 1] == "1"
