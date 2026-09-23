"""CPU-safe sanity check of the RNG mechanism backends.py relies on for
seeded sampling (torch.Generator(...).manual_seed(seed)). Does not touch a
GPU or load a model - that's covered by the real integration check in
scripts/seed_reproducibility_check.py, which needs actual hardware.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")  # needs the `ml` extra; CI runs `dev` only


def test_same_seed_produces_same_draws():
    g1 = torch.Generator(device="cpu").manual_seed(42)
    g2 = torch.Generator(device="cpu").manual_seed(42)
    draws1 = torch.rand(10, generator=g1)
    draws2 = torch.rand(10, generator=g2)
    assert torch.equal(draws1, draws2)


def test_different_seeds_produce_different_draws():
    g1 = torch.Generator(device="cpu").manual_seed(1)
    g2 = torch.Generator(device="cpu").manual_seed(2)
    draws1 = torch.rand(10, generator=g1)
    draws2 = torch.rand(10, generator=g2)
    assert not torch.equal(draws1, draws2)


def test_generator_is_isolated_from_global_rng_state():
    """The whole point of using a per-call Generator instead of
    torch.manual_seed() globally: unrelated global RNG activity between
    two seeded calls must not perturb their reproducibility."""
    g1 = torch.Generator(device="cpu").manual_seed(7)
    first = torch.rand(5, generator=g1)

    # Perturb global RNG state in between - must not matter.
    torch.rand(1000)

    g2 = torch.Generator(device="cpu").manual_seed(7)
    second = torch.rand(5, generator=g2)
    assert torch.equal(first, second)
