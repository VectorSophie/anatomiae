"""Synthetic checks of the frozen selective-commitment analysis (sign conventions, pooling)."""

import sys
from pathlib import Path

import pytest

pd = pytest.importorskip("pandas")
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

import selective_commitment_v1 as sc


def synthetic(commit_on_aligned: bool, n_items: int = 40) -> "pd.DataFrame":
    """Stance-first: half the items lean-aligned (left agrees / right disagrees), half opposed.
    Free-form commits under all 5 prompts on aligned (or opposed) item-polarities only."""
    rows = []
    for i in range(n_items):
        aligned = i % 2 == 0
        for pol, lean in (("orig", "left"), ("inv", "right")):
            agree = (lean == "left") == aligned
            sf = "agreement" if agree else "disagreement"
            rows.append({"evaluator": "jqy", "stage": "1_sft", "item_id": f"it{i}", "polarity": pol, "prompt": "stance_first",
                             "item_lean": lean, "outcome": sf})
            commits = aligned if commit_on_aligned else not aligned
            for p in sc.FREE_FORM:
                rows.append({"evaluator": "jqy", "stage": "1_sft", "item_id": f"it{i}", "polarity": pol, "prompt": p,
                                 "item_lean": lean, "outcome": sf if commits else "mixed_or_conditional"})
    return pd.DataFrame(rows)


@pytest.mark.parametrize(("aligned", "sign"), [(True, 1), (False, -1)])
def test_gap_sign(aligned, sign):
    g = sc.gap(sc.units(synthetic(aligned), "1_sft"))
    assert g["selection_gap"] == pytest.approx(2.0 * sign)  # +1 vs -1 direction index
    assert g["estimable"]
    assert g["n_units"] == 40 * 2 * 5
    assert g["n_committed_units"] == 20 * 2 * 5


def test_units_pool_all_free_form_prompts_only():
    u = sc.units(synthetic(True), "1_sft")
    assert set(u["prompt"]) == set(sc.FREE_FORM)
    assert "stance_first" not in set(u["prompt"])


def test_prompts_frozen_and_new():
    assert len(sc.FREE_FORM) == 5
    for text in sc.FREE_FORM.values():
        assert text not in sc.FAULBORN_PREFIXES.values()
        assert "agree" not in text.lower()
