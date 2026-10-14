"""Synthetic checks of the commitment-audit tooling (no real labels or keys involved)."""

import sys
from pathlib import Path

import pytest

pd = pytest.importorskip("pandas")
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

import analyze_commitment_audit as ca


def sheet(relations, modes=None, prefix="CA"):
    n = len(relations)
    return pd.DataFrame({"sample_id": [f"{prefix}{i:03d}" for i in range(1, n + 1)], "statement": "s",
                         "response": "r", "relation": relations, "mode": modes or ["explicit_stance"] * n,
                         "confidence_1to3": "3", "notes": ""})


def test_validate_flags_missing_and_bad_values():
    s = sheet(["supports", "", "maybe"])
    problems = ca.validate(s, 3, "CA", "A")
    assert any("missing relation" in p for p in problems)
    assert any("outside the taxonomy" in p for p in problems)
    assert ca.validate(sheet(["supports", "opposes", "unclear"]), 3, "CA", "A") == []
    assert ca.validate(sheet(["supports"]), 3, "CA", "A")  # wrong n


def test_agreement_and_queue():
    a = sheet(["supports", "opposes", "mixed_or_conditional", "supports"]).set_index("sample_id")
    c = sheet(["supports", "neutral_or_no_position", "mixed_or_conditional", "opposes"]).set_index("sample_id")
    t = ca.agreement_table(a, c).set_index("dimension")
    assert t.at["relation", "exact_agreement"] == pytest.approx(0.5)
    assert t.at["commitment_binary", "exact_agreement"] == pytest.approx(0.75)
    assert t.at["direction_when_both_directional", "n"] == 2
    q = ca.adjudication_queue(a, c)
    assert list(q["sample_id"]) == ["CA002", "CA004"]
    assert list(q["commitment_differs"]) == [True, False]


def test_consensus_uses_adjudication_only_for_disagreements():
    a = sheet(["supports", "opposes"]).set_index("sample_id")
    c = sheet(["supports", "neutral_or_no_position"]).set_index("sample_id")
    q = ca.adjudication_queue(a, c)
    q["relation_adjudicated"] = "mixed_or_conditional"
    q["mode_adjudicated"] = "analytical_exposition"
    cons = ca.build_consensus(a, c, q).set_index("sample_id")
    assert cons.at["CA001", "provenance"] == "human_A_C_agreed"
    assert cons.at["CA002", "relation"] == "mixed_or_conditional"


def test_commitment_metrics():
    c_jqy = pd.Series([1, 1, 0, 0, 1])
    c_h = pd.Series([1, 0, 1, 0, 1])
    h_rel = pd.Series(["supports", "mixed_or_conditional", "opposes", "neutral_or_no_position", "opposes"])
    jqy = pd.Series(["agreement", "agreement", "neutral_or_no_position", "neutral_or_no_position", "agreement"])
    m = ca.commitment_metrics(c_jqy, c_h, h_rel, jqy)
    assert m["overread_k"] == 1 and m["overread_n"] == 2
    assert m["underread_k"] == 1 and m["underread_n"] == 3
    assert m["reversals"] == 1  # last row: judge agreement vs human opposes
    assert m["accuracy"] == pytest.approx(0.6)
