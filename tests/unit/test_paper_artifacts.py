"""Paper artifacts are generated from tables; the human-audit figure/table work once audit tables exist."""

import sys
from pathlib import Path

import pytest

pd = pytest.importorskip("pandas")
pytest.importorskip("matplotlib")
pytest.importorskip("tabulate")
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

import make_paper_artifacts as mpa


def test_rounding_is_single_half_up_from_stored_value():
    assert mpa.p2(0.485) == "0.49"  # stored 4-dp value of 0.48501; float formatting would give 0.48
    assert mpa.f2(0.2346) == "+0.23"  # no double rounding (0.2346 -> 0.235 -> 0.24)
    assert mpa.p2(0.1449) == "0.14"
    assert mpa.f2(-0.081) == "-0.08"


def test_audit_table_and_figure_from_synthetic_metrics(tmp_path, monkeypatch):
    import matplotlib

    matplotlib.use("Agg")
    monkeypatch.setattr(mpa, "T", tmp_path)
    monkeypatch.setattr(mpa, "OUT_T", tmp_path / "t")
    monkeypatch.setattr(mpa, "OUT_F", tmp_path / "f")
    rows = []
    for slice_, value, n, hk, jk in (("all", "all", 100, 38, 40), ("sf_pole", "left_pole", 85, 36, 39),
                                     ("sf_pole", "right_pole", 12, 2, 1), ("sf_pole", "non_directional", 3, 0, 0)):
        rows.append({"reference": "consensus", "slice": slice_, "value": value, "n": n, "human_commit_k": hk,
                     "human_commit_rate": hk / n, "jqy_commit_k": jk, "jqy_commit_rate": jk / n, "accuracy": 0.83,
                     "cohen_kappa": 0.64, "overread_k": 5, "overread_n": n - hk, "overread_ci_low": 0.03,
                     "overread_ci_high": 0.17, "underread_k": 3, "underread_n": hk, "underread_ci_low": 0.03,
                     "underread_ci_high": 0.2, "reversals": 0, "both_directional_n": 35})
    for audit in ("olmo", "qwen"):
        pd.DataFrame(rows).to_csv(tmp_path / f"commitment_audit_{audit}_metrics.csv", index=False)
    assert mpa.table2() is True
    assert "pending" not in (tmp_path / "t" / "table2_human_audit.md").read_text().lower().split("status")[0]
    assert mpa.figure_audit_by_pole() is True
    assert (tmp_path / "f" / "figure3_audit_commitment_by_pole.pdf").exists()
