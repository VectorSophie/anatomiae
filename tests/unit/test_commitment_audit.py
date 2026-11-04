"""Synthetic checks of the commitment-audit tooling (no real labels or keys involved)."""

import math
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


# ---------------------------------------------------------------- import, consensus, key isolation


def _audit_dir(tmp_path, monkeypatch, n=4):
    """A synthetic audit 'syn' with blank frozen sheets in tmp_path; returns (root, completed A, completed C)."""
    root = tmp_path / "syn"
    root.mkdir()
    monkeypatch.setitem(ca.AUDITS, "syn", (str(root), n, "CA"))
    blank = sheet([""] * n)
    blank["mode"], blank["confidence_1to3"] = "", ""
    blank["statement"] = [f"s{i}" for i in range(n)]
    blank["response"] = [f"r, \"{i}\"\nmore" for i in range(n)]
    order_a, order_c = blank.sample(frac=1, random_state=1), blank.sample(frac=1, random_state=2)
    order_a.to_csv(root / "sheet_A.csv", index=False)
    order_c.to_csv(root / "sheet_C.csv", index=False)
    rel_a = {"CA001": "supports", "CA002": "opposes", "CA003": "mixed_or_conditional", "CA004": "supports"}
    rel_c = {"CA001": "supports", "CA002": "neutral_or_no_position", "CA003": "mixed_or_conditional", "CA004": "supports"}
    mode_c = {"CA004": "analytical_exposition"}
    done = {}
    for who, frozen, rel in (("A", order_a, rel_a), ("C", order_c, rel_c)):
        d = frozen.copy()
        d["relation"] = d["sample_id"].map(rel)
        d["mode"] = [mode_c.get(s, "explicit_stance") if who == "C" else "explicit_stance" for s in d["sample_id"]]
        d["confidence_1to3"] = "2"
        p = tmp_path / f"syn_sheet_{who}.completed.csv"
        d.to_csv(p, index=False)
        done[who] = p
    return root, done["A"], done["C"]


def test_import_is_non_destructive_and_never_replaces(tmp_path, monkeypatch):
    root, done_a, _ = _audit_dir(tmp_path, monkeypatch)
    frozen_before = (root / "sheet_A.csv").read_bytes()
    ca.cmd_import_sheet("syn", "A", done_a)
    assert (root / "sheet_A.csv").read_bytes() == frozen_before  # frozen blank sheet untouched
    assert (root / "sheet_A.completed.csv").read_bytes() == done_a.read_bytes()  # byte-identical copy
    ca.cmd_import_sheet("syn", "A", done_a)  # identical re-import is a no-op
    other = tmp_path / "other.csv"
    d = ca.read_csv(done_a)
    d.loc[0, "relation"] = "unclear"
    d.to_csv(other, index=False)
    with pytest.raises(SystemExit, match="never replaced"):
        ca.cmd_import_sheet("syn", "A", other)
    assert (root / "sheet_A.completed.csv").read_bytes() == done_a.read_bytes()
    with pytest.raises(SystemExit, match="not imported"):  # A's labels against C's frozen order
        ca.cmd_import_sheet("syn", "C", done_a)
    imports = __import__("json").loads((root / "imports.json").read_text())
    assert [e["artifact"] for e in imports] == ["sheet_A"]


def test_check_and_consensus_never_open_the_key(tmp_path, monkeypatch):
    root, done_a, done_c = _audit_dir(tmp_path, monkeypatch)
    (root / "key.csv").write_text("poison")
    monkeypatch.setattr(ca, "OUT", str(tmp_path / "tables" / "syn"))
    ca.cmd_import_sheet("syn", "A", done_a)
    ca.cmd_import_sheet("syn", "C", done_c)
    real_read, real_open = pd.read_csv, open

    def guard_read(path, *a, **k):
        assert "key.csv" not in str(path), "key opened during a blind step"
        return real_read(path, *a, **k)

    def guard_open(path, *a, **k):
        assert "key.csv" not in str(path), "key opened during a blind step"
        return real_open(path, *a, **k)

    monkeypatch.setattr(pd, "read_csv", guard_read)
    monkeypatch.setattr("builtins.open", guard_open)
    ca.cmd_check("syn")
    q = ca.read_csv(root / "adjudication_queue.csv")
    assert list(q["sample_id"]) == ["CA002", "CA004"]  # relation differs; mode differs
    q.loc[q["sample_id"] == "CA002", ["relation_adjudicated", "mode_adjudicated"]] = ["opposes", "explicit_stance"]
    q.loc[q["sample_id"] == "CA004", "mode_adjudicated"] = "analytical_exposition"  # relation agreed: may stay blank
    src = tmp_path / "adjudication_queue_syn.completed.csv"
    q.to_csv(src, index=False)
    ca.cmd_import_adjudication("syn", src)
    ca.cmd_consensus("syn")
    cons = ca.read_csv(root / "labels_consensus.csv").set_index("sample_id")
    assert cons.at["CA002", "relation"] == "opposes" and cons.at["CA002", "provenance"] == "researcher_adjudicated"
    assert cons.at["CA004", "relation"] == "supports" and cons.at["CA004", "mode"] == "analytical_exposition"
    assert cons.at["CA001", "provenance"] == "human_A_C_agreed"
    # the independent A and C imports are untouched by consensus
    assert (root / "sheet_A.completed.csv").read_bytes() == done_a.read_bytes()
    assert (root / "sheet_C.completed.csv").read_bytes() == done_c.read_bytes()
    with pytest.raises(SystemExit, match="never rebuilt"):
        ca.cmd_consensus("syn")


def test_adjudication_must_resolve_every_disputed_field():
    q = pd.DataFrame({"sample_id": ["CA001"], "relation_A": ["supports"], "relation_C": ["opposes"],
                      "mode_A": ["explicit_stance"], "mode_C": ["explicit_stance"], "commitment_differs": ["False"],
                      "relation_adjudicated": [""], "mode_adjudicated": [""], "adjudication_notes": [""]})
    with pytest.raises(ValueError, match="incomplete"):
        ca.resolve_adjudication(q.set_index("sample_id"))
    q["relation_adjudicated"] = "opposes"
    assert ca.resolve_adjudication(q.set_index("sample_id")).at["CA001", "mode"] == "explicit_stance"
    expected = q.copy()
    assert ca.verify_queue(expected, q.assign(relation_A="mixed_or_conditional"))  # tampered A column
    assert ca.verify_queue(expected, q.iloc[0:0])  # wrong id set
    assert ca.verify_queue(expected, q) == []


def test_consensus_refuses_unqueued_disagreement():
    a = sheet(["supports", "opposes"]).set_index("sample_id")
    c = sheet(["supports", "supports"]).set_index("sample_id")
    q = ca.adjudication_queue(a, c).iloc[0:0]
    with pytest.raises(SystemExit, match="not in the adjudication queue"):
        ca.build_consensus(a, c, q)


# ---------------------------------------------------------------- secondary statistics


def test_fisher_and_newcombe_reference_values():
    # scipy.stats.fisher_exact([[1, 9], [6, 4]]) two-sided p = 0.0573
    assert ca.fisher_exact(1, 10, 6, 10) == pytest.approx(0.05728, abs=1e-4)
    assert ca.fisher_exact(3, 10, 3, 10) == pytest.approx(1.0)
    # Newcombe (1998) example: 56/70 vs 48/80 -> 0.0524 to 0.3339
    lo, hi = ca.newcombe(56, 70, 48, 80)
    assert lo == pytest.approx(0.0524, abs=1e-3) and hi == pytest.approx(0.3339, abs=1e-3)
    assert math.isnan(ca.fisher_exact(0, 0, 1, 2))


def _keyed(n_items=12):
    rows = []
    for i in range(n_items):
        for j, pole_ in enumerate(("left_pole", "left_pole", "right_pole")):
            rows.append({"item_id": f"it{i}", "sf_pole": pole_, "C_jqy": int(pole_ == "left_pole" and (i + j) % 2 == 0),
                         "C_human": int(pole_ == "left_pole" and i % 3 == 0 or pole_ == "right_pole" and i % 4 == 0)})
    return pd.DataFrame(rows)


def test_pole_contrasts_counts_and_bootstrap_determinism():
    d = _keyed()
    rows = {r["quantity"]: r for r in ca.pole_contrasts(d)}
    o = rows["overread_left_minus_right"]
    left_neg = d[(d["sf_pole"] == "left_pole") & (d["C_human"] == 0)]
    assert o["left_n"] == len(left_neg) and o["left_k"] == int(left_neg["C_jqy"].sum())
    assert o["right_k"] == 0
    assert o["diff"] == pytest.approx(o["left_rate"] - o["right_rate"])
    assert ca.pole_contrasts(d) == ca.pole_contrasts(d)  # fixed-seed bootstrap
    h = rows["human_commit_left_minus_right"]
    assert h["left_n"] + h["right_n"] == len(d)


def test_measurement_effect_point_estimates():
    d = _keyed()
    m = {r["quantity"]: r["estimate"] for r in ca.measurement_effect(d)}
    assert m["human_minus_jqy_commit_rate"] == pytest.approx(d["C_human"].mean() - d["C_jqy"].mean())
    L, R = d[d["sf_pole"] == "left_pole"], d[d["sf_pole"] == "right_pole"]
    ah, aj = L["C_human"].mean() - R["C_human"].mean(), L["C_jqy"].mean() - R["C_jqy"].mean()
    assert m["jqy_minus_human_asymmetry"] == pytest.approx(aj - ah)


def test_cluster_boot_resamples_clusters_not_rows():
    d = pd.DataFrame({"item_id": ["a"] * 50 + ["b"] * 50, "x": [1] * 50 + [0] * 50})
    r = ca.cluster_boot(d, {"x": d["x"] == 1, "n": pd.Series(True, index=d.index)}, {"m": lambda s: s["x"] / s["n"]})
    est, lo, hi = r["m"]
    assert est == pytest.approx(0.5) and lo == pytest.approx(0.0) and hi == pytest.approx(1.0)


def test_analyze_end_to_end_on_synthetic_audit(tmp_path, monkeypatch):
    pytest.importorskip("tabulate")
    root, done_a, done_c = _audit_dir(tmp_path, monkeypatch)
    monkeypatch.setattr(ca, "OUT", str(tmp_path / "tables" / "syn"))
    ca.cmd_import_sheet("syn", "A", done_a)
    ca.cmd_import_sheet("syn", "C", done_c)
    ca.cmd_check("syn")
    q = ca.read_csv(root / "adjudication_queue.csv")
    q["relation_adjudicated"] = ["opposes", "supports"]
    q["mode_adjudicated"] = "explicit_stance"
    q.to_csv(root / "adjudication_queue.completed.csv", index=False)
    ca.cmd_consensus("syn")
    pd.DataFrame({"sample_id": ["CA001", "CA002", "CA003", "CA004"], "item_id": ["i1", "i1", "i2", "i3"],
                  "prompt": "p", "jqy_outcome": ["agreement", "disagreement", "agreement", "mixed_or_conditional"],
                  "C_jqy": ["True", "True", "True", "False"],
                  "sf_pole": ["left_pole", "left_pole", "right_pole", "right_pole"]}).to_csv(root / "key.csv", index=False)
    ca.cmd_analyze("syn")
    m = pd.read_csv(tmp_path / "tables" / "syn_syn_metrics.csv")
    allc = m[(m["reference"] == "consensus") & (m["slice"] == "all")].iloc[0]
    assert allc["n"] == 4 and allc["human_commit_k"] == 3 and allc["jqy_commit_k"] == 3
    assert allc["overread_k"] == 1 and allc["underread_k"] == 1  # CA003 over-read, CA004 under-read
    assert set(m["reference"]) == {"consensus", "human_A", "human_C"}
    assert (tmp_path / "tables" / "syn_syn_report.md").exists()


def test_cli_accepts_documented_argument_order(tmp_path, monkeypatch):
    """Regression: `import-sheet --audit X --annotator A <path>` (path after the options) must parse."""
    root, done_a, _ = _audit_dir(tmp_path, monkeypatch)
    ca.main(["import-sheet", "--audit", "syn", "--annotator", "A", str(done_a)])
    assert (root / "sheet_A.completed.csv").read_bytes() == done_a.read_bytes()
    with pytest.raises(SystemExit):
        ca.main(["import-sheet", "--audit", "syn", str(done_a)])  # missing --annotator
