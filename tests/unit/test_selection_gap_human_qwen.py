"""Qwen human selection-gap re-estimation: sampling design, label reuse and the weighted estimator."""

import json
import sys
from pathlib import Path

import pytest

pd = pytest.importorskip("pandas")
np = pytest.importorskip("numpy")
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))

import selection_gap_human_qwen_v1 as g

PROMPTS = ["ff_assessment", "ff_comment", "ff_make_of", "ff_thoughts", "ff_where_stand"]


def synthetic_units(n_items=40, seed=0):
    """Item-polarity clusters with 5 prompts each; stratum constant within cluster."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_items):
        for pol, lean in (("orig", "left"), ("inv", "right")):
            stratum = rng.choice(list(g.STRATA), p=[0.75, 0.2, 0.05])
            sf = ("non" if stratum == "non_directional" else
                  ("agreement" if lean == "left" else "disagreement") if stratum == "left_pole" else
                  ("disagreement" if lean == "left" else "agreement"))
            for p in PROMPTS:
                rows.append({"cache_key": f"k{i}{pol}{p}", "item_id": f"it{i:02d}", "polarity": pol, "prompt": p,
                             "item_lean": lean, "sf_outcome": sf, "sf_outcome_det": sf, "stratum": stratum,
                             "C_jqy": int(rng.random() < (0.45 if stratum == "left_pole" else 0.05)),
                             "statement": f"s{i}{pol}", "response": f"r{i}{pol}{p}"})
    u = pd.DataFrame(rows)
    u["C_h"] = u["C_jqy"]
    return u


def test_draw_is_deterministic_complete_and_stratified():
    u = synthetic_units()
    cl = g.clusters(u)
    a, b = g.draw(cl, seed=5, left_n=10), g.draw(cl, seed=5, left_n=10)
    pd.testing.assert_frame_equal(a, b)
    assert not g.draw(cl, seed=6, left_n=10).equals(a)
    counts = a["stratum"].value_counts()
    assert counts["left_pole"] == 10
    for h in ("right_pole", "non_directional"):  # census strata
        assert counts.get(h, 0) == (cl["stratum"] == h).sum()
    # whole clusters only: every sampled item-polarity carries all five prompts
    s = u.merge(a[["item_id", "polarity"]], on=["item_id", "polarity"])
    assert s.groupby(["item_id", "polarity"]).size().eq(5).all()


def test_reuse_matches_generation_identity_not_text():
    u = synthetic_units()
    cl = g.draw(g.clusters(u), seed=g.SEED, left_n=10)
    picked = u.merge(cl[["item_id", "polarity"]], on=["item_id", "polarity"])
    k_reused = picked["cache_key"].iloc[0]
    other = u[~u["cache_key"].isin(picked["cache_key"])].iloc[0]
    # an audit row with the same cache_key is reused; one with identical text but another key is not
    audit = pd.DataFrame({"sample_id": ["CQ001", "CQ002"], "cache_key": [k_reused, other["cache_key"]]})
    u2 = u.copy()
    u2.loc[u2["cache_key"] == picked["cache_key"].iloc[1], "response"] = other["response"]
    key, blind = _build(u2, audit, left_n=10)
    reused = key[key["label_source"] == g.LABEL_SOURCE_AUDIT]
    assert list(reused["cache_key"]) == [k_reused] and list(reused["label_sample_id"]) == ["CQ001"]
    assert len(blind) == len(key) - 1
    assert set(blind["sample_id"]) == set(key.loc[key["label_source"] == g.LABEL_SOURCE_NEW, "label_sample_id"])
    assert (blind[["relation", "mode", "confidence_1to3", "notes"]] == "").all().all()  # never prefilled


def _build(units, audit, left_n):
    orig = g.LEFT_N
    try:
        g.LEFT_N = left_n
        g.draw.__defaults__ = (g.SEED, left_n)
        return g.build_sample(units, audit)
    finally:
        g.LEFT_N = orig
        g.draw.__defaults__ = (g.SEED, orig)


def test_sheets_have_independent_orders_of_the_same_rows():
    u = synthetic_units()
    _, blind = _build(u, pd.DataFrame({"sample_id": [], "cache_key": []}), left_n=10)
    sh = g.sheets(blind)
    assert sorted(sh["A"]["sample_id"]) == sorted(sh["C"]["sample_id"]) == sorted(blind["sample_id"])
    assert list(sh["A"]["sample_id"]) != list(sh["C"]["sample_id"])


def test_weighted_estimator_recovers_population_when_strata_are_homogeneous():
    """If every left-pole cluster were identical, any left subsample with weight N/n gives the census value."""
    u = synthetic_units()
    left = u["stratum"] == "left_pole"
    u.loc[left, "C_h"] = (u.loc[left, "prompt"].isin(PROMPTS[:2])).astype(int)
    cl = g.clusters(u)
    census = g.estimate(cl, u, {"h": "C_h"}).iloc[0]
    s = u.merge(g.draw(cl, seed=3, left_n=8)[["item_id", "polarity"]], on=["item_id", "polarity"])
    w = g.estimate(cl, s, {"h": "C_h"}).iloc[0]
    assert w["commit_rate_weighted"] == pytest.approx(census["commit_rate_weighted"])
    # the raw (unweighted) sample proportion is not the population value under this design
    assert s["C_h"].mean() != pytest.approx(census["commit_rate_weighted"])


def test_paired_difference_and_bootstrap_determinism():
    u = synthetic_units(n_items=60)
    cl = g.clusters(u)
    s = u.merge(g.draw(cl, seed=1, left_n=20)[["item_id", "polarity"]], on=["item_id", "polarity"])
    s["C_h"] = s["C_jqy"]
    t1 = g.estimate(cl, s, {"h": "C_h", "j": "C_jqy"})
    t2 = g.estimate(cl, s, {"h": "C_h", "j": "C_jqy"})
    pd.testing.assert_frame_equal(t1, t2)
    diff = t1[t1["label_source"] == "h_minus_j"].iloc[0]
    assert diff["selection_gap"] == pytest.approx(0.0)
    assert diff["ci95_low"] == pytest.approx(0.0) and diff["ci95_high"] == pytest.approx(0.0)


def test_unrepresented_stratum_makes_replicate_undefined():
    N = np.array([[[2.0, 1.0, 0.0]]])
    n = np.array([[[1.0, 0.0, 0.0]]])
    S = np.ones((1, 1, 3, 2))
    assert np.isnan(g.weighted_sums(N, n, S)).all()


# ---------------------------------------------------------------- the real frozen run (needs the public cache)


@pytest.fixture(scope="module")
def real_units():
    try:
        return g.population()
    except (ImportError, OSError) as e:  # pragma: no cover - environment without data/extras
        pytest.skip(f"population unavailable: {e}")


def test_population_counts_and_census_reproduces_frozen_gap(real_units):
    import selective_commitment_v1 as sc

    cl = g.clusters(real_units)
    assert len(real_units) == 460 and len(cl) == 92
    assert cl["stratum"].value_counts().to_dict() == g.EXPECTED_POP
    census = g.estimate(cl, real_units, {"jqy": "C_jqy"}).iloc[0]
    df = sc.frame()  # bound to the xl artifacts by g.population()
    u = sc.units(df, "q_inst")
    frozen = sc.gap(u[u["item_id"].isin(sc.CLASS_A)])
    for k in ("selection_gap", "ci95_low", "ci95_high", "D_SF_committed", "D_SF_uncommitted"):
        assert census[k] == pytest.approx(frozen[k], abs=1e-12), k


def test_committed_sample_is_reproducible_and_matches_manifest(real_units):
    key, blind = g.build_sample(real_units, g.audit_key())
    assert len(key) == 165 and key.groupby(["item_id", "polarity"]).size().eq(5).all()
    strata = key[["item_id", "polarity", "stratum"]].drop_duplicates()["stratum"].value_counts().to_dict()
    assert strata == {"left_pole": 20, "right_pole": 10, "non_directional": 3}
    manifest = json.loads((REPO / g.MANIFEST).read_text())
    assert manifest["total_rows"] == 165
    assert manifest["rows_requiring_new_A_label"] == manifest["rows_requiring_new_C_label"] == len(blind)
    assert manifest["rows_with_reused_A_label"] == len(key) - len(blind)
    for a, sh in g.sheets(blind).items():
        committed = pd.read_csv(REPO / g.ROOT / f"sheet_{a}.csv", keep_default_na=False, dtype=str)
        pd.testing.assert_frame_equal(sh[g.COLUMNS].astype(str).reset_index(drop=True), committed)
    g.check_reuse_text(key, real_units)


def test_analyze_pipeline_runs_once_followup_labels_exist(real_units, tmp_path, monkeypatch):
    """Readiness check with SYNTHETIC follow-up labels in a temp dir (never written to the repo).
    Needs the local key and the imported Qwen audit consensus; skipped otherwise."""
    import analyze_commitment_audit as ca

    if not (REPO / g.KEY).exists() or not (REPO / g.AUDIT_ROOT / "labels_consensus.csv").exists():
        pytest.skip("local key or Qwen audit consensus not present")
    pytest.importorskip("tabulate")
    root = tmp_path / "qwen_gap"
    root.mkdir()
    for a in "AC":
        sh = pd.read_csv(REPO / g.ROOT / f"sheet_{a}.csv", keep_default_na=False, dtype=str)
        sh["relation"], sh["mode"], sh["confidence_1to3"] = "neutral_or_no_position", "analytical_exposition", "2"
        sh.to_csv(root / f"sheet_{a}.completed.csv", index=False)
        sh.to_csv(root / f"sheet_{a}.csv", index=False)
    pd.DataFrame({"sample_id": sh["sample_id"], "relation": sh["relation"], "mode": sh["mode"],
                  "provenance": "human_A_C_agreed"}).to_csv(root / "labels_consensus.csv", index=False)
    monkeypatch.setitem(ca.AUDITS, "qwen_gap", (str(root), 142, "QG"))
    monkeypatch.chdir(REPO)
    monkeypatch.setattr(g, "OUT", str(tmp_path / "out" / "sg"))
    g.cmd_analyze()
    prim = pd.read_csv(tmp_path / "out" / "sg_primary.csv").set_index("label_source")
    # the J-qy comparator on the same units reproduces the label-free design check in the protocol
    assert prim.at["jqy_same_units", "selection_gap"] == pytest.approx(0.2619, abs=1e-3)
    census = pd.read_csv(tmp_path / "out" / "sg_design_check_census.csv").iloc[0]
    assert census["selection_gap"] == pytest.approx(0.3158, abs=1e-3)
    assert "human_consensus_minus_jqy_same_units" in prim.index
    assert not (REPO / g.ROOT / "labels_consensus.csv").exists()  # nothing written to the real follow-up dir
