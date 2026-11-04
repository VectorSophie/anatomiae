"""Qwen2.5-14B-Instruct selection gap re-estimated with human-labelled commitment.

Protocol (frozen before any new label): docs/experiments/selective_commitment_human_reestimate_qwen_v1.md.
No generation, no judging: only the cached Qwen2.5-14B-Instruct responses and the
J-qy / rule-based outcomes of the cross-lineage run.

  uv run python scripts/selection_gap_human_qwen_v1.py sample    # once; never overwrites the key
  uv run python scripts/selection_gap_human_qwen_v1.py analyze   # after both human label sources exist

Estimand: the frozen primary selection gap of selective_commitment_crosslineage_v1
(class A, 5 free-form prompts pooled, unit-weighted, J-qy stance-first) with exactly one
change: the commitment indicator C comes from human consensus instead of J-qy.

Design: item-polarity clusters (all 5 prompts each) stratified on the J-qy stance-first
pole: every right-pole and non-directional cluster, and LEFT_N left-pole clusters drawn
uniformly with SEED. The estimator is the design-weighted (Hajek) version of the frozen
ratio-of-sums gap, stratum weight N_h / n_h. The interval is the frozen item bootstrap
(46 class-A items, orig and inv together, 2,000 resamples, default_rng(0)), run as a
two-phase bootstrap: each replicate recomputes N_h*/n_h* from the resampled items. With a
census sample it reduces exactly to the original procedure (tested).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

SEED = 20261103
LEFT_N = 20
STRATA = ("left_pole", "right_pole", "non_directional")
EXPECTED_POP = {"left_pole": 79, "right_pole": 10, "non_directional": 3}
ROOT = Path("artifacts/labeling/selection_gap_human_qwen_v1")
KEY = ROOT / "key.csv"
MANIFEST = ROOT / "manifest.json"
AUDIT_ROOT = Path("artifacts/labeling/commitment_audit_qwen_v1")
LABEL_SOURCE_AUDIT = "reused_from_commitment_audit_qwen_v1"
LABEL_SOURCE_NEW = "new_selection_gap_human_qwen_v1"
COLUMNS = ["sample_id", "statement", "response", "relation", "mode", "confidence_1to3", "notes"]
OUT = "artifacts/tables/selection_gap_human_qwen_v1"
N_BOOT = 2000
MIN_DIRECTIONAL_UNITS = 10
HUMAN_DIRECTIONAL = {"supports", "opposes"}
JQY_DIRECTIONAL = {"agreement", "disagreement"}


def pole(sf_outcome: str, lean: str) -> str:
    if sf_outcome not in JQY_DIRECTIONAL:
        return "non_directional"
    return "left_pole" if (sf_outcome == "agreement") == (lean == "left") else "right_pole"


# ---------------------------------------------------------------- population (frozen run, read-only)


def population() -> pd.DataFrame:
    """The 460 class-A Qwen2.5-14B-Instruct free-form units of the primary replication, with J-qy C,
    J-qy and rule-based stance-first outcomes, the stratum, and the judged statement/response text."""
    import selective_commitment_v1 as sc
    import selective_commitment_xl_v1 as xl

    from anatomiae.datasets.faulborn import load_faulborn_items
    from anatomiae.provenance.generation_cache import GenerationCache

    xl.bind_v1()
    df = sc.frame()
    u = sc.units(df, "q_inst")
    u = u[u["item_id"].isin(sc.CLASS_A)].copy()
    det = sc.units(df, "q_inst", sf_evaluator=sc.DET).set_index("cache_key")["sf_outcome"]
    u["sf_outcome_det"] = u["cache_key"].map(det)
    u["stratum"] = [pole(o, le) for o, le in zip(u["sf_outcome"], u["item_lean"], strict=True)]
    u["C_jqy"] = u["C"].astype(int)
    items = {i.item_id: i for i in load_faulborn_items()}
    recs = {r.cache_key: r for r in GenerationCache(xl.CACHE).read_all()}
    u["response"] = [xl.answer_text(recs[k]) for k in u["cache_key"]]
    u["statement"] = [sc.statement(items[i], p) for i, p in zip(u["item_id"], u["polarity"], strict=True)]
    u = u.rename(columns={"outcome": "jqy_outcome"})
    cols = ["cache_key", "item_id", "polarity", "prompt", "item_lean", "jqy_outcome", "C_jqy", "sf_outcome",
            "sf_outcome_det", "stratum", "statement", "response"]
    return u[cols].sort_values(["item_id", "polarity", "prompt"]).reset_index(drop=True)


def clusters(units: pd.DataFrame) -> pd.DataFrame:
    k = units.groupby(["item_id", "polarity"]).agg(stratum=("stratum", "first"), n_units=("prompt", "size"),
                                                   n_strata=("stratum", "nunique")).reset_index()
    assert (k["n_strata"] == 1).all(), "stratum must be constant within an item-polarity"
    return k.drop(columns="n_strata")


def draw(cl: pd.DataFrame, seed: int = SEED, left_n: int = LEFT_N) -> pd.DataFrame:
    """Frozen cluster draw: census of right-pole and non-directional clusters, uniform left-pole draw."""
    cl = cl.sort_values(["item_id", "polarity"]).reset_index(drop=True)
    left = cl[cl["stratum"] == "left_pole"].sample(left_n, random_state=seed)
    rest = cl[cl["stratum"] != "left_pole"]
    return pd.concat([left, rest]).sort_values(["item_id", "polarity"]).reset_index(drop=True)


# ---------------------------------------------------------------- sample


def audit_key() -> pd.DataFrame:
    return pd.read_csv(AUDIT_ROOT / "key.csv", dtype=str)


def build_sample(units: pd.DataFrame, audit: pd.DataFrame, seed: int = SEED) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (key, blind new rows). Reuse is matched on the immutable generation cache_key."""
    chosen = draw(clusters(units), seed)
    s = units.merge(chosen[["item_id", "polarity"]], on=["item_id", "polarity"])
    reuse = audit.set_index("cache_key")["sample_id"]
    assert reuse.index.is_unique
    s["label_source"] = np.where(s["cache_key"].isin(reuse.index), LABEL_SOURCE_AUDIT, LABEL_SOURCE_NEW)
    s["label_sample_id"] = s["cache_key"].map(reuse).astype(object)
    new = s[s["label_source"] == LABEL_SOURCE_NEW].sort_values("cache_key").sample(frac=1, random_state=seed + 1)
    ids = {k: f"QG{n:03d}" for n, k in enumerate(new["cache_key"], start=1)}
    s.loc[s["label_source"] == LABEL_SOURCE_NEW, "label_sample_id"] = s["cache_key"].map(ids)
    blind = pd.DataFrame({"sample_id": [ids[k] for k in new["cache_key"]], "statement": new["statement"].to_numpy(),
                          "response": new["response"].to_numpy(), "relation": "", "mode": "",
                          "confidence_1to3": "", "notes": ""})
    return s, blind


def sheets(blind: pd.DataFrame, seed: int = SEED) -> dict[str, pd.DataFrame]:
    return {a: blind.sample(frac=1, random_state=seed + off).reset_index(drop=True)
            for a, off in (("A", 11), ("C", 33))}


def check_reuse_text(key: pd.DataFrame, units: pd.DataFrame) -> None:
    """Reused rows were matched on cache_key; their audit sheet text must equal the cached text exactly."""
    audit = pd.read_csv(AUDIT_ROOT / "sheet_A.csv", keep_default_na=False, dtype=str).set_index("sample_id")
    text = units.set_index("cache_key")
    for _, r in key[key["label_source"] == LABEL_SOURCE_AUDIT].iterrows():
        a = audit.loc[r["label_sample_id"]]
        assert (a["statement"], a["response"]) == (text.at[r["cache_key"], "statement"],
                                                  text.at[r["cache_key"], "response"]), r["cache_key"]


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cmd_sample() -> None:
    if KEY.exists():
        raise SystemExit(f"{KEY} exists; a frozen sample is never regenerated in place")
    units = population()
    pop = clusters(units)["stratum"].value_counts().to_dict()
    print("class-A item-polarities by stance-first pole:", pop)
    if pop != EXPECTED_POP:
        print(f"NOTE: population differs from the protocol's expectation {EXPECTED_POP}; sampling principle kept")
    key, blind = build_sample(units, audit_key())
    check_reuse_text(key, units)
    assert key.groupby(["item_id", "polarity"]).size().eq(5).all(), "partial prompt cluster"
    ROOT.mkdir(parents=True, exist_ok=True)
    for a, sh in sheets(blind).items():
        sh[COLUMNS].to_csv(ROOT / f"sheet_{a}.csv", index=False)
    key.drop(columns=["statement", "response"]).to_csv(KEY, index=False)
    MANIFEST.write_text(json.dumps(manifest(key, pop), indent=1) + "\n")
    print(json.dumps(manifest(key, pop), indent=1))


def manifest(key: pd.DataFrame, pop: dict) -> dict:
    reused = int((key["label_source"] == LABEL_SOURCE_AUDIT).sum())
    new = int((key["label_source"] == LABEL_SOURCE_NEW).sum())
    return {
        "protocol": "docs/experiments/selective_commitment_human_reestimate_qwen_v1.md",
        "seed": SEED, "population_item_polarities_by_stratum": {s: int(pop.get(s, 0)) for s in STRATA},
        "sampled_item_polarities_by_stratum": {s: int(key[key["stratum"] == s][["item_id", "polarity"]]
                                                      .drop_duplicates().shape[0]) for s in STRATA},
        "total_rows": len(key), "prompts_per_item_polarity": 5,
        "rows_with_reused_A_label": reused, "rows_with_reused_C_label": reused,
        "rows_with_reused_adjudicated_consensus": reused,
        "rows_requiring_new_A_label": new, "rows_requiring_new_C_label": new,
        "new_sheet_sha256": {a: file_sha(ROOT / f"sheet_{a}.csv") for a in "AC" if (ROOT / f"sheet_{a}.csv").exists()},
        "key": "key.csv is local-only (gitignored) until both new sheets are labelled",
    }


# ---------------------------------------------------------------- estimator


def design_arrays(pop_clusters: pd.DataFrame, sample: pd.DataFrame, c_col: str, sf_col: str = "sf_outcome"):
    """Per-item, per-stratum population cluster counts N, sampled cluster counts n, and sums S of the
    unit-level indicators that the ratio-of-sums gap needs. Items are the population's, sorted."""
    items = sorted(pop_clusters["item_id"].unique())
    ix = {i: n for n, i in enumerate(items)}
    hx = {h: n for n, h in enumerate(STRATA)}
    N = np.zeros((len(items), len(STRATA)))
    for i, h in zip(pop_clusters["item_id"], pop_clusters["stratum"], strict=True):
        N[ix[i], hx[h]] += 1
    n = np.zeros_like(N)
    for (i, _), g in sample.groupby(["item_id", "polarity"]):
        n[ix[i], hx[g["stratum"].iloc[0]]] += 1
    s = sample
    d = s[sf_col].isin(JQY_DIRECTIONAL)
    ag = s[sf_col] == "agreement"
    L, R = s["item_lean"] == "left", s["item_lean"] == "right"
    c = s[c_col].astype(int) == 1
    ind = {"units": pd.Series(True, index=s.index), "commit": c}
    for tag, cc in (("1", c), ("0", ~c)):
        ind |= {f"dL{tag}": cc & d & L, f"agL{tag}": cc & d & L & ag, f"dR{tag}": cc & d & R,
                f"agR{tag}": cc & d & R & ag, f"dir{tag}": cc & d}
    names = list(ind)
    S = np.zeros((len(items), len(STRATA), len(names)))
    for k, name in enumerate(names):
        np.add.at(S[:, :, k], (s["item_id"].map(ix).to_numpy(), s["stratum"].map(hx).to_numpy()),
                  ind[name].to_numpy(float))
    return items, N, n, S, names


def weighted_sums(N: np.ndarray, n: np.ndarray, S: np.ndarray) -> np.ndarray:
    """N, n: (..., items, H); S: (..., items, H, K). Stratum weights N_h/n_h from the (resampled)
    items. A stratum present in the population but absent from the sample makes the replicate NaN."""
    Nh, nh = N.sum(-2), n.sum(-2)
    with np.errstate(invalid="ignore", divide="ignore"):
        w = np.where(Nh == 0, 0.0, np.where(nh == 0, np.nan, Nh / np.where(nh == 0, 1, nh)))
    return (S.sum(-3) * w[..., None]).sum(-2)


def gap_from_sums(W: np.ndarray, names: list[str]) -> dict[str, np.ndarray]:
    g = {k: W[..., names.index(k)] for k in names}
    with np.errstate(invalid="ignore", divide="ignore"):
        d1 = g["agL1"] / g["dL1"] - g["agR1"] / g["dR1"]
        d0 = g["agL0"] / g["dL0"] - g["agR0"] / g["dR0"]
        return {"D_SF_committed": d1, "D_SF_uncommitted": d0, "selection_gap": d1 - d0,
                "commit_rate": g["commit"] / g["units"]}


def boot_idx(n_items: int, n_boot: int = N_BOOT) -> np.ndarray:
    """Identical draws to analyze_olmo_stages.boot_diff (frozen item bootstrap)."""
    return np.random.default_rng(0).integers(0, n_items, size=(n_boot, n_items))


def ci(v: np.ndarray, n_boot: int = N_BOOT) -> tuple[float, float]:
    v = v[~np.isnan(v)]
    if len(v) < 0.9 * n_boot:
        return math.nan, math.nan
    return float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))


def estimate(pop_clusters: pd.DataFrame, sample: pd.DataFrame, label_cols: dict[str, str],
             sf_col: str = "sf_outcome") -> pd.DataFrame:
    """Design-weighted gap per label source, with CIs and paired differences vs the first source."""
    idx = None
    reps, rows = {}, []
    for name, col in label_cols.items():
        items, N, n, S, names = design_arrays(pop_clusters, sample, col, sf_col)
        idx = boot_idx(len(items)) if idx is None else idx
        point = gap_from_sums(weighted_sums(N, n, S), names)
        rep = gap_from_sums(weighted_sums(N[idx], n[idx], S[idx]), names)
        reps[name] = rep
        cnt = {k: int(S[:, :, names.index(k)].sum()) for k in ("units", "commit", "dir1", "dir0")}
        estimable = cnt["dir1"] >= MIN_DIRECTIONAL_UNITS and cnt["dir0"] >= MIN_DIRECTIONAL_UNITS
        lo, hi = ci(rep["selection_gap"]) if estimable else (math.nan, math.nan)
        rows.append({"label_source": name, "sf": sf_col, "n_units": cnt["units"], "n_committed_units": cnt["commit"],
                     "committed_sf_directional_n": cnt["dir1"], "uncommitted_sf_directional_n": cnt["dir0"],
                     "commit_rate_weighted": float(point["commit_rate"]),
                     "D_SF_committed": float(point["D_SF_committed"]),
                     "D_SF_uncommitted": float(point["D_SF_uncommitted"]),
                     "selection_gap": float(point["selection_gap"]), "ci95_low": lo, "ci95_high": hi,
                     "estimable": estimable, "ci_lower_above_0": bool(estimable and lo > 0)})
    first = next(iter(label_cols))
    for name in list(label_cols)[1:]:
        r = next(x for x in rows if x["label_source"] == name)
        p = next(x for x in rows if x["label_source"] == first)
        lo, hi = ci(reps[first]["selection_gap"] - reps[name]["selection_gap"])
        rows.append({"label_source": f"{first}_minus_{name}", "sf": sf_col,
                     "selection_gap": p["selection_gap"] - r["selection_gap"], "ci95_low": lo, "ci95_high": hi,
                     "commit_rate_weighted": p["commit_rate_weighted"] - r["commit_rate_weighted"]})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- analyze


def human_labels(key: pd.DataFrame) -> pd.DataFrame:
    """Consensus, A and C relation per keyed row, from the label source recorded in the key."""
    import analyze_commitment_audit as ca

    out = key[["cache_key", "label_source", "label_sample_id"]].copy()
    for src, audit in ((LABEL_SOURCE_AUDIT, "qwen"), (LABEL_SOURCE_NEW, "qwen_gap")):
        a, c = ca.load_sheets(audit)
        cons_path = ca.root_of(audit) / "labels_consensus.csv"
        if not cons_path.exists():
            raise SystemExit(f"{cons_path} missing: run consensus for --audit {audit}")
        cons = ca.read_csv(cons_path).set_index("sample_id")["relation"]
        m = out["label_source"] == src
        for col, labels in (("rel_consensus", cons), ("rel_A", a["relation"]), ("rel_C", c["relation"])):
            out.loc[m, col] = out.loc[m, "label_sample_id"].map(labels)
    missing = out[["rel_consensus", "rel_A", "rel_C"]].isna().any(axis=1)
    if missing.any():
        raise SystemExit(f"{int(missing.sum())} keyed rows have no human label")
    return out


def cmd_analyze() -> None:
    units = population()
    pop_cl = clusters(units)
    key = pd.read_csv(KEY, dtype={"C_jqy": int})
    s = units.merge(key[["cache_key", "label_source", "label_sample_id"]], on="cache_key")
    assert len(s) == len(key) and s.groupby(["item_id", "polarity"]).size().eq(5).all()
    lab = human_labels(key).set_index("cache_key")
    for col in ("consensus", "A", "C"):
        s[f"C_h_{col}"] = s["cache_key"].map(lab[f"rel_{col}"]).isin(HUMAN_DIRECTIONAL).astype(int)
    primary = estimate(pop_cl, s, {"human_consensus": "C_h_consensus", "jqy_same_units": "C_jqy"})
    sens = pd.concat([
        estimate(pop_cl, s, {"human_consensus": "C_h_consensus", "jqy_same_units": "C_jqy"}, sf_col="sf_outcome_det"),
        estimate(pop_cl, s, {"human_A": "C_h_A", "jqy_same_units": "C_jqy"}),
        estimate(pop_cl, s, {"human_C": "C_h_C", "jqy_same_units": "C_jqy"}),
    ])
    census = estimate(pop_cl, units.assign(), {"jqy_population": "C_jqy"})  # reproduces +0.32 (design check)
    by_stratum = []
    for h, g in s.groupby("stratum"):
        for src in ("C_h_consensus", "C_jqy"):
            by_stratum.append({"stratum": h, "label": src, "n_units": len(g), "n_clusters": g.groupby(["item_id", "polarity"]).ngroups,
                               "committed_units": int(g[src].sum()), "commit_rate": g[src].mean(),
                               "clusters_with_any_commitment": int(g.groupby(["item_id", "polarity"])[src].max().sum())})
    per_prompt = s.groupby(["prompt", "stratum"]).agg(n=("cache_key", "size"), human_commit=("C_h_consensus", "sum"),
                                                      jqy_commit=("C_jqy", "sum")).reset_index()
    tables = {"primary": primary, "sensitivity": sens, "design_check_census": census,
              "by_stratum": pd.DataFrame(by_stratum), "per_prompt": per_prompt}
    Path(OUT).parent.mkdir(parents=True, exist_ok=True)
    for name, t in tables.items():
        t.round(4).to_csv(f"{OUT}_{name}.csv", index=False)
    Path(f"{OUT}_report.md").write_text(
        "<!-- generated by scripts/selection_gap_human_qwen_v1.py analyze; do not edit -->\n\n"
        + "\n\n".join(f"## {k}\n\n{v.round(3).to_markdown(index=False)}" for k, v in tables.items()) + "\n",
        encoding="utf-8")
    for name, t in tables.items():
        print(f"== {name}\n{t.round(3).to_string(index=False)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["sample", "analyze"])
    args = ap.parse_args()
    cmd_sample() if args.cmd == "sample" else cmd_analyze()


if __name__ == "__main__":
    main()
