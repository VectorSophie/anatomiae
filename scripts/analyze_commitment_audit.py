"""Human audits of the commitment indicator C (OLMo: commitment_audit_v1.md;
Qwen self-judging: commitment_audit_qwen_v1.md). Same code path for both. The
blind sheets of the Qwen human selection-gap re-estimation
(selective_commitment_human_reestimate_qwen_v1.md) also use import/check/consensus.

  uv run python scripts/analyze_commitment_audit.py import-sheet --audit olmo --annotator A path/olmo_sheet_A.completed.csv
  uv run python scripts/analyze_commitment_audit.py check     --audit olmo   # no key access
  uv run python scripts/analyze_commitment_audit.py import-adjudication --audit olmo path/adjudication_queue_olmo.completed.csv
  uv run python scripts/analyze_commitment_audit.py consensus --audit olmo   # after adjudication; no key access
  uv run python scripts/analyze_commitment_audit.py analyze   --audit olmo   # joins the frozen key
  uv run python scripts/analyze_commitment_audit.py compare                  # Qwen vs OLMo (reading rule, part i)

`import-sheet` checks a completed export against the frozen blind sheet and stores
a byte-identical copy as `sheet_<X>.completed.csv` next to it. The blank frozen
sheet is never modified; an existing import is never replaced. `check` validates
the completed sheets, reports human A<->C agreement and writes the adjudication
queue. It never reads the key. `import-adjudication` stores the researcher's
completed queue after checking it against the A/C disagreements. `consensus`
builds a separate researcher-adjudicated consensus. `analyze` is the only step
that opens key.csv and runs the frozen analyses plus the secondary calculations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from analyze_human_validation import kappa, wilson

AUDITS = {"olmo": ("artifacts/labeling/commitment_audit_v1", 150, "CA"),
          "qwen": ("artifacts/labeling/commitment_audit_qwen_v1", 100, "CQ"),
          # blind sheets of the Qwen human selection-gap re-estimation: new rows only
          # (n fixed by the frozen sample; see selective_commitment_human_reestimate_qwen_v1.md)
          "qwen_gap": ("artifacts/labeling/selection_gap_human_qwen_v1", 142, "QG")}
KEYED_AUDITS = ("olmo", "qwen")
COLUMNS = ["sample_id", "statement", "response", "relation", "mode", "confidence_1to3", "notes"]
RELATIONS = {"supports", "opposes", "mixed_or_conditional", "neutral_or_no_position", "unclear"}
MODES = {"explicit_stance", "analytical_exposition", "voiced_continuation", "refusal", "empty", "incomplete", "other"}
DIRECTIONAL = {"supports", "opposes"}
QUEUE_COLUMNS = ["sample_id", "relation_A", "relation_C", "mode_A", "mode_C", "commitment_differs",
                 "relation_adjudicated", "mode_adjudicated", "adjudication_notes"]
OUT = "artifacts/tables/commitment_audit"
N_BOOT = 2000
POLES = ("left_pole", "right_pole", "non_directional")


def root_of(audit: str) -> Path:
    return Path(AUDITS[audit][0])


def sheet_path(audit: str, annotator: str) -> Path:
    """The imported completed sheet if present, else the frozen blind sheet."""
    done = root_of(audit) / f"sheet_{annotator}.completed.csv"
    return done if done.exists() else root_of(audit) / f"sheet_{annotator}.csv"


def queue_path(audit: str) -> Path:
    done = root_of(audit) / "adjudication_queue.completed.csv"
    return done if done.exists() else root_of(audit) / "adjudication_queue.csv"


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, keep_default_na=False, dtype=str)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate(sheet: pd.DataFrame, n: int, prefix: str, name: str) -> list[str]:
    problems = []
    if len(sheet) != n or sheet["sample_id"].nunique() != n:
        problems.append(f"{name}: expected {n} unique sample ids, got {sheet['sample_id'].nunique()} of {len(sheet)} rows")
    if not sheet["sample_id"].str.startswith(prefix).all():
        problems.append(f"{name}: sample ids not of the form {prefix}nnn")
    for col in ("relation", "mode", "confidence_1to3"):
        missing = (sheet[col].astype(str).str.strip() == "").sum()
        if missing:
            problems.append(f"{name}: {missing} missing {col}")
    bad_rel = set(sheet["relation"]) - RELATIONS - {""}
    bad_mode = set(sheet["mode"]) - MODES - {""}
    bad_conf = set(sheet["confidence_1to3"].astype(str)) - {"1", "2", "3", ""}
    for label, bad in (("relation", bad_rel), ("mode", bad_mode), ("confidence_1to3", bad_conf)):
        if bad:
            problems.append(f"{name}: values outside the taxonomy in {label}: {sorted(bad)}")
    return problems


def load_sheets(audit: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    _, n, prefix = AUDITS[audit]
    sheets = {a: read_csv(sheet_path(audit, a)) for a in "AC"}
    problems = [p for a, s in sheets.items() for p in validate(s, n, prefix, f"sheet_{a}")]
    a, c = (sheets[x].set_index("sample_id").sort_index() for x in "AC")
    if set(a.index) != set(c.index):
        problems.append("sheet A and sheet C sample-id sets differ")
    elif not a[["statement", "response"]].equals(c[["statement", "response"]]):
        problems.append("sheet A and sheet C differ in statement/response text")
    if problems:
        raise SystemExit("validation failed:\n  " + "\n  ".join(problems))
    return a, c


# ---------------------------------------------------------------- import (non-destructive)


def verify_completed(frozen: pd.DataFrame, done: pd.DataFrame, n: int, prefix: str) -> list[str]:
    """A completed export must be the frozen blind sheet, same order and text, with labels filled."""
    problems = []
    if list(done.columns) != COLUMNS:
        return [f"columns {list(done.columns)} != {COLUMNS}"]
    if len(done) != len(frozen):
        return [f"row count {len(done)} != frozen {len(frozen)}"]
    for col in ("sample_id", "statement", "response"):
        bad = (done[col].to_numpy() != frozen[col].to_numpy()).sum()
        if bad:
            problems.append(f"{bad} rows differ from the frozen sheet in {col} (order or text)")
    return problems + validate(done, n, prefix, "completed")


def cmd_import_sheet(audit: str, annotator: str, src: Path) -> None:
    root, n, prefix = AUDITS[audit]
    frozen_path = Path(root) / f"sheet_{annotator}.csv"
    dest = Path(root) / f"sheet_{annotator}.completed.csv"
    problems = verify_completed(read_csv(frozen_path), read_csv(src), n, prefix)
    if problems:
        raise SystemExit("not imported:\n  " + "\n  ".join(problems[:50]))
    if dest.exists():
        if sha256(dest) == sha256(src):
            print(f"{dest} already imported (identical)")
            return
        raise SystemExit(f"{dest} exists with different content; imported labels are never replaced")
    shutil.copyfile(src, dest)
    record_import(audit, f"sheet_{annotator}", src, dest)
    print(f"imported {src.name} -> {dest} (sha256 {sha256(dest)[:16]})")


def record_import(audit: str, what: str, src: Path, dest: Path) -> None:
    log = root_of(audit) / "imports.json"
    entries = json.loads(log.read_text()) if log.exists() else []
    entries.append({"artifact": what, "source_filename": src.name, "stored_as": dest.name,
                    "sha256": sha256(dest), "imported_utc": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds")})
    log.write_text(json.dumps(entries, indent=1) + "\n")


def verify_queue(expected: pd.DataFrame, done: pd.DataFrame) -> list[str]:
    missing = [c for c in QUEUE_COLUMNS if c not in done.columns]
    if missing:
        return [f"adjudication file lacks columns {missing}"]
    problems = []
    if sorted(done["sample_id"]) != sorted(expected["sample_id"]) or done["sample_id"].duplicated().any():
        problems.append(f"sample ids ({len(done)}) do not match the {len(expected)} current A/C disagreements")
        return problems
    d, e = done.set_index("sample_id"), expected.set_index("sample_id").reindex(done["sample_id"])
    for col in ("relation_A", "relation_C", "mode_A", "mode_C"):
        bad = (d[col] != e[col]).sum()
        if bad:
            problems.append(f"{bad} rows: {col} differs from the imported A/C sheets")
    try:
        resolve_adjudication(d)
    except ValueError as err:
        problems.append(str(err))
    return problems


def cmd_import_adjudication(audit: str, src: Path) -> None:
    a, c = load_sheets(audit)
    dest = root_of(audit) / "adjudication_queue.completed.csv"
    problems = verify_queue(adjudication_queue(a, c), read_csv(src))
    if problems:
        raise SystemExit("not imported:\n  " + "\n  ".join(problems[:50]))
    if dest.exists():
        if sha256(dest) == sha256(src):
            print(f"{dest} already imported (identical)")
            return
        raise SystemExit(f"{dest} exists with different content; adjudication is never replaced")
    shutil.copyfile(src, dest)
    record_import(audit, "adjudication_queue", src, dest)
    print(f"imported {src.name} -> {dest}")


# ---------------------------------------------------------------- blind steps


def committed(relation: pd.Series) -> pd.Series:
    return relation.isin(DIRECTIONAL).astype(int)


def agreement_table(a: pd.DataFrame, c: pd.DataFrame) -> pd.DataFrame:
    both_dir = a["relation"].isin(DIRECTIONAL) & c["relation"].isin(DIRECTIONAL)
    rows = [
        ("relation", len(a), (a["relation"] == c["relation"]).mean(), kappa(a["relation"], c["relation"])),
        ("mode", len(a), (a["mode"] == c["mode"]).mean(), kappa(a["mode"], c["mode"])),
        ("commitment_binary", len(a), (committed(a["relation"]) == committed(c["relation"])).mean(),
         kappa(committed(a["relation"]), committed(c["relation"]))),
        ("direction_when_both_directional", int(both_dir.sum()),
         (a["relation"][both_dir] == c["relation"][both_dir]).mean() if both_dir.any() else float("nan"),
         kappa(a["relation"][both_dir], c["relation"][both_dir]) if both_dir.any() else float("nan")),
    ]
    t = pd.DataFrame(rows, columns=["dimension", "n", "exact_agreement", "cohen_kappa"])
    t["n_agree"] = [int((a["relation"] == c["relation"]).sum()), int((a["mode"] == c["mode"]).sum()),
                    int((committed(a["relation"]) == committed(c["relation"])).sum()),
                    int((a["relation"][both_dir] == c["relation"][both_dir]).sum())]
    return t[["dimension", "n", "n_agree", "exact_agreement", "cohen_kappa"]]


def adjudication_queue(a: pd.DataFrame, c: pd.DataFrame) -> pd.DataFrame:
    differ = (a["relation"] != c["relation"]) | (a["mode"] != c["mode"])
    q = pd.DataFrame({"sample_id": a.index[differ], "relation_A": a["relation"][differ].values,
                      "relation_C": c["relation"][differ].values, "mode_A": a["mode"][differ].values,
                      "mode_C": c["mode"][differ].values,
                      "commitment_differs": (committed(a["relation"]) != committed(c["relation"]))[differ].values})
    q["relation_adjudicated"] = ""
    q["mode_adjudicated"] = ""
    q["adjudication_notes"] = ""
    return q


def cmd_check(audit: str) -> None:
    a, c = load_sheets(audit)
    agree = agreement_table(a, c)
    Path(OUT).parent.mkdir(parents=True, exist_ok=True)
    agree.round(4).to_csv(f"{OUT}_{audit}_human_agreement.csv", index=False)
    qp = root_of(audit) / "adjudication_queue.csv"
    if qp.exists():
        print(f"{qp} exists; not overwritten")
    else:
        q = adjudication_queue(a, c)
        q.to_csv(qp, index=False)
        print(f"wrote {len(q)} disagreement rows to {qp}")
    print(agree.round(3).to_string(index=False))


def resolve_adjudication(adj: pd.DataFrame) -> pd.DataFrame:
    """Final relation/mode per queue row. A blank adjudicated field is allowed only where A and C
    already agree on that field (the row is queued for the other field); it then takes the agreed value."""
    out = []
    for sid, r in adj.iterrows():
        rel = r["relation_adjudicated"].strip() or (r["relation_A"] if r["relation_A"] == r["relation_C"] else "")
        mode = r["mode_adjudicated"].strip() or (r["mode_A"] if r["mode_A"] == r["mode_C"] else "")
        if rel not in RELATIONS or mode not in MODES:
            raise ValueError(f"{sid}: adjudication incomplete or outside the taxonomy ({rel!r}, {mode!r})")
        out.append({"sample_id": sid, "relation": rel, "mode": mode})
    return pd.DataFrame(out, columns=["sample_id", "relation", "mode"]).set_index("sample_id")


def build_consensus(a: pd.DataFrame, c: pd.DataFrame, adjudicated: pd.DataFrame) -> pd.DataFrame:
    try:
        adj = resolve_adjudication(adjudicated.set_index("sample_id"))
    except ValueError as err:
        raise SystemExit(str(err)) from err
    rows = []
    for sid in a.index:
        if sid in adj.index:
            rel, mode, provenance = adj.at[sid, "relation"], adj.at[sid, "mode"], "researcher_adjudicated"
        else:
            if (a.at[sid, "relation"], a.at[sid, "mode"]) != (c.at[sid, "relation"], c.at[sid, "mode"]):
                raise SystemExit(f"{sid}: A and C disagree but the row is not in the adjudication queue")
            rel, mode, provenance = a.at[sid, "relation"], a.at[sid, "mode"], "human_A_C_agreed"
        rows.append({"sample_id": sid, "relation": rel, "mode": mode, "provenance": provenance})
    return pd.DataFrame(rows)


def cmd_consensus(audit: str) -> None:
    a, c = load_sheets(audit)
    out = root_of(audit) / "labels_consensus.csv"
    if out.exists():
        raise SystemExit(f"{out} exists; consensus is never rebuilt in place")
    adjudicated = read_csv(queue_path(audit))
    expected = set(adjudication_queue(a, c)["sample_id"])
    if set(adjudicated["sample_id"]) != expected:
        raise SystemExit("adjudication queue does not match the current A/C disagreements")
    cons = build_consensus(a, c, adjudicated)
    cons.to_csv(out, index=False)
    print(cons["provenance"].value_counts().to_string())


# ---------------------------------------------------------------- keyed analysis


def commitment_metrics(c_jqy: pd.Series, c_human: pd.Series, h_rel: pd.Series, jqy_out: pd.Series) -> dict:
    n = len(c_jqy)
    pos, neg = c_human == 1, c_human == 0
    over_k, under_k = int((c_jqy[neg] == 1).sum()), int((c_jqy[pos] == 0).sum())
    both = (c_jqy == 1) & pos
    to_dir = {"agreement": "supports", "disagreement": "opposes"}
    rev = int((jqy_out[both].map(to_dir) != h_rel[both]).sum())
    a11 = int(((c_jqy == 1) & pos).sum())
    a00 = int(((c_jqy == 0) & neg).sum())
    o_lo, o_hi = wilson(over_k, int(neg.sum()))
    u_lo, u_hi = wilson(under_k, int(pos.sum()))
    return {"n": n, "human_commit_k": int(pos.sum()), "human_commit_rate": float(pos.mean()) if n else math.nan,
            "jqy_commit_k": int(c_jqy.sum()), "jqy_commit_rate": float(c_jqy.mean()) if n else math.nan,
            "accuracy": float((c_jqy == c_human).mean()) if n else math.nan, "cohen_kappa": kappa(c_jqy, c_human),
            "positive_agreement": 2 * a11 / (2 * a11 + (n - a11 - a00)) if (2 * a11 + n - a11 - a00) else math.nan,
            "negative_agreement": 2 * a00 / (2 * a00 + (n - a11 - a00)) if (2 * a00 + n - a11 - a00) else math.nan,
            "overread_k": over_k, "overread_n": int(neg.sum()),
            "overread_rate": over_k / neg.sum() if neg.any() else math.nan, "overread_ci_low": o_lo,
            "overread_ci_high": o_hi, "underread_k": under_k, "underread_n": int(pos.sum()),
            "underread_rate": under_k / pos.sum() if pos.any() else math.nan, "underread_ci_low": u_lo,
            "underread_ci_high": u_hi, "both_directional_n": int(both.sum()), "reversals": rev}


def fisher_exact(k1: int, n1: int, k2: int, n2: int) -> float:
    """Two-sided Fisher exact p for k1/n1 vs k2/n2 (sum of tables no more likely than observed)."""
    k, n = k1 + k2, n1 + n2
    if n1 == 0 or n2 == 0:
        return math.nan
    lo, hi = max(0, k - n2), min(k, n1)
    p = {x: math.comb(n1, x) * math.comb(n2, k - x) / math.comb(n, k) for x in range(lo, hi + 1)}
    return float(min(1.0, sum(v for v in p.values() if v <= p[k1] * (1 + 1e-9))))


def newcombe(k1: int, n1: int, k2: int, n2: int) -> tuple[float, float]:
    """Newcombe hybrid-score 95% CI for p1 - p2 (method 10), from two Wilson intervals."""
    if n1 == 0 or n2 == 0:
        return math.nan, math.nan
    p1, p2 = k1 / n1, k2 / n2
    l1, u1 = wilson(k1, n1)
    l2, u2 = wilson(k2, n2)
    d = p1 - p2
    return d - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2), d + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)


def cluster_boot(d: pd.DataFrame, masks: dict[str, pd.Series], exprs: dict, cluster: str = "item_id",
                 n_boot: int = N_BOOT, seed: int = 0) -> dict[str, tuple[float, float, float]]:
    """Point estimate and 95% percentile CI of ratio-of-sums expressions, resampling clusters.
    `masks` are 0/1 per row; each expression maps a dict of (summed) masks to a value/array.
    NaN interval if the expression is undefined in more than 10% of resamples (project rule)."""
    clusters = sorted(d[cluster].unique())
    m = pd.DataFrame({k: v.astype(float).to_numpy() for k, v in masks.items()}, index=d.index)
    per = m.groupby(d[cluster].to_numpy()).sum().reindex(clusters, fill_value=0.0)
    idx = np.random.default_rng(seed).integers(0, len(clusters), size=(n_boot, len(clusters)))
    sums = {k: per[k].to_numpy()[idx].sum(1) for k in per.columns}
    point = {k: float(per[k].sum()) for k in per.columns}
    out = {}
    with np.errstate(invalid="ignore", divide="ignore"):
        for name, f in exprs.items():
            est = float(f({k: np.float64(v) for k, v in point.items()}))
            v = np.asarray(f(sums), float)
            v = v[~np.isnan(v)]
            lo, hi = ((math.nan, math.nan) if len(v) < 0.9 * n_boot else
                      (float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))))
            out[name] = (est, lo, hi)
    return out


def pole_contrasts(d: pd.DataFrame) -> list[dict]:
    """Left- vs right-pole differences of evaluator error and commitment rates, with counts,
    Newcombe CI, Fisher exact p (descriptive) and an item-cluster bootstrap CI."""
    left, right = d["sf_pole"] == "left_pole", d["sf_pole"] == "right_pole"
    pos, neg = d["C_human"] == 1, d["C_human"] == 0
    jq = d["C_jqy"] == 1
    defs = {  # name: (event, base)
        "overread": (jq & neg, neg), "underread": (~jq & pos, pos),
        "human_commit": (pos, pd.Series(True, index=d.index)), "jqy_commit": (jq, pd.Series(True, index=d.index)),
    }
    rows = []
    for name, (event, base) in defs.items():
        k1, n1 = int((event & base & left).sum()), int((base & left).sum())
        k2, n2 = int((event & base & right).sum()), int((base & right).sum())
        masks = {"kL": event & base & left, "nL": base & left, "kR": event & base & right, "nR": base & right}
        b = cluster_boot(d, masks, {"diff": lambda s: s["kL"] / s["nL"] - s["kR"] / s["nR"]})["diff"]
        nc = newcombe(k1, n1, k2, n2)
        rows.append({"quantity": f"{name}_left_minus_right", "left_k": k1, "left_n": n1,
                     "left_rate": k1 / n1 if n1 else math.nan, "right_k": k2, "right_n": n2,
                     "right_rate": k2 / n2 if n2 else math.nan, "diff": b[0], "newcombe_low": nc[0],
                     "newcombe_high": nc[1], "fisher_p": fisher_exact(k1, n1, k2, n2),
                     "boot_ci_low": b[1], "boot_ci_high": b[2]})
    return rows


def measurement_effect(d: pd.DataFrame) -> list[dict]:
    """Does the evaluator preserve, attenuate or amplify selective commitment? Paired, item-cluster bootstrap."""
    left, right = d["sf_pole"] == "left_pole", d["sf_pole"] == "right_pole"
    h, j = d["C_human"] == 1, d["C_jqy"] == 1
    masks = {"n": pd.Series(True, index=d.index), "h": h, "j": j, "nL": left, "nR": right,
             "hL": h & left, "hR": h & right, "jL": j & left, "jR": j & right}
    exprs = {
        "human_minus_jqy_commit_rate": lambda s: (s["h"] - s["j"]) / s["n"],
        "human_commit_asymmetry_left_minus_right": lambda s: s["hL"] / s["nL"] - s["hR"] / s["nR"],
        "jqy_commit_asymmetry_left_minus_right": lambda s: s["jL"] / s["nL"] - s["jR"] / s["nR"],
        "jqy_minus_human_asymmetry": lambda s: (s["jL"] / s["nL"] - s["jR"] / s["nR"])
        - (s["hL"] / s["nL"] - s["hR"] / s["nR"]),
    }
    res = cluster_boot(d, masks, exprs)
    return [{"quantity": k, "estimate": v[0], "boot_ci_low": v[1], "boot_ci_high": v[2],
             "n": len(d), "n_left": int(left.sum()), "n_right": int(right.sum())} for k, v in res.items()]


def keyed_frame(key: pd.DataFrame, relation: pd.Series) -> pd.DataFrame:
    d = key.join(relation.rename("h_rel"), how="inner")
    assert len(d) == len(key), "label source does not cover every keyed row"
    d["C_human"] = committed(d["h_rel"])
    d["C_jqy"] = d["C_jqy"].astype(str).str.lower().map({"true": 1, "false": 0, "1": 1, "0": 0}).astype(int)
    return d


def references(audit: str) -> dict[str, pd.Series]:
    a, c = load_sheets(audit)
    cons_path = root_of(audit) / "labels_consensus.csv"
    if not cons_path.exists():
        raise SystemExit("run `consensus` after adjudication first")
    cons = read_csv(cons_path).set_index("sample_id")
    assert set(cons.index) == set(a.index), "consensus / sheet id sets differ"
    return {"consensus": cons["relation"], "human_A": a["relation"], "human_C": c["relation"]}


def cmd_analyze(audit: str) -> None:
    refs = references(audit)  # validates sheets and consensus before the key is opened
    a, c = load_sheets(audit)
    key = pd.read_csv(root_of(audit) / "key.csv").set_index("sample_id")
    assert set(key.index) == set(refs["consensus"].index), "key / consensus id sets differ"
    slice_cols = ["sf_pole", "prompt"] + (["stage"] if "stage" in key.columns else [])
    rows, contrasts, effects = [], [], []
    for reference, labels in refs.items():
        d = keyed_frame(key, labels)
        rows.append({"reference": reference, "slice": "all", "value": "all",
                     **commitment_metrics(d["C_jqy"], d["C_human"], d["h_rel"], d["jqy_outcome"])})
        for col in slice_cols:
            for val, g in d.groupby(col):
                rows.append({"reference": reference, "slice": col, "value": val,
                             **commitment_metrics(g["C_jqy"], g["C_human"], g["h_rel"], g["jqy_outcome"])})
        contrasts += [{"reference": reference, **r} for r in pole_contrasts(d)]
        effects += [{"reference": reference, **r} for r in measurement_effect(d)]
    tables = {"metrics": pd.DataFrame(rows), "pole_contrasts": pd.DataFrame(contrasts),
              "measurement_effect": pd.DataFrame(effects), "human_agreement": agreement_table(a, c)}
    Path(OUT).parent.mkdir(parents=True, exist_ok=True)
    for name, t in tables.items():
        t.round(4).to_csv(f"{OUT}_{audit}_{name}.csv", index=False)
    Path(f"{OUT}_{audit}_report.md").write_text(report(audit, tables), encoding="utf-8")
    show = tables["metrics"]
    print(show[(show["reference"] == "consensus") & show["slice"].isin(["all", "sf_pole"])].round(3).to_string(index=False))
    print(tables["pole_contrasts"][tables["pole_contrasts"]["reference"] == "consensus"].round(3).to_string(index=False))


def report(audit: str, t: dict[str, pd.DataFrame]) -> str:
    m = t["metrics"]
    core = ["reference", "slice", "value", "n", "human_commit_k", "jqy_commit_k", "accuracy", "cohen_kappa",
            "overread_k", "overread_n", "overread_ci_low", "overread_ci_high", "underread_k", "underread_n",
            "underread_ci_low", "underread_ci_high", "both_directional_n", "reversals"]
    parts = [f"<!-- generated by scripts/analyze_commitment_audit.py analyze --audit {audit}; do not edit -->\n",
             "## Human A <-> C\n", t["human_agreement"].round(3).to_markdown(index=False),
             "\n## J-qy vs human reference (all, by stance-first pole)\n",
             m[m["slice"].isin(["all", "sf_pole"])][core].round(3).to_markdown(index=False),
             "\n## Pole contrasts (left minus right)\n", t["pole_contrasts"].round(3).to_markdown(index=False),
             "\n## Measurement effect\n", t["measurement_effect"].round(3).to_markdown(index=False),
             "\n## Slices (descriptive)\n", m[~m["slice"].isin(["all", "sf_pole"])][core].round(3).to_markdown(index=False)]
    return "\n".join(parts) + "\n"


def cmd_compare() -> None:
    """Part (i) of the frozen Qwen reading rule: do Qwen over/under-read CIs overlap OLMo's (consensus)?"""
    rows = []
    m = {a: pd.read_csv(f"{OUT}_{a}_metrics.csv") for a in KEYED_AUDITS}
    for a in KEYED_AUDITS:
        x = m[a][(m[a]["reference"] == "consensus") & (m[a]["slice"] == "all")].iloc[0]
        rows.append({"audit": a, **{k: x[k] for k in ("n", "accuracy", "cohen_kappa", "overread_rate", "overread_ci_low",
                                                      "overread_ci_high", "underread_rate", "underread_ci_low",
                                                      "underread_ci_high")}})
    t = pd.DataFrame(rows).set_index("audit")
    overlap = {e: bool(t.at["qwen", f"{e}_ci_low"] <= t.at["olmo", f"{e}_ci_high"]
                       and t.at["olmo", f"{e}_ci_low"] <= t.at["qwen", f"{e}_ci_high"]) for e in ("overread", "underread")}
    t.reset_index().round(4).to_csv(f"{OUT}_qwen_vs_olmo.csv", index=False)
    print(t.round(3).to_string())
    print("Wilson CIs overlap:", overlap)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["import-sheet", "import-adjudication", "check", "consensus", "analyze", "compare"])
    ap.add_argument("--audit", choices=sorted(AUDITS))
    ap.add_argument("--annotator", choices=["A", "C"])
    ap.add_argument("path", nargs="?", type=Path)
    args = ap.parse_args()
    if args.cmd == "compare":
        return cmd_compare()
    if not args.audit:
        ap.error("--audit is required")
    if args.cmd == "import-sheet":
        if not (args.annotator and args.path):
            ap.error("import-sheet needs --annotator and a path")
        return cmd_import_sheet(args.audit, args.annotator, args.path)
    if args.cmd == "import-adjudication":
        if not args.path:
            ap.error("import-adjudication needs a path")
        return cmd_import_adjudication(args.audit, args.path)
    if args.cmd == "analyze" and args.audit not in KEYED_AUDITS:
        ap.error("analyze applies to the keyed audits; the re-estimation is analysed by "
                 "scripts/selection_gap_human_qwen_v1.py")
    {"check": cmd_check, "consensus": cmd_consensus, "analyze": cmd_analyze}[args.cmd](args.audit)


if __name__ == "__main__":
    main()
