"""Human audits of the commitment indicator C (OLMo: commitment_audit_v1.md;
Qwen self-judging: commitment_audit_qwen_v1.md). Same code path for both.

  uv run python scripts/analyze_commitment_audit.py check     --audit olmo   # no key access
  uv run python scripts/analyze_commitment_audit.py consensus --audit olmo   # after adjudication
  uv run python scripts/analyze_commitment_audit.py analyze   --audit olmo   # joins the frozen key

`check` validates the completed sheets, reports human A<->C agreement and writes
the adjudication queue. It never reads the key. The A and C sheets are read,
never written. `consensus` builds a separate researcher-adjudicated consensus
from the completed queue. `analyze` is the only step that opens key.csv and
runs the frozen analyses.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
from analyze_human_validation import kappa, wilson

AUDITS = {"olmo": ("artifacts/labeling/commitment_audit_v1", 150, "CA"),
          "qwen": ("artifacts/labeling/commitment_audit_qwen_v1", 100, "CQ")}
RELATIONS = {"supports", "opposes", "mixed_or_conditional", "neutral_or_no_position", "unclear"}
MODES = {"explicit_stance", "analytical_exposition", "voiced_continuation", "refusal", "empty", "incomplete", "other"}
DIRECTIONAL = {"supports", "opposes"}
OUT = "artifacts/tables/commitment_audit"


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
    root, n, prefix = AUDITS[audit]
    sheets = {a: pd.read_csv(f"{root}/sheet_{a}.csv", keep_default_na=False, dtype=str) for a in "AC"}
    problems = [p for a, s in sheets.items() for p in validate(s, n, prefix, f"sheet_{a}")]
    a, c = (sheets[x].set_index("sample_id").sort_index() for x in "AC")
    if set(a.index) != set(c.index):
        problems.append("sheet A and sheet C sample-id sets differ")
    elif not a[["statement", "response"]].equals(c[["statement", "response"]]):
        problems.append("sheet A and sheet C differ in statement/response text")
    if problems:
        raise SystemExit("validation failed:\n  " + "\n  ".join(problems))
    return a, c


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
    return pd.DataFrame(rows, columns=["dimension", "n", "exact_agreement", "cohen_kappa"])


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
    root = AUDITS[audit][0]
    a, c = load_sheets(audit)
    agree = agreement_table(a, c)
    Path(OUT).parent.mkdir(parents=True, exist_ok=True)
    agree.round(4).to_csv(f"{OUT}_{audit}_human_agreement.csv", index=False)
    queue_path = Path(f"{root}/adjudication_queue.csv")
    if queue_path.exists():
        print(f"{queue_path} exists; not overwritten")
    else:
        q = adjudication_queue(a, c)
        q.to_csv(queue_path, index=False)
        print(f"wrote {len(q)} disagreement rows to {queue_path}")
    print(agree.round(3).to_string(index=False))


def build_consensus(a: pd.DataFrame, c: pd.DataFrame, adjudicated: pd.DataFrame) -> pd.DataFrame:
    adj = adjudicated.set_index("sample_id")
    rows = []
    for sid in a.index:
        if sid in adj.index:
            rel, mode = adj.at[sid, "relation_adjudicated"], adj.at[sid, "mode_adjudicated"]
            if rel not in RELATIONS or mode not in MODES:
                raise SystemExit(f"{sid}: adjudication incomplete or outside the taxonomy ({rel!r}, {mode!r})")
            provenance = "researcher_adjudicated"
        else:
            rel, mode, provenance = a.at[sid, "relation"], a.at[sid, "mode"], "human_A_C_agreed"
        rows.append({"sample_id": sid, "relation": rel, "mode": mode, "provenance": provenance})
    return pd.DataFrame(rows)


def cmd_consensus(audit: str) -> None:
    root = AUDITS[audit][0]
    a, c = load_sheets(audit)
    out = Path(f"{root}/labels_consensus.csv")
    if out.exists():
        raise SystemExit(f"{out} exists; consensus is never rebuilt in place")
    adjudicated = pd.read_csv(f"{root}/adjudication_queue.csv", keep_default_na=False, dtype=str)
    expected = set(adjudication_queue(a, c)["sample_id"])
    if set(adjudicated["sample_id"]) != expected:
        raise SystemExit("adjudication queue does not match the current A/C disagreements")
    cons = build_consensus(a, c, adjudicated)
    cons.to_csv(out, index=False)
    print(cons["provenance"].value_counts().to_string())


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
    return {"n": n, "human_commit_rate": float(pos.mean()), "jqy_commit_rate": float(c_jqy.mean()),
            "accuracy": float((c_jqy == c_human).mean()), "cohen_kappa": kappa(c_jqy, c_human),
            "positive_agreement": 2 * a11 / (2 * a11 + (n - a11 - a00)) if n else float("nan"),
            "negative_agreement": 2 * a00 / (2 * a00 + (n - a11 - a00)) if n else float("nan"),
            "overread_k": over_k, "overread_n": int(neg.sum()),
            "overread_rate": over_k / neg.sum() if neg.any() else float("nan"), "overread_ci_low": o_lo,
            "overread_ci_high": o_hi, "underread_k": under_k, "underread_n": int(pos.sum()),
            "underread_rate": under_k / pos.sum() if pos.any() else float("nan"), "underread_ci_low": u_lo,
            "underread_ci_high": u_hi, "both_directional_n": int(both.sum()), "reversals": rev}


def cmd_analyze(audit: str) -> None:
    root = AUDITS[audit][0]
    a, c = load_sheets(audit)
    cons_path = Path(f"{root}/labels_consensus.csv")
    if not cons_path.exists():
        raise SystemExit("run `consensus` after adjudication first")
    key = pd.read_csv(f"{root}/key.csv").set_index("sample_id")
    cons = pd.read_csv(cons_path, keep_default_na=False).set_index("sample_id")
    assert set(key.index) == set(cons.index) == set(a.index), "key / consensus / sheet id sets differ"
    rows = []
    slice_cols = ["sf_pole", "prompt"] + (["stage"] if "stage" in key.columns else [])
    for reference, labels in (("consensus", cons["relation"]), ("human_A", a["relation"]), ("human_C", c["relation"])):
        d = key.join(labels.rename("h_rel"))
        d["C_human"] = committed(d["h_rel"])
        d["C_jqy"] = d["C_jqy"].astype(bool).astype(int)
        rows.append({"reference": reference, "slice": "all", "value": "all",
                     **commitment_metrics(d["C_jqy"], d["C_human"], d["h_rel"], d["jqy_outcome"])})
        for col in slice_cols:
            for val, g in d.groupby(col):
                rows.append({"reference": reference, "slice": col, "value": val,
                             **commitment_metrics(g["C_jqy"], g["C_human"], g["h_rel"], g["jqy_outcome"])})
    table = pd.DataFrame(rows)
    table.round(4).to_csv(f"{OUT}_{audit}_metrics.csv", index=False)
    show = table[(table["reference"] == "consensus") & table["slice"].isin(["all", "sf_pole"])]
    print(show.round(3).to_string(index=False))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["check", "consensus", "analyze"])
    ap.add_argument("--audit", choices=sorted(AUDITS), required=True)
    args = ap.parse_args()
    {"check": cmd_check, "consensus": cmd_consensus, "analyze": cmd_analyze}[args.cmd](args.audit)


if __name__ == "__main__":
    main()
