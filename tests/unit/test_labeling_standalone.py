"""Offline single-file labeling pages: data isolation, CSP integrity, export verification."""

import base64
import csv
import hashlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "labeling_ui"))

import build_standalone as bs
import server as ui
import verify_export as ve


def embedded(html: str) -> dict:
    m = re.search(r'<script type="application/json" id="sheet-data">(.*?)</script>', html, re.DOTALL)
    return json.loads(m.group(1))


@pytest.mark.parametrize("annotator", ["A", "C"])
def test_page_embeds_only_own_blind_rows(annotator):
    html = bs.build(annotator)
    data = embedded(html)
    assert data["annotator"] == annotator
    for audit in ("olmo", "qwen"):
        source = ui.read_sheet(ui.SHEETS[(audit, annotator)])
        assert data["sheets"][audit]["rows"] == ui.blind_rows(source)  # same ids, text, order
    other = "C" if annotator == "A" else "A"
    other_order = [r["sample_id"] for r in ui.read_sheet(ui.SHEETS[("olmo", other)])]
    assert [r["sample_id"] for r in data["sheets"]["olmo"]["rows"]] != other_order
    assert set(data) == {"annotator", "columns", "sheets", "taxonomy"}
    assert all(set(r) == {"sample_id", "statement", "response"} for s in data["sheets"].values() for r in s["rows"])
    # hidden-field names must not appear in the page code/markup (response text may contain any word)
    code = re.sub(r'<script type="application/json" id="sheet-data">.*?</script>', "", html, flags=re.DOTALL)
    leaked = [w for w in ("key.csv", "C_jqy", "jqy_outcome", "sf_pole", "item_lean", "cache_key") if w in code]
    assert leaked == []


def test_csp_pins_inline_code_and_blocks_network():
    html = bs.build("A")
    csp = re.search(r'http-equiv="Content-Security-Policy" content="([^"]+)"', html).group(1)
    assert "connect-src 'none'" in csp and "default-src 'none'" in csp
    scripts = re.findall(r"<script>(.*?)</script>", html, re.DOTALL)
    styles = re.findall(r"<style>(.*?)</style>", html, re.DOTALL)
    assert len(scripts) == 1 and len(styles) == 1
    sha = lambda t: "sha256-" + base64.b64encode(hashlib.sha256(t.encode()).digest()).decode()
    assert f"script-src '{sha(scripts[0])}'" in csp and f"style-src '{sha(styles[0])}'" in csp
    assert not re.search(r'(src|href)="', html)
    assert "fetch(" not in scripts[0] and "XMLHttpRequest" not in scripts[0]


def test_build_is_deterministic_and_committed_pages_are_current():
    for stem, (annotator, _, _, _) in bs.PAGES.items():
        page = bs.OUT_DIR / f"{stem}.html"
        assert bs.build(annotator, stem) == bs.build(annotator, stem)
        if page.exists():
            assert page.read_text(encoding="utf-8") == bs.build(annotator, stem), "rebuild the standalone pages"


def _complete(audit, annotator, tmp_path, mutate=None):
    rows = ui.read_sheet(ui.SHEETS[(audit, annotator)])
    for r in rows:
        r.update(relation="supports", mode="explicit_stance", confidence_1to3="3", notes="")
    if mutate:
        mutate(rows)
    path = tmp_path / "done.csv"
    path.write_text(ui.to_csv(rows), encoding="utf-8", newline="")
    return path


def test_verify_export_accepts_valid_and_rejects_tampering(tmp_path):
    assert ve.verify("olmo", "A", _complete("olmo", "A", tmp_path)) == []
    assert ve.verify("olmo", "A", _complete("olmo", "A", tmp_path, lambda r: r.__setitem__(0, {**r[0], "response": r[0]["response"] + " "})))
    assert ve.verify("olmo", "A", _complete("olmo", "A", tmp_path, lambda r: r.reverse()))
    assert ve.verify("olmo", "A", _complete("olmo", "A", tmp_path, lambda r: r[3].update(mode="")))
    assert ve.verify("qwen", "C", _complete("olmo", "A", tmp_path))  # wrong sheet


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_js_csv_writer_round_trips_through_python_csv(tmp_path):
    js = (bs.UI_DIR / "backend_standalone.js").read_text()
    csv_fn = re.search(r"function csvField\(v\) \{.*?\n  \}", js, re.DOTALL).group(0)
    rows = ui.read_sheet(ui.SHEETS[("olmo", "C")])
    for r in rows:
        r.update(relation="opposes", mode="other", confidence_1to3="1", notes='x, "y"\nz')
    script = (csv_fn + f"\nconst cols={json.dumps(ui.COLUMNS)};const rows={json.dumps(rows)};"
              "process.stdout.write([cols.join(','),...rows.map(r=>cols.map(c=>csvField(r[c])).join(','))].join('\\n')+'\\n');")
    out = subprocess.run(["node", "-"], input=script, capture_output=True, text=True, check=True).stdout
    p = tmp_path / "js.csv"
    p.write_text(out, encoding="utf-8", newline="")
    with p.open(newline="", encoding="utf-8") as f:
        back = list(csv.DictReader(f))
    assert back == rows
    assert ve.verify("olmo", "C", p) == []


FOLLOWUP = {"A": "selection_gap_qwen_annotator_A", "C": "selection_gap_qwen_annotator_C"}


@pytest.mark.parametrize("annotator", ["A", "C"])
def test_followup_page_embeds_only_new_blind_rows(annotator):
    html = bs.build(annotator, FOLLOWUP[annotator])
    data = embedded(html)
    assert list(data["sheets"]) == ["qwen_gap"] and data["annotator"] == annotator
    source = ui.read_sheet(ui.FOLLOWUP_SHEETS[("qwen_gap", annotator)])
    rows = data["sheets"]["qwen_gap"]["rows"]
    assert [{k: r[k] for k in ("sample_id", "statement", "response")} for r in rows] == ui.blind_rows(source)
    assert all(r["sample_id"].startswith("QG") for r in rows)  # no reused audit rows (CQ...) are shown
    expected = {"sample_id", "statement", "response"} | ({"statement_fr"} if annotator == "C" else set())
    assert all(set(r) == expected for r in rows)
    code = re.sub(r'<script type="application/json" id="sheet-data">.*?</script>', "", html, flags=re.DOTALL)
    leaked = [w for w in ("key.csv", "C_jqy", "jqy_outcome", "sf_pole", "stratum", "item_lean", "cache_key",
                          "label_source", "Qwen") if w in code]
    assert leaked == []
    csp = re.search(r'http-equiv="Content-Security-Policy" content="([^"]+)"', html).group(1)
    assert "connect-src 'none'" in csp and not re.search(r'(src|href)="', html)


def test_french_page_is_display_only_translation():
    html = bs.build("C", FOLLOWUP["C"])
    data = embedded(html)
    assert data["lang"] == "fr" and '<html lang="fr">' in html and "Énoncé (traduction française)" in html
    fr = bs.french_statements()
    for r in data["sheets"]["qwen_gap"]["rows"]:
        assert r["statement_fr"] == fr[r["statement"]] and r["statement_fr"] != r["statement"]
    # the canonical English taxonomy is what the page stores and exports
    assert data["taxonomy"]["relations"] == ui.RELATIONS and data["taxonomy"]["modes"] == ui.MODES
    english = bs.build("A", FOLLOWUP["A"])
    assert "lang" not in embedded(english) and '<html lang="en">' in english
    assert all("statement_fr" not in r for r in embedded(english)["sheets"]["qwen_gap"]["rows"])


def test_verify_export_accepts_followup_sheets(tmp_path):
    rows = ui.read_sheet(ui.FOLLOWUP_SHEETS[("qwen_gap", "C")])
    for r in rows:
        r.update(relation="mixed_or_conditional", mode="analytical_exposition", confidence_1to3="2")
    p = tmp_path / "qwen_gap_sheet_C.completed.csv"
    p.write_text(ui.to_csv(rows), encoding="utf-8", newline="")
    assert ve.verify("qwen_gap", "C", p) == []
    assert ve.verify("qwen_gap", "A", p)  # A's frozen order differs


def test_prior_c_translations_reused_verbatim_and_rest_researcher_reviewed():
    root = ui.FOLLOWUP_FR.parent
    with ui.FOLLOWUP_FR.open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    with (root / "prior_C_translation_reuse_reference.csv").open(newline="", encoding="utf-8") as f:
        prior = {r["statement"]: r["statement_fr"] for r in csv.DictReader(f)}
    assert len(rows) == 33 and len(prior) == 14
    reused = [r for r in rows if r["translation_provenance"] == "reused_from_human_C_retry150_FR_completed"]
    review = [r for r in rows if r["translation_provenance"] == "researcher_reviewed_2026-11-04"]
    assert len(reused) == 14 and len(review) == 19
    assert {r["statement"] for r in reused} == set(prior)
    assert all(r["statement_fr"] == prior[r["statement"]] for r in reused)  # exact, not re-paraphrased
    assert all(r["statement_fr"].strip() for r in review)  # no unreviewed draft or blank translation remains
    # the page shows exactly these translations; translation reuse is not annotation reuse: no labels embedded
    data = embedded(bs.build("C", FOLLOWUP["C"]))
    fr = {r["statement"]: r["statement_fr"] for r in rows}
    page_rows = data["sheets"]["qwen_gap"]["rows"]
    assert all(r["statement_fr"] == fr[r["statement"]] for r in page_rows)
    assert all(set(r) == {"sample_id", "statement", "response", "statement_fr"} for r in page_rows)


@pytest.mark.parametrize("annotator", ["A", "C"])
def test_followup_pages_hold_142_unlabelled_rows(annotator):
    data = embedded(bs.build(annotator, FOLLOWUP[annotator]))
    rows = data["sheets"]["qwen_gap"]["rows"]
    assert len(rows) == 142 and len({r["sample_id"] for r in rows}) == 142
    source = ui.read_sheet(ui.FOLLOWUP_SHEETS[("qwen_gap", annotator)])
    assert all(not (r["relation"] or r["mode"] or r["confidence_1to3"] or r["notes"]) for r in source)
