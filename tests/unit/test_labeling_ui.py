"""The local labeling UI: never exposes keys or labels, never writes frozen sheets, validates exports."""

import csv
import hashlib
import json
import sys
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "labeling_ui"))

import server as ui

AUDIT_ROOT = Path(__file__).resolve().parents[2] / "artifacts" / "labeling"


@pytest.fixture
def running(tmp_path, monkeypatch):
    monkeypatch.setattr(ui, "DRAFTS", tmp_path / "drafts")
    monkeypatch.setattr(ui, "EXPORTS", tmp_path / "exports")
    before = {k: hashlib.sha256(p.read_bytes()).hexdigest() for k, p in ui.SHEETS.items()}
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), ui.Handler)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}", tmp_path
    httpd.shutdown()
    after = {k: hashlib.sha256(p.read_bytes()).hexdigest() for k, p in ui.SHEETS.items()}
    assert before == after, "a frozen sheet was modified"


def get(url):
    try:
        with urllib.request.urlopen(url) as r:
            return r.status, json.loads(r.read() or b"null") if "json" in r.headers["Content-Type"] else r.read()
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"null")


def post(url, body):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_whitelist_is_sheets_only():
    assert len(ui.SHEETS) == 4
    assert all(p.name in ("sheet_A.csv", "sheet_C.csv") for p in ui.SHEETS.values())


def test_sheet_endpoint_sends_only_blind_fields_in_order(running):
    base, _ = running
    code, data = get(f"{base}/api/sheet?audit=olmo&annotator=A")
    assert code == 200
    source = ui.read_sheet(ui.SHEETS[("olmo", "A")])
    assert [r["sample_id"] for r in data["rows"]] == [r["sample_id"] for r in source]
    assert all(set(r) == {"sample_id", "statement", "response"} for r in data["rows"])
    assert data["rows"][0]["response"] == source[0]["response"]


@pytest.mark.parametrize("path", ["/key.csv", "/../commitment_audit_v1/key.csv",
                                  "/api/sheet?audit=olmo&annotator=key", "/artifacts/labeling/commitment_audit_v1/key.csv",
                                  "/api/sheet?audit=../commitment_audit_v1&annotator=A", "/server.py"])
def test_no_other_files(running, path):
    base, _ = running
    code, _ = get(base + path)
    assert code == 404


def test_csp_blocks_external_connections(running):
    base, _ = running
    with urllib.request.urlopen(base + "/") as r:
        csp = r.headers["Content-Security-Policy"]
    assert "connect-src 'self'" in csp and "default-src 'none'" in csp


def test_incomplete_export_refused_and_complete_export_round_trips(running):
    base, tmp = running
    _, sheet = get(f"{base}/api/sheet?audit=qwen&annotator=C")
    rows, digest = sheet["rows"], sheet["digest"]
    code, res = post(f"{base}/api/export?audit=qwen&annotator=C", {"digest": digest, "labels": {}})
    assert code == 422 and not (tmp / "exports").exists()
    labels = {r["sample_id"]: {"relation": "unclear", "mode": "other", "confidence_1to3": "2", "notes": "a, \"b\"\nc"}
              for r in rows}
    labels[rows[0]["sample_id"]]["relation"] = "not_a_label"
    code, res = post(f"{base}/api/export?audit=qwen&annotator=C", {"digest": digest, "labels": labels})
    assert code == 422 and any("relation" in p for p in res["problems"])
    labels[rows[0]["sample_id"]]["relation"] = "supports"
    code, res = post(f"{base}/api/export?audit=qwen&annotator=C", {"digest": digest, "labels": labels})
    assert code == 200 and res["valid"]
    out = tmp / "exports" / "qwen_sheet_C.completed.csv"
    with out.open(newline="", encoding="utf-8") as f:
        back = list(csv.DictReader(f))
    source = ui.read_sheet(ui.SHEETS[("qwen", "C")])
    assert list(back[0]) == ui.COLUMNS
    assert [(r["sample_id"], r["statement"], r["response"]) for r in back] == \
        [(r["sample_id"], r["statement"], r["response"]) for r in source]
    assert back[0]["relation"] == "supports" and back[1]["notes"] == "a, \"b\"\nc"


def test_drafts_are_per_annotator_and_reject_stale_digest(running):
    base, tmp = running
    _, sheet = get(f"{base}/api/sheet?audit=olmo&annotator=A")
    code, _ = post(f"{base}/api/draft?audit=olmo&annotator=A", {"digest": sheet["digest"], "labels": {"CA001": {"relation": "opposes"}}})
    assert code == 200
    assert (tmp / "drafts" / "olmo_A.json").exists() and not (tmp / "drafts" / "olmo_C.json").exists()
    _, other = get(f"{base}/api/draft?audit=olmo&annotator=C")
    assert other == {}
    code, _ = post(f"{base}/api/draft?audit=olmo&annotator=A", {"digest": "stale", "labels": {}})
    assert code == 409


def test_taxonomy_matches_analysis():
    pytest.importorskip("pandas")
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
    import analyze_commitment_audit as ca

    assert set(ui.RELATIONS) == ca.RELATIONS and set(ui.MODES) == ca.MODES
