"""Strictly local labeling UI for the two frozen commitment audits.

A convenience layer only. It never writes, reorders, filters or prefills the
frozen sheets, never reads any key.csv, and makes no network requests.

  uv run python scripts/labeling_ui/server.py        # then open http://127.0.0.1:8765/

What it reads: exactly the four whitelisted blind sheets below. It sends the
browser only sample_id, statement and response, in the sheet's own row order.
What it writes:
  artifacts/labeling/_ui_drafts/<audit>_<annotator>.json     autosaved progress
  artifacts/labeling/_ui_exports/<audit>_sheet_<annotator>.completed.csv
                                                           only on explicit Export, only if valid
Both directories are gitignored. Replacing a frozen sheet with an export is a
manual step the annotator takes themselves (see README.md).

Standard library only.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parents[2]
UI_DIR = Path(__file__).resolve().parent
SHEETS = {
    ("olmo", "A"): ROOT / "artifacts/labeling/commitment_audit_v1/sheet_A.csv",
    ("olmo", "C"): ROOT / "artifacts/labeling/commitment_audit_v1/sheet_C.csv",
    ("qwen", "A"): ROOT / "artifacts/labeling/commitment_audit_qwen_v1/sheet_A.csv",
    ("qwen", "C"): ROOT / "artifacts/labeling/commitment_audit_qwen_v1/sheet_C.csv",
}
DRAFTS = ROOT / "artifacts/labeling/_ui_drafts"
EXPORTS = ROOT / "artifacts/labeling/_ui_exports"
COLUMNS = ["sample_id", "statement", "response", "relation", "mode", "confidence_1to3", "notes"]
LABEL_COLUMNS = ["relation", "mode", "confidence_1to3", "notes"]
RELATIONS = ["supports", "opposes", "mixed_or_conditional", "neutral_or_no_position", "unclear"]
MODES = ["explicit_stance", "analytical_exposition", "voiced_continuation", "refusal", "empty", "incomplete", "other"]
CONFIDENCE = ["1", "2", "3"]
CSP = ("default-src 'none'; script-src 'self'; style-src 'self'; connect-src 'self'; img-src 'none'; "
       "font-src 'none'; form-action 'none'; base-uri 'none'; frame-ancestors 'none'")


def read_sheet(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != COLUMNS:
            raise ValueError(f"{path.name}: unexpected columns {reader.fieldnames}")
        return list(reader)


def sheet_digest(rows: list[dict[str, str]]) -> str:
    h = hashlib.sha256()
    for r in rows:
        h.update(json.dumps([r["sample_id"], r["statement"], r["response"]]).encode("utf-8"))
    return h.hexdigest()


def blind_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """Only what the annotator may see. Existing label columns are never sent."""
    return [{"sample_id": r["sample_id"], "statement": r["statement"], "response": r["response"]} for r in rows]


def validate_export(source: list[dict[str, str]], labels: dict[str, dict[str, str]]) -> tuple[list[str], list[dict]]:
    """Merge labels onto the frozen rows (original order) and validate. Returns (problems, rows)."""
    problems: list[str] = []
    ids = [r["sample_id"] for r in source]
    if len(set(ids)) != len(ids):
        problems.append("source sheet has duplicate sample_ids")
    unknown = set(labels) - set(ids)
    if unknown:
        problems.append(f"labels for unknown sample_ids: {sorted(unknown)[:5]}")
    out = []
    for r in source:
        lab = labels.get(r["sample_id"], {})
        row = {"sample_id": r["sample_id"], "statement": r["statement"], "response": r["response"],
               "relation": str(lab.get("relation", "")), "mode": str(lab.get("mode", "")),
               "confidence_1to3": str(lab.get("confidence_1to3", "")), "notes": str(lab.get("notes", ""))}
        if row["relation"] not in RELATIONS:
            problems.append(f"{r['sample_id']}: relation missing or invalid ({row['relation']!r})")
        if row["mode"] not in MODES:
            problems.append(f"{r['sample_id']}: mode missing or invalid ({row['mode']!r})")
        if row["confidence_1to3"] not in CONFIDENCE:
            problems.append(f"{r['sample_id']}: confidence missing or invalid ({row['confidence_1to3']!r})")
        out.append(row)
    if len(out) != len(source):
        problems.append("row count changed")
    return problems, out


def to_csv(rows: list[dict[str, str]]) -> str:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=COLUMNS, lineterminator="\n")
    w.writeheader()
    w.writerows(rows)
    return buf.getvalue()


def verify_written(path: Path, source: list[dict[str, str]]) -> list[str]:
    """Re-read an export and check it against the frozen sheet exactly."""
    back = read_sheet(path)
    problems = []
    if len(back) != len(source):
        problems.append("re-read export: row count differs")
    for a, b in zip(source, back, strict=False):
        for col in ("sample_id", "statement", "response"):
            if a[col] != b[col]:
                problems.append(f"re-read export: {col} differs for {a['sample_id']}")
    return problems


def display(path: Path) -> str:
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def key_of(query: dict) -> tuple[str, str] | None:
    audit = (query.get("audit") or [""])[0]
    annotator = (query.get("annotator") or [""])[0]
    return (audit, annotator) if (audit, annotator) in SHEETS else None


class Handler(BaseHTTPRequestHandler):
    server_version = "anatomiae-labeling-ui"

    def log_message(self, fmt, *args):  # no request logging to anywhere but the terminal
        print("[ui]", fmt % args)

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Content-Security-Policy", CSP)
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code: int, obj) -> None:
        self._send(code, json.dumps(obj).encode("utf-8"), "application/json; charset=utf-8")

    def _local_only(self) -> bool:
        if self.client_address[0] not in ("127.0.0.1", "::1"):
            self._json(403, {"error": "local only"})
            return False
        host = (self.headers.get("Host") or "").split(":")[0]
        if host not in ("127.0.0.1", "localhost"):
            self._json(403, {"error": "bad host"})  # DNS-rebinding guard
            return False
        return True

    def do_GET(self):
        if not self._local_only():
            return
        url = urlparse(self.path)
        q = parse_qs(url.query)
        if url.path == "/":
            return self._send(200, (UI_DIR / "index.html").read_bytes(), "text/html; charset=utf-8")
        if url.path in ("/app.js", "/app.css", "/backend_server.js"):
            ctype = "text/javascript" if url.path.endswith(".js") else "text/css"
            return self._send(200, (UI_DIR / url.path[1:]).read_bytes(), f"{ctype}; charset=utf-8")
        if url.path == "/api/taxonomy":
            return self._json(200, {"relations": RELATIONS, "modes": MODES, "confidence": CONFIDENCE})
        if url.path == "/api/sheet":
            k = key_of(q)
            if not k:
                return self._json(404, {"error": "unknown audit/annotator"})
            rows = read_sheet(SHEETS[k])
            prelabelled = sum(any(r[c].strip() for c in ("relation", "mode", "confidence_1to3")) for r in rows)
            return self._json(200, {"audit": k[0], "annotator": k[1], "rows": blind_rows(rows),
                                    "digest": sheet_digest(rows), "source_has_labels": prelabelled})
        if url.path == "/api/draft":
            k = key_of(q)
            if not k:
                return self._json(404, {"error": "unknown audit/annotator"})
            p = DRAFTS / f"{k[0]}_{k[1]}.json"
            return self._json(200, json.loads(p.read_text(encoding="utf-8")) if p.exists() else {})
        return self._json(404, {"error": "not found"})

    def do_POST(self):
        if not self._local_only():
            return
        url = urlparse(self.path)
        k = key_of(parse_qs(url.query))
        if not k:
            return self._json(404, {"error": "unknown audit/annotator"})
        if self.headers.get("Content-Type", "").split(";")[0] != "application/json":
            return self._json(415, {"error": "json only"})
        length = int(self.headers.get("Content-Length") or 0)
        if length > 5_000_000:
            return self._json(413, {"error": "too large"})
        body = json.loads(self.rfile.read(length) or b"{}")
        source = read_sheet(SHEETS[k])
        if body.get("digest") != sheet_digest(source):
            return self._json(409, {"error": "the frozen sheet changed since this session was loaded"})
        labels = body.get("labels") or {}
        if not isinstance(labels, dict):
            return self._json(400, {"error": "labels must be an object"})
        if url.path == "/api/draft":
            DRAFTS.mkdir(parents=True, exist_ok=True)
            p = DRAFTS / f"{k[0]}_{k[1]}.json"
            tmp = p.with_suffix(".tmp")
            tmp.write_text(json.dumps({"digest": body["digest"], "labels": labels}, indent=1), encoding="utf-8")
            tmp.replace(p)
            return self._json(200, {"saved": len(labels)})
        if url.path == "/api/export":
            problems, rows = validate_export(source, labels)
            if problems:
                return self._json(422, {"valid": False, "problems": problems[:50], "n_problems": len(problems)})
            EXPORTS.mkdir(parents=True, exist_ok=True)
            out = EXPORTS / f"{k[0]}_sheet_{k[1]}.completed.csv"
            out.write_text(to_csv(rows), encoding="utf-8", newline="")
            problems = verify_written(out, source)
            if problems:
                out.unlink()
                return self._json(500, {"valid": False, "problems": problems})
            return self._json(200, {"valid": True, "path": display(out), "rows": len(rows),
                                    "replace_with": f"cp {display(out)} {display(SHEETS[k])}"})
        return self._json(404, {"error": "not found"})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()
    for k, p in SHEETS.items():
        assert p.exists(), f"missing frozen sheet {p}"
        assert re.search(r"/sheet_[AC]\.csv$", str(p)), k
    httpd = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"labeling UI on http://127.0.0.1:{args.port}/  (local only; Ctrl+C to stop)")
    httpd.serve_forever()


if __name__ == "__main__":
    main()
