"""Build two offline, single-file labeling pages: one per annotator.

  uv run python scripts/labeling_ui/build_standalone.py
  -> artifacts/labeling/standalone/annotator_A.html   (Audit 1 + Audit 2, annotator A's sheets only)
     artifacts/labeling/standalone/annotator_C.html   (annotator C's sheets only)

Each file embeds only sample_id/statement/response of that annotator's two blind
sheets, in the sheets' own row order. There are no keys, no hidden metadata and no
other annotator's data. It works opened directly from disk (file://) with no
server. A meta Content-Security-Policy allows only the page's own hashed inline
script and style and sets connect-src 'none', so the page cannot make network
requests. Drafts stay in that browser's localStorage, with an optional draft-file
backup. Export is a CSV download, checked with verify_export.py before it
replaces any frozen sheet. Standard library only; deterministic output.
"""

from __future__ import annotations

import base64
import hashlib
import json
import re
from pathlib import Path

import server as ui

UI_DIR = Path(__file__).resolve().parent
OUT_DIR = ui.ROOT / "artifacts/labeling/standalone"


def _sha(text: str) -> str:
    return "sha256-" + base64.b64encode(hashlib.sha256(text.encode("utf-8")).digest()).decode()


def build(annotator: str) -> str:
    sheets = {}
    for audit in ("olmo", "qwen"):
        rows = ui.read_sheet(ui.SHEETS[(audit, annotator)])
        sheets[audit] = {"rows": ui.blind_rows(rows), "digest": ui.sheet_digest(rows)}
    data = {"annotator": annotator, "columns": ui.COLUMNS, "sheets": sheets,
            "taxonomy": {"relations": ui.RELATIONS, "modes": ui.MODES, "confidence": ui.CONFIDENCE}}
    data_json = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    css = (UI_DIR / "app.css").read_text(encoding="utf-8")
    js = (UI_DIR / "backend_standalone.js").read_text(encoding="utf-8") + "\n" + \
        (UI_DIR / "app.js").read_text(encoding="utf-8")
    assert "</script" not in js.lower() and "</style" not in css.lower()
    csp = (f"default-src 'none'; script-src '{_sha(js)}'; style-src '{_sha(css)}'; connect-src 'none'; "
           "img-src 'none'; font-src 'none'; media-src 'none'; object-src 'none'; frame-src 'none'; "
           "worker-src 'none'; form-action 'none'; base-uri 'none'")
    html = (UI_DIR / "index.html").read_text(encoding="utf-8")
    html = html.replace('<meta charset="utf-8">',
                        f'<meta charset="utf-8">\n<meta http-equiv="Content-Security-Policy" content="{csp}">')
    html = html.replace('<link rel="stylesheet" href="/app.css">', f"<style>{css}</style>")
    html = html.replace("<title>Commitment audit labeling</title>",
                        f"<title>Commitment audit labeling — Annotator {annotator}</title>")
    scripts = '<script src="/backend_server.js"></script>\n<script src="/app.js"></script>'
    assert scripts in html
    html = html.replace(scripts, f'<script type="application/json" id="sheet-data">{data_json}</script>\n'
                                 f"<script>{js}</script>")
    assert not re.search(r'(src|href)="(?!#)', html), "external reference left in standalone page"
    return html


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for annotator in ("A", "C"):
        path = OUT_DIR / f"annotator_{annotator}.html"
        path.write_text(build(annotator), encoding="utf-8")
        print(f"wrote {path.relative_to(ui.ROOT)} ({path.stat().st_size // 1024} KiB)")


if __name__ == "__main__":
    main()
