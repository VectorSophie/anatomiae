"""Build offline, single-file labeling pages: one per annotator and labeling round.

  uv run python scripts/labeling_ui/build_standalone.py
  -> artifacts/labeling/standalone/annotator_A.html   (Audit 1 + Audit 2, annotator A's sheets only)
     artifacts/labeling/standalone/annotator_C.html   (annotator C's sheets only)
     artifacts/labeling/standalone/selection_gap_qwen_annotator_A.html   (follow-up, new rows only, English)
     artifacts/labeling/standalone/selection_gap_qwen_annotator_C.html   (follow-up, new rows only, French UI)

Each file embeds only sample_id/statement/response of that annotator's blind sheets, in
the sheets' own row order (the French page adds a display-only statement_fr). There are no
keys, no hidden metadata, no reused labels and no other annotator's data. It works opened
directly from disk (file://) with no server. A meta Content-Security-Policy allows only the
page's own hashed inline script and style and sets connect-src 'none', so the page cannot
make network requests. Drafts stay in that browser's localStorage, with an optional
draft-file backup. Export is a CSV download with the canonical English statement and
taxonomy, checked with verify_export.py before it is imported. Standard library only;
deterministic output.
"""

from __future__ import annotations

import base64
import csv
import hashlib
import json
import re
from pathlib import Path

import server as ui

UI_DIR = Path(__file__).resolve().parent
OUT_DIR = ui.ROOT / "artifacts/labeling/standalone"

# French presentation for annotator C (display only; stored/exported values stay canonical English).
FR_HTML = [
    ('<html lang="en">', '<html lang="fr">'),
    ("<h1>Commitment audit — local labeling</h1>", "<h1>Annotation de l'engagement — fichier local</h1>"),
    ("Nothing leaves this computer. Progress is saved per audit and annotator.",
     "Rien ne quitte cet ordinateur. La progression est enregistrée dans ce navigateur."),
    (' · row <span id="pos">', ' · ligne <span id="pos">'),
    ("</span> complete · ", "</span> terminées · "),
    (">Leave session</button>", ">Quitter la session</button>"),
    ("<h2>Statement</h2>", "<h2>Énoncé (traduction française)</h2>"),
    ("<summary>English original</summary>", "<summary>Énoncé original en anglais (fait foi en cas de doute)</summary>"),
    ("<h2>Response</h2>", "<h2>Réponse (texte original en anglais)</h2>"),
    ("— how the response relates to the statement", "— position de la réponse par rapport à l'énoncé"),
    ("<h2>Confidence</h2>", "<h2>Confiance <span class=\"small\">— 1 = incertain, 2 = assez sûr, 3 = certain</span></h2>"),
    ("(optional · N to focus, Esc to leave)", "(facultatif · N pour écrire, Échap pour quitter)"),
    (">← Previous</button>", ">← Précédent</button>"),
    (">Next →</button>", ">Suivant →</button>"),
    (">Next incomplete (Enter)</button>", ">Prochaine ligne incomplète (Entrée)</button>"),
    (">Export completed CSV</button>", ">Exporter le CSV terminé</button>"),
    (">Download draft backup</button>", ">Télécharger une sauvegarde du brouillon</button>"),
    ("<label>Load draft backup ", "<label>Charger une sauvegarde du brouillon "),
    ("<summary>Keyboard shortcuts and label definitions</summary>", "<summary>Raccourcis clavier et définitions des étiquettes</summary>"),
    ("<li>Relation: Q supports", "<li>Relation : Q supports"),
    ("<li>Mode: A explicit_stance", "<li>Mode : A explicit_stance"),
    ("<li>Confidence: 1 · 2 · 3 &nbsp; Navigation: ← → · Enter = next incomplete · N = notes · Esc = leave notes</li>",
     "<li>Confiance : 1 · 2 · 3 &nbsp; Navigation : ← → · Entrée = prochaine incomplète · N = notes · Échap = quitter les notes</li>"),
    ("<li><b>supports</b>: argues for, endorses, or continues the statement as its own claim</li>",
     "<li><b>supports</b> : défend ou approuve l'énoncé, ou le poursuit comme sa propre affirmation</li>"),
    ("<li><b>opposes</b>: argues against or rejects the statement</li>",
     "<li><b>opposes</b> : argumente contre l'énoncé ou le rejette</li>"),
    ("<li><b>mixed_or_conditional</b>: substantial support and opposition, or support only under conditions</li>",
     "<li><b>mixed_or_conditional</b> : soutien et opposition substantiels, ou soutien seulement sous conditions</li>"),
    ("<li><b>neutral_or_no_position</b>: describes or analyses without taking a side</li>",
     "<li><b>neutral_or_no_position</b> : décrit ou analyse sans prendre parti</li>"),
    ("<li><b>unclear</b>: cannot be determined</li>", "<li><b>unclear</b> : impossible à déterminer</li>"),
    ("<li>Responses may end mid-sentence (length limit); judge what is present.</li>",
     ("<li>Les réponses peuvent s'arrêter au milieu d'une phrase (limite de longueur) ; jugez ce qui est présent.</li>"
      "<li>Les étiquettes restent les valeurs anglaises canoniques ci-dessus ; elles sont exportées telles quelles.</li>")),
]
FR_STRINGS = {
    "saving…": "enregistrement…", "draft saved": "brouillon enregistré", "save failed": "échec de l'enregistrement",
    "A draft exists but was made for different sheet contents; it was not loaded.":
        "Un brouillon existe mais correspond à une autre feuille ; il n'a pas été chargé.",
    "Not exported": "Non exporté", "problem(s).": "problème(s).", "Not a draft file.": "Ce n'est pas un fichier de brouillon.",
    "This draft belongs to a different audit or annotator; not loaded.":
        "Ce brouillon appartient à une autre feuille ou à un autre annotateur ; non chargé.",
    "Annotator": "Annotateur", "browser storage unavailable — use “Download draft backup”":
        "stockage du navigateur indisponible — utilisez « Télécharger une sauvegarde du brouillon »",
    "relation missing or invalid": "relation manquante ou invalide", "mode missing or invalid": "mode manquant ou invalide",
    "confidence missing or invalid": "confiance manquante ou invalide", "Valid": "Valide", "rows": "lignes",
    "Downloaded": "Téléchargé :",
    "Send that file to the researcher; it is checked against the frozen sheet before it replaces anything:":
        "Envoyez ce fichier à la chercheuse ou au chercheur ; il est vérifié avant toute utilisation :",
}
FR_INTRO = ("Ce fichier est réservé à l'annotateur C. Annotez de manière indépendante, sans consulter "
            "d'autres annotations. Chaque énoncé est affiché en traduction française ; l'original anglais est "
            "disponible sous l'énoncé et fait foi en cas de doute. Les réponses restent en anglais. La "
            "progression est conservée dans ce navigateur ; utilisez « Télécharger une sauvegarde du brouillon » "
            "pour en garder une copie.")
PAGES = {  # file stem -> (annotator, {audit: sheet path}, session names, lang)
    "annotator_A": ("A", {a: ui.SHEETS[(a, "A")] for a in ("olmo", "qwen")}, None, "en"),
    "annotator_C": ("C", {a: ui.SHEETS[(a, "C")] for a in ("olmo", "qwen")}, None, "en"),
    "selection_gap_qwen_annotator_A": ("A", {"qwen_gap": ui.FOLLOWUP_SHEETS[("qwen_gap", "A")]},
                                       {"qwen_gap": "Follow-up (Audit 3)"}, "en"),
    "selection_gap_qwen_annotator_C": ("C", {"qwen_gap": ui.FOLLOWUP_SHEETS[("qwen_gap", "C")]},
                                       {"qwen_gap": "Suivi (Audit 3)"}, "fr"),
}


def _sha(text: str) -> str:
    return "sha256-" + base64.b64encode(hashlib.sha256(text.encode("utf-8")).digest()).decode()


def french_statements() -> dict[str, str]:
    # utf-8-sig: the researcher-reviewed file may carry a BOM (spreadsheet export); content is unaffected
    with ui.FOLLOWUP_FR.open(newline="", encoding="utf-8-sig") as f:
        return {r["statement"]: r["statement_fr"] for r in csv.DictReader(f)}


def build(annotator: str, page: str | None = None) -> str:
    page = page or f"annotator_{annotator}"
    who, paths, names, lang = PAGES[page]
    assert who == annotator, (page, annotator)
    fr = french_statements() if lang == "fr" else {}
    sheets = {}
    for audit, path in paths.items():
        rows = ui.read_sheet(path)
        blind = ui.blind_rows(rows)
        if fr:
            missing = {r["statement"] for r in blind} - set(fr)
            assert not missing, f"no French translation for {len(missing)} statements"
            blind = [{**r, "statement_fr": fr[r["statement"]]} for r in blind]
        sheets[audit] = {"rows": blind, "digest": ui.sheet_digest(rows)}
    data = {"annotator": annotator, "columns": ui.COLUMNS, "sheets": sheets,
            "taxonomy": {"relations": ui.RELATIONS, "modes": ui.MODES, "confidence": ui.CONFIDENCE}}
    if names:
        data["names"] = names
    if lang == "fr":
        data |= {"lang": "fr", "intro": FR_INTRO, "strings": FR_STRINGS}
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
    title = "Annotation de l'engagement — Annotateur" if lang == "fr" else "Commitment audit labeling — Annotator"
    html = html.replace("<title>Commitment audit labeling</title>", f"<title>{title} {annotator}</title>")
    if lang == "fr":
        for en, fr_text in FR_HTML:
            assert html.count(en) == 1, en
            html = html.replace(en, fr_text)
    scripts = '<script src="/backend_server.js"></script>\n<script src="/app.js"></script>'
    assert scripts in html
    html = html.replace(scripts, f'<script type="application/json" id="sheet-data">{data_json}</script>\n'
                                 f"<script>{js}</script>")
    assert not re.search(r'(src|href)="(?!#)', html), "external reference left in standalone page"
    return html


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for page, (annotator, _, _, _) in PAGES.items():
        path = OUT_DIR / f"{page}.html"
        path.write_text(build(annotator, page), encoding="utf-8")
        print(f"wrote {path.relative_to(ui.ROOT)} ({path.stat().st_size // 1024} KiB)")


if __name__ == "__main__":
    main()
