"use strict";
// Backend for the offline single-file mode (built by build_standalone.py).
// Data is embedded in <script type="application/json" id="sheet-data">. Drafts live in this
// browser's localStorage (plus optional draft-file backup); export is a file download.
// No network access: the page's CSP sets connect-src 'none'.
(() => {
  const DATA = JSON.parse(document.getElementById("sheet-data").textContent);
  const NAMES = { olmo: "Audit 1", qwen: "Audit 2" };
  const storeKey = (a, n) => `anatomiae-commitment-audit:${a}:${n}:${DATA.sheets[a].digest}`;

  function csvField(v) {
    // Python csv QUOTE_MINIMAL: quote if the field contains the delimiter, quote char or a line break.
    return /[",\r\n]/.test(v) ? `"${v.replace(/"/g, '""')}"` : v;
  }
  function toCsv(rows) {
    const cols = DATA.columns;
    return [cols.join(","), ...rows.map((r) => cols.map((c) => csvField(r[c])).join(","))].join("\n") + "\n";
  }
  function download(name, text, type) {
    const url = URL.createObjectURL(new Blob([text], { type }));
    const a = document.createElement("a");
    a.href = url;
    a.download = name;
    document.body.append(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  window.Backend = {
    intro: `This file is for Annotator ${DATA.annotator} only. Label independently. ` +
      "Progress is kept in this browser on this computer; use “Download draft backup” to keep a copy.",
    draftFiles: true,
    sessions: () => Object.keys(DATA.sheets).map((audit) => ({ audit, annotator: DATA.annotator })),
    sessionLabel: (a, n) => `${NAMES[a]} · Annotator ${n}`,
    taxonomy: async () => DATA.taxonomy,
    loadSheet: async (a) => ({ ok: true, rows: DATA.sheets[a].rows, digest: DATA.sheets[a].digest, source_has_labels: 0 }),
    loadDraft: async (a, n) => {
      try { return JSON.parse(localStorage.getItem(storeKey(a, n)) || "{}"); } catch { return {}; }
    },
    saveDraft: async (a, n, draft) => {
      try {
        localStorage.setItem(storeKey(a, n), JSON.stringify(draft));
        return { ok: true };
      } catch (err) {
        return { ok: false, message: "browser storage unavailable — use “Download draft backup”" };
      }
    },
    downloadDraft: (a, n, draft) =>
      download(`draft_${a}_annotator_${n}.json`, JSON.stringify({ audit: a, annotator: n, ...draft }, null, 1),
        "application/json"),
    exportCsv: async (a, n, rows, draft, tax) => {
      const problems = [];
      const out = rows.map((r) => {
        const l = draft.labels[r.sample_id] || {};
        const row = { sample_id: r.sample_id, statement: r.statement, response: r.response,
          relation: l.relation || "", mode: l.mode || "", confidence_1to3: l.confidence_1to3 || "", notes: l.notes || "" };
        if (!tax.relations.includes(row.relation)) problems.push(`${r.sample_id}: relation missing or invalid`);
        if (!tax.modes.includes(row.mode)) problems.push(`${r.sample_id}: mode missing or invalid`);
        if (!tax.confidence.includes(row.confidence_1to3)) problems.push(`${r.sample_id}: confidence missing or invalid`);
        return row;
      });
      if (out.length !== DATA.sheets[a].rows.length) problems.push("row count changed");
      if (problems.length) return { ok: false, problems };
      const name = `${a}_sheet_${n}.completed.csv`;
      download(name, toCsv(out), "text/csv;charset=utf-8");
      return { ok: true, lines: [
        { text: `Valid (${out.length} rows). Downloaded ${name}. Send that file to the researcher; ` +
          "it is checked against the frozen sheet before it replaces anything:" },
        { text: `uv run python scripts/labeling_ui/verify_export.py ${a} ${n} path/to/${name}`, code: true }] };
    },
  };
})();
