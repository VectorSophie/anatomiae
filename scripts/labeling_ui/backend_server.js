"use strict";
// Backend for the local-server mode (scripts/labeling_ui/server.py). Same origin only.
(() => {
  const q = (a, n) => `audit=${encodeURIComponent(a)}&annotator=${encodeURIComponent(n)}`;
  async function call(path, body) {
    const opts = body === undefined ? {} :
      { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) };
    const res = await fetch(path, opts);
    return { ok: res.ok, data: await res.json() };
  }
  const NAMES = { olmo: "Audit 1", qwen: "Audit 2" };
  window.Backend = {
    intro: "Choose your audit and your annotator letter. Label independently; do not open the other annotator's session.",
    draftFiles: false,
    sessions: () => [["olmo", "A"], ["olmo", "C"], ["qwen", "A"], ["qwen", "C"]].map(([audit, annotator]) => ({ audit, annotator })),
    sessionLabel: (a, n) => `${NAMES[a]} · Annotator ${n}`,
    taxonomy: async () => (await call("/api/taxonomy")).data,
    loadSheet: async (a, n) => {
      const r = await call(`/api/sheet?${q(a, n)}`);
      return r.ok ? { ok: true, ...r.data } : { ok: false, message: r.data.error };
    },
    loadDraft: async (a, n) => (await call(`/api/draft?${q(a, n)}`)).data,
    saveDraft: async (a, n, draft) => {
      const r = await call(`/api/draft?${q(a, n)}`, draft);
      return { ok: r.ok, message: r.data.error };
    },
    downloadDraft: () => {},
    exportCsv: async (a, n, _rows, draft) => {
      const r = await call(`/api/export?${q(a, n)}`, draft);
      if (!r.ok) return { ok: false, problems: r.data.problems || [r.data.error] };
      return { ok: true, lines: [
        { text: `Valid. Wrote ${r.data.rows} rows to ${r.data.path}. The frozen sheet was NOT changed. To submit, replace it yourself after checking:` },
        { text: r.data.replace_with, code: true }] };
    },
  };
})();
