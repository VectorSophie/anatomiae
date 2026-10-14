"use strict";
// Labeling UI core. Storage/export goes through window.Backend, which is either
// backend_server.js (local server) or backend_standalone.js (offline single file).
// No suggestions, no prefilling; rows are shown in the sheet's own order.

const KEYS = {
  relation: ["q", "w", "e", "r", "t"],
  mode: ["a", "s", "d", "f", "g", "h", "j"],
  confidence_1to3: ["1", "2", "3"],
};
const state = { audit: null, annotator: null, rows: [], digest: null, labels: {}, i: 0, tax: null, timer: null };
const $ = (id) => document.getElementById(id);
const B = window.Backend;

function complete(sid) {
  const l = state.labels[sid] || {};
  return Boolean(l.relation && l.mode && l.confidence_1to3);
}

function buildButtons(container, field, values) {
  container.textContent = "";
  values.forEach((v, n) => {
    const b = document.createElement("button");
    const k = document.createElement("kbd");
    k.textContent = KEYS[field][n].toUpperCase();
    b.append(k, document.createTextNode(v));
    b.dataset.value = v;
    b.addEventListener("click", () => setField(field, v));
    container.append(b);
  });
}

function setField(field, value) {
  const sid = state.rows[state.i].sample_id;
  state.labels[sid] = { ...(state.labels[sid] || {}), [field]: value };
  render();
  scheduleSave();
}

function render() {
  const row = state.rows[state.i];
  const l = state.labels[row.sample_id] || {};
  $("pos").textContent = `${state.i + 1} / ${state.rows.length}`;
  $("sid").textContent = row.sample_id;
  $("statement").textContent = row.statement;
  $("response").textContent = row.response;
  $("response").scrollTop = 0;
  for (const [field, id] of [["relation", "relation"], ["mode", "mode"], ["confidence_1to3", "confidence"]]) {
    for (const b of $(id).children) b.setAttribute("aria-pressed", String(b.dataset.value === l[field]));
  }
  if (document.activeElement !== $("notes")) $("notes").value = l.notes || "";
  $("done").textContent = `${state.rows.filter((r) => complete(r.sample_id)).length} / ${state.rows.length}`;
}

function scheduleSave() {
  $("saved").textContent = "saving…";
  clearTimeout(state.timer);
  state.timer = setTimeout(async () => {
    const r = await B.saveDraft(state.audit, state.annotator, { digest: state.digest, labels: state.labels });
    $("saved").textContent = r.ok ? "draft saved" : `save failed: ${r.message}`;
  }, 300);
}

function go(i) {
  state.i = Math.max(0, Math.min(state.rows.length - 1, i));
  render();
}

function nextIncomplete() {
  for (let k = 1; k <= state.rows.length; k++) {
    const j = (state.i + k) % state.rows.length;
    if (!complete(state.rows[j].sample_id)) return go(j);
  }
  go(state.i);
}

function applyDraft(draft, warn) {
  if (draft && draft.digest && draft.digest !== state.digest) {
    warn.push("A draft exists but was made for different sheet contents; it was not loaded.");
    return false;
  }
  if (draft && draft.labels) {
    const ids = new Set(state.rows.map((r) => r.sample_id));
    state.labels = Object.fromEntries(Object.entries(draft.labels).filter(([sid]) => ids.has(sid)));
  }
  return true;
}

async function start(audit, annotator) {
  state.audit = audit;
  state.annotator = annotator;
  state.tax = await B.taxonomy();
  const sheet = await B.loadSheet(audit, annotator);
  if (!sheet.ok) return alert(sheet.message);
  state.rows = sheet.rows;
  state.digest = sheet.digest;
  const warn = [];
  if (sheet.source_has_labels) {
    warn.push(`The frozen sheet already contains labels in ${sheet.source_has_labels} rows. They are not shown or loaded here.`);
  }
  applyDraft(await B.loadDraft(audit, annotator), warn);
  $("warning").hidden = warn.length === 0;
  $("warning").textContent = warn.join(" ");
  $("session").textContent = B.sessionLabel(audit, annotator);
  buildButtons($("relation"), "relation", state.tax.relations);
  buildButtons($("mode"), "mode", state.tax.modes);
  buildButtons($("confidence"), "confidence_1to3", state.tax.confidence);
  $("draftFiles").hidden = !B.draftFiles;
  $("start").hidden = true;
  $("label").hidden = false;
  const firstOpen = state.rows.findIndex((r) => !complete(r.sample_id));
  go(firstOpen === -1 ? 0 : firstOpen);
}

function showResult(lines, problems) {
  const out = $("exportResult");
  out.textContent = "";
  for (const line of lines) {
    const el = document.createElement(line.code ? "pre" : "p");
    el.textContent = line.text;
    out.append(el);
  }
  if (problems && problems.length) {
    const ul = document.createElement("ul");
    for (const msg of problems) {
      const li = document.createElement("li");
      li.textContent = msg;
      ul.append(li);
    }
    out.append(ul);
  }
}

async function exportCsv() {
  clearTimeout(state.timer);
  await B.saveDraft(state.audit, state.annotator, { digest: state.digest, labels: state.labels });
  const r = await B.exportCsv(state.audit, state.annotator, state.rows,
    { digest: state.digest, labels: state.labels }, state.tax);
  if (r.ok) showResult(r.lines);
  else showResult([{ text: `Not exported: ${r.problems.length} problem(s).` }], r.problems.slice(0, 50));
}

function renderStart() {
  const box = $("choices");
  box.textContent = "";
  for (const s of B.sessions()) {
    const b = document.createElement("button");
    b.textContent = B.sessionLabel(s.audit, s.annotator);
    b.addEventListener("click", () => start(s.audit, s.annotator));
    box.append(b);
  }
  $("intro").textContent = B.intro;
}

renderStart();
$("prev").addEventListener("click", () => go(state.i - 1));
$("next").addEventListener("click", () => go(state.i + 1));
$("nextOpen").addEventListener("click", nextIncomplete);
$("export").addEventListener("click", exportCsv);
$("leave").addEventListener("click", () => location.reload());
$("notes").addEventListener("input", (e) => {
  const sid = state.rows[state.i].sample_id;
  state.labels[sid] = { ...(state.labels[sid] || {}), notes: e.target.value };
  scheduleSave();
});
$("saveDraftFile").addEventListener("click", () =>
  B.downloadDraft(state.audit, state.annotator, { digest: state.digest, labels: state.labels }));
$("loadDraftFile").addEventListener("change", async (e) => {
  const file = e.target.files[0];
  if (!file) return;
  const warn = [];
  let draft;
  try { draft = JSON.parse(await file.text()); } catch { return alert("Not a draft file."); }
  if (draft.audit !== state.audit || draft.annotator !== state.annotator) {
    return alert("This draft belongs to a different audit or annotator; not loaded.");
  }
  if (applyDraft(draft, warn)) {
    scheduleSave();
    render();
  }
  if (warn.length) alert(warn.join(" "));
  e.target.value = "";
});

document.addEventListener("keydown", (e) => {
  if ($("label").hidden || e.ctrlKey || e.metaKey || e.altKey) return;
  if (document.activeElement === $("notes")) {
    if (e.key === "Escape") $("notes").blur();
    return;
  }
  const k = e.key.toLowerCase();
  if (k === "arrowleft") return go(state.i - 1);
  if (k === "arrowright") return go(state.i + 1);
  if (k === "enter") { e.preventDefault(); return nextIncomplete(); }
  if (k === "n") { e.preventDefault(); return $("notes").focus(); }
  const values = { relation: state.tax.relations, mode: state.tax.modes, confidence_1to3: state.tax.confidence };
  for (const field of Object.keys(KEYS)) {
    const n = KEYS[field].indexOf(k);
    if (n !== -1) return setField(field, values[field][n]);
  }
});
