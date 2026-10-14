"use strict";
// Local labeling UI. Talks only to the same-origin local server; no external
// requests, no suggestions, no prefilling. Rows are shown in the sheet's own order.

const KEYS = {
  relation: ["q", "w", "e", "r", "t"],
  mode: ["a", "s", "d", "f", "g", "h", "j"],
  confidence_1to3: ["1", "2", "3"],
};
const state = { audit: null, annotator: null, rows: [], digest: null, labels: {}, i: 0, tax: null, timer: null };

const $ = (id) => document.getElementById(id);
const q = () => `audit=${encodeURIComponent(state.audit)}&annotator=${encodeURIComponent(state.annotator)}`;

async function api(path, body) {
  const opts = body === undefined ? {} :
    { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) };
  const res = await fetch(path, opts);
  return { ok: res.ok, status: res.status, data: await res.json() };
}

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
    const r = await api(`/api/draft?${q()}`, { digest: state.digest, labels: state.labels });
    $("saved").textContent = r.ok ? "draft saved" : `save failed: ${r.data.error}`;
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

async function start(audit, annotator) {
  state.audit = audit;
  state.annotator = annotator;
  state.tax = (await api("/api/taxonomy")).data;
  const sheet = await api(`/api/sheet?${q()}`);
  if (!sheet.ok) return alert(sheet.data.error);
  state.rows = sheet.data.rows;
  state.digest = sheet.data.digest;
  const draft = (await api(`/api/draft?${q()}`)).data;
  const warn = [];
  if (sheet.data.source_has_labels) {
    warn.push(`The frozen sheet already contains labels in ${sheet.data.source_has_labels} rows. They are not shown or loaded here.`);
  }
  if (draft.digest && draft.digest !== state.digest) {
    warn.push("A draft exists but was made for different sheet contents; it was not loaded.");
  } else if (draft.labels) {
    state.labels = draft.labels;
  }
  $("warning").hidden = warn.length === 0;
  $("warning").textContent = warn.join(" ");
  $("session").textContent = `${audit === "olmo" ? "Audit 1" : "Audit 2"} · Annotator ${annotator}`;
  buildButtons($("relation"), "relation", state.tax.relations);
  buildButtons($("mode"), "mode", state.tax.modes);
  buildButtons($("confidence"), "confidence_1to3", state.tax.confidence);
  $("start").hidden = true;
  $("label").hidden = false;
  const firstOpen = state.rows.findIndex((r) => !complete(r.sample_id));
  go(firstOpen === -1 ? 0 : firstOpen);
}

async function exportCsv() {
  clearTimeout(state.timer);
  await api(`/api/draft?${q()}`, { digest: state.digest, labels: state.labels });
  const r = await api(`/api/export?${q()}`, { digest: state.digest, labels: state.labels });
  const out = $("exportResult");
  out.textContent = "";
  if (r.ok) {
    const p = document.createElement("p");
    p.textContent = `Valid. Wrote ${r.data.rows} rows to ${r.data.path}. The frozen sheet was NOT changed. ` +
      "To submit, replace it yourself after checking:";
    const pre = document.createElement("pre");
    pre.textContent = r.data.replace_with;
    out.append(p, pre);
  } else {
    const p = document.createElement("p");
    p.textContent = `Not exported (${r.data.n_problems || (r.data.problems || []).length || r.data.error} problem(s)):`;
    const ul = document.createElement("ul");
    for (const msg of r.data.problems || [r.data.error]) {
      const li = document.createElement("li");
      li.textContent = msg;
      ul.append(li);
    }
    out.append(p, ul);
  }
}

document.querySelectorAll("#start button").forEach((b) =>
  b.addEventListener("click", () => start(b.dataset.audit, b.dataset.annotator)));
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
