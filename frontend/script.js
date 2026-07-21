const API_BASE = "http://localhost:5000/api";

const state = {
  classes: [],
  datasetFiles: [],   // [{name, file, url}]
  selected: new Set(), // filenames picked as the test sample
  galleryTotal: 0,
};

// ---------- API status ----------
function setApiStatus(ok) {
  document.getElementById("apiDot").className = "dot " + (ok ? "dot-done" : "dot-idle");
  document.getElementById("apiLabel").textContent = ok ? "API OK" : "API OFF";
}
async function pingApi() {
  try { setApiStatus((await fetch(`${API_BASE}/health`)).ok); }
  catch { setApiStatus(false); }
}
pingApi();
setInterval(pingApi, 8000);

function setStatus(mode, label) {
  document.getElementById("statusDot").className =
    "dot " + (mode === "live" ? "dot-live" : mode === "done" ? "dot-done" : "dot-idle");
  document.getElementById("statusLabel").textContent = label;
}

// ---------- 1. Classes (tag input) ----------
const tagList = document.getElementById("tagList");
const classInput = document.getElementById("classInput");

function renderTags() {
  tagList.innerHTML = "";
  state.classes.forEach((name, i) => {
    const tag = document.createElement("span");
    tag.className = "tag";
    tag.innerHTML = `${name} <button class="tag-remove" data-i="${i}">✕</button>`;
    tagList.appendChild(tag);
  });
}
tagList.addEventListener("click", (e) => {
  if (e.target.classList.contains("tag-remove")) {
    state.classes.splice(parseInt(e.target.dataset.i, 10), 1);
    renderTags();
  }
});
function addClassesFromInput() {
  const raw = classInput.value;
  const parts = raw.split(",").map((s) => s.trim()).filter(Boolean);
  parts.forEach((val) => {
    if (!state.classes.includes(val)) state.classes.push(val);
  });
  classInput.value = "";
  renderTags();
}

classInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" || e.key === ",") {
    e.preventDefault();
    addClassesFromInput();
  }
});
// also catch a paste like "car, truck, van" losing focus without Enter
classInput.addEventListener("blur", () => {
  if (classInput.value.trim()) addClassesFromInput();
});

document.getElementById("resetClassesBtn").addEventListener("click", () => {
  state.classes = [];
  renderTags();
  document.getElementById("applyClassesBtn").textContent = "Appliquer";
});
document.getElementById("applyClassesBtn").addEventListener("click", async () => {
  if (classInput.value.trim()) addClassesFromInput(); // catch anything not yet committed
  if (state.classes.length === 0) { alert("Ajoute au moins un nom de classe."); return; }
  const btn = document.getElementById("applyClassesBtn");
  btn.disabled = true; btn.textContent = "Application…";
  try {
    const res = await fetch(`${API_BASE}/set_classes`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ classes: state.classes }),
    });
    const data = await res.json();
    if (data.error) throw new Error(data.error);
    setApiStatus(true);
    btn.textContent = "Appliquées ✓";
    document.getElementById("activeClassesHint").textContent =
      "Classes actives sur le serveur : " + data.classes.join(", ");
  } catch (err) {
    alert("Erreur : " + err.message);
    btn.textContent = "Appliquer";
  } finally { btn.disabled = false; }
});

// ---------- 2. Dataset upload + selectable thumbnail grid ----------
const datasetInput = document.getElementById("datasetInput");
const thumbGrid = document.getElementById("thumbGrid");
const datasetHint = document.getElementById("datasetHint");

datasetInput.addEventListener("change", async (e) => {
  const files = Array.from(e.target.files);
  if (!files.length) return;

  datasetHint.textContent = `Envoi de ${files.length} image(s)…`;

  // keep local previews + selection state
  files.forEach((f) => {
    state.datasetFiles.push({ name: f.name, file: f, url: URL.createObjectURL(f) });
  });
  renderThumbs();

  const fd = new FormData();
  files.forEach((f) => fd.append("images", f));
  try {
    const res = await fetch(`${API_BASE}/upload_dataset`, { method: "POST", body: fd });
    const data = await res.json();
    if (data.error) throw new Error(data.error);
    datasetHint.textContent = `${state.datasetFiles.length} image(s) chargée(s) — ${state.selected.size} sélectionnée(s) pour le test.`;
    document.getElementById("phaseHint").textContent =
      state.selected.size ? `${state.selected.size} image(s) sélectionnée(s).` : "Sélectionne 4 images ci-dessus, puis teste.";
    setApiStatus(true);
  } catch (err) {
    datasetHint.textContent = "Échec de l'envoi — backend lancé ?";
    setApiStatus(false);
  }
  e.target.value = "";
});

function renderThumbs() {
  thumbGrid.innerHTML = "";
  state.datasetFiles.forEach((item) => {
    const div = document.createElement("div");
    div.className = "thumb" + (state.selected.has(item.name) ? " selected" : "");
    div.innerHTML = `<img src="${item.url}" alt="${item.name}"><span class="thumb-check">✓</span>`;
    div.addEventListener("click", () => {
      if (state.selected.has(item.name)) {
        state.selected.delete(item.name);
      } else {
        state.selected.add(item.name);
      }
      renderThumbs();
      datasetHint.textContent = `${state.datasetFiles.length} image(s) chargée(s) — ${state.selected.size} sélectionnée(s) pour le test.`;
    });
    thumbGrid.appendChild(div);
  });
}

// ---------- 3. Run detection ----------
const testBtn = document.getElementById("testBtn");
const runBtn = document.getElementById("runBtn");
const phaseHint = document.getElementById("phaseHint");

async function callDetection(files) {
  const conf = parseFloat(document.getElementById("confInput").value) || 0.25;
  const imgsz = parseInt(document.getElementById("imgszInput").value, 10) || 640;
  const res = await fetch(`${API_BASE}/run_detection`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ classes: state.classes, dataset_dir: "dataset", conf, imgsz, files }),
  });
  const data = await res.json();
  if (data.error) throw new Error(data.error);
  return data;
}

testBtn.addEventListener("click", async () => {
  if (state.classes.length === 0) { alert("Applique au moins une classe (étape 1)."); return; }
  if (state.selected.size === 0) { alert("Sélectionne au moins une image (clique sur ses vignettes)."); return; }

  testBtn.disabled = true; runBtn.disabled = true;
  setStatus("live", `TEST (${state.selected.size} img)…`);
  try {
    const data = await callDetection([...state.selected]);
    renderResults(data, "replace");
    setStatus("done", "TERMINÉ");
    phaseHint.textContent = `Test ok sur ${data.summary.n_images} image(s). Vérifie les résultats en bas, puis clique "Lancer sur le reste".`;
  } catch (err) {
    setStatus("idle", "ERREUR");
    alert("Erreur : " + err.message);
  } finally { testBtn.disabled = false; runBtn.disabled = false; }
});

runBtn.addEventListener("click", async () => {
  if (state.classes.length === 0) { alert("Applique au moins une classe (étape 1)."); return; }
  const rest = state.datasetFiles.map((f) => f.name).filter((n) => !state.selected.has(n));
  if (rest.length === 0) { alert("Rien à traiter : toutes les images sont déjà dans la sélection de test."); return; }

  testBtn.disabled = true; runBtn.disabled = true;
  setStatus("live", `SCANNING (${rest.length} img)…`);
  try {
    const data = await callDetection(rest);
    renderResults(data, "append");
    setStatus("done", "TERMINÉ");
    phaseHint.textContent = `Reste traité : ${data.summary.n_images} image(s).`;
  } catch (err) {
    setStatus("idle", "ERREUR");
    alert("Erreur : " + err.message);
  } finally { testBtn.disabled = false; runBtn.disabled = false; }
});

// ---------- Render results ----------
state.allResults = [];

function renderResults(data, mode) {
  const { summary, results } = data;

  if (mode === "replace") state.allResults = [];
  state.allResults.push(...results);

  // recompute cumulative totals across everything shown so far
  const totalImages = state.allResults.length;
  const totalTime = state.allResults.reduce((s, r) => s + r.time_sec, 0);
  const totalDet = state.allResults.reduce((s, r) => s + r.detections, 0);

  document.getElementById("statImages").textContent = totalImages;
  document.getElementById("statTotalTime").textContent = totalTime.toFixed(2) + "s";
  document.getElementById("statAvgTime").textContent = (totalImages ? totalTime / totalImages : 0).toFixed(3) + "s";
  document.getElementById("statDetections").textContent = totalDet;
  document.getElementById("statRam").textContent =
    summary.peak_ram_mb + " MB" + (summary.peak_gpu_mb ? ` · GPU ${summary.peak_gpu_mb} MB` : ` · ${summary.device.toUpperCase()}`);

  const gallery = document.getElementById("gallery");
  const logTable = document.getElementById("logTable");

  if (mode === "replace") {
    gallery.innerHTML = "";
    logTable.querySelectorAll(".log-row:not(.log-row-head)").forEach(n => n.remove());
  } else {
    const ph = gallery.querySelector(".empty-state");
    if (ph) ph.remove();
  }
  logTable.querySelectorAll(".empty-state").forEach(n => n.remove());

  results.forEach((r) => {
    const item = document.createElement("div");
    item.className = "gallery-item";
    item.innerHTML = `
      <img src="${API_BASE}/outputs/${encodeURIComponent(r.image)}?t=${Date.now()}" alt="${r.image}">
      <span class="badge">${r.detections} obj · ${r.time_sec}s</span>
    `;
    gallery.appendChild(item);

    const row = document.createElement("div");
    row.className = "log-row";
    row.innerHTML = `
      <span title="${r.image}">${r.image}</span>
      <span>${r.time_sec}s</span>
      <span>${r.ram_delta_mb}MB</span>
      <span>${r.detections}</span>
    `;
    logTable.appendChild(row);
  });
}
