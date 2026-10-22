/* DataConfessor — browser side.
 *
 * Two ways to drive one command-line tool. The assistant loop lives here rather than on the
 * server: the model proposes one command, the page shows it, the server runs it if it is on
 * the allow list, and the output goes back. Every step is visible and interruptible, and the
 * server stays a dumb executor.
 *
 * The page never reads the per-patient CSV. It cannot: the server refuses to serve it.
 */

const TOKEN = document.querySelector('meta[name="dc-token"]').content;
const $ = (id) => document.getElementById(id);
const el = (tag, cls, text) => {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (text != null) n.textContent = text;
  return n;
};

async function api(path, body) {
  const res = await fetch(path, {
    method: body ? "POST" : "GET",
    headers: { "content-type": "application/json", "x-dc-token": TOKEN },
    body: body ? JSON.stringify(body) : undefined,
  });
  const data = await res.json().catch(() => ({ error: `${res.status} ${res.statusText}` }));
  if (!res.ok || data.error) throw new Error(data.error || `${res.status}`);
  return data;
}

const state = {
  workdir: "",
  manifest: "",
  patientsCsv: "",
  summaryJson: "",
  reportHtml: "",
  job: null,
  halted: false,
  columns: [],
};

/* ── theme ──────────────────────────────────────────────────────────────── */

const THEME_KEY = "dc-theme";
document.documentElement.dataset.theme = localStorage.getItem(THEME_KEY) || "auto";
$("theme-toggle").onclick = () => {
  const now = document.documentElement.dataset.theme;
  const next = now === "auto" ? (matchMedia("(prefers-color-scheme: dark)").matches ? "light" : "dark")
                              : now === "dark" ? "light" : "dark";
  document.documentElement.dataset.theme = next;
  localStorage.setItem(THEME_KEY, next);
};

/* ── console ────────────────────────────────────────────────────────────── */

function logLine(text, bad) {
  const log = $("log");
  const line = el("span", bad ? "bad" : null, text + "\n");
  log.appendChild(line);
  log.scrollTop = log.scrollHeight;
}
$("clear-log").onclick = () => ($("log").textContent = "");

function setVerdict(stateName, label, text) {
  $("verdict").dataset.state = stateName;
  $("verdict-label").textContent = label;
  $("verdict-text").textContent = text;
}

/* ── running commands ───────────────────────────────────────────────────── */

/** Start a command and resolve with its whole output once it exits. */
function run(cmd, onLine) {
  return new Promise(async (resolve, reject) => {
    let started;
    try {
      started = await api("/api/exec", { cmd });
    } catch (err) {
      return reject(err);
    }
    state.job = started.job;
    document.body.classList.add("is-running");
    const lines = [];
    const source = new EventSource(`/api/stream?job=${started.job}&t=${TOKEN}`);
    source.onmessage = (ev) => {
      const { kind, data } = JSON.parse(ev.data);
      if (kind === "line") {
        lines.push(data);
        trackProgress(data);
        onLine ? onLine(data) : logLine(data, /ERROR|\*\*/.test(data));
      } else if (kind === "exit") {
        source.close();
        state.job = null;
        document.body.classList.remove("is-running");
        showScanner(false);
        resolve({ exit: Number(data), text: lines.join("\n") });
      }
    };
    source.onerror = () => {
      source.close();
      state.job = null;
      document.body.classList.remove("is-running");
      showScanner(false);
      reject(new Error("the connection to the run was lost"));
    };
  });
}

/* ── the scanner ────────────────────────────────────────────────────────── */

/** Light one lamp per quarter of the proxy budget, from the tool's own progress lines.
 *  It is a progress indicator that cannot lie about time: it reports which proxy is
 *  training, not a guess at how long is left. */
function trackProgress(line) {
  let m;
  if ((m = line.match(/proxy (\d+)\/(\d+) at (\d+) epoch/))) {
    const k = Number(m[1]), n = Number(m[2]), ep = Number(m[3]);
    showScanner(true);
    setLamps(Math.ceil((k / n) * 4));
    $("scanner-note").textContent = `proxy ${k} of ${n}, ${ep} epoch${ep > 1 ? "s" : ""} each`;
  } else if ((m = line.match(/teacher (\d+)\/(\d+)/))) {
    showScanner(true);
    setLamps(Math.ceil((Number(m[1]) / Number(m[2])) * 4));
    $("scanner-note").textContent = `teacher ${m[1]} of ${m[2]}`;
  } else if (/caching records/.test(line)) {
    showScanner(true);
    setLamps(0);
    $("scanner-note").textContent = line.trim();
  } else if (/records into memory/.test(line)) {
    $("scanner-note").textContent = line.trim();
  } else if (/is below the .* floor/.test(line)) {
    setLamps(0);
    $("scanner-note").textContent = "the check failed — retraining the proxies for longer";
  }
}

function showScanner(on) {
  $("scanner").classList.toggle("is-hidden", !on);
  if (!on) setLamps(0);
}

function setLamps(count) {
  document.querySelectorAll(".scanner .lamp").forEach((lamp) => {
    lamp.classList.toggle("on", Number(lamp.dataset.lamp) <= count);
  });
}

/* ── reading what the tool printed ──────────────────────────────────────── */

/* The CLI prints an aggregate summary and nothing patient-level, so parsing stdout is safe
 * and keeps the page in step with the tool rather than reimplementing it. */
function readScreen(text) {
  const out = {};
  let m;
  if ((m = text.match(/screened\s+([\d,]+)\s+(patients|records)\s+\/\s+([\d,]+)\s+records/)))
    (out.patients = m[1]), (out.unit = m[2]), (out.records = m[3]);
  if ((m = text.match(/flagged\s+([\d,]+)\s+\w+\s+\((\d+)%\)/)))
    (out.flagged = m[1]), (out.flagRate = Number(m[2]));
  // Read the value from the SUMMARY lines, not from the fail-safe's progress lines: a
  // retried screen prints several self-consistency values and only the last one is the
  // verdict.
  if ((m = text.match(/self-consistency ([\d.]+) \(floor/)))
    (out.consistency = Number(m[1])), (out.failed = false);
  else if ((m = text.match(/SELF-CONSISTENCY ([\d.]+) </)))
    (out.consistency = Number(m[1])), (out.failed = true);
  if ((m = text.match(/fail-safe escalated the proxies: (.+)/))) out.ladder = m[1].trim();
  if ((m = text.match(/x (\d+) epoch\(s\) of (\S+)/))) (out.epochs = m[1]), (out.arch = m[2]);
  if ((m = text.match(/([\d.]+) minutes/))) out.minutes = m[1];
  if ((m = text.match(/([\d.]+) kWh/))) out.energy = m[1] + " kWh";
  else if ((m = text.match(/([\d.]+) kJ/))) out.energy = m[1] + " kJ";
  else if (/energy not metered/.test(text)) out.energy = "not metered here";
  return out;
}

/** The flag-rate grid: 100 squares, one per percent of the cohort, as in the paper's Fig 2a. */
function paintGrid(onPercent, flaggedText, totalText, animate) {
  const grid = $("grid");
  grid.innerHTML = "";
  grid.classList.toggle("is-new", !!animate);
  for (let i = 0; i < 100; i++) {
    const cell = el("i", i < onPercent ? "on" : null);
    if (animate) cell.style.animationDelay = `${i * 5}ms`;
    grid.appendChild(cell);
  }
  $("grid-flagged").textContent = flaggedText;
  $("grid-rest").textContent = totalText;
}

function paintResult(r) {
  if (!r.patients) return;
  const grid = $("grid");
  paintGrid(Math.round(r.flagRate || 20), `${r.flagged} flagged`, `${r.patients} screened`, true);

  const stats = $("stats");
  stats.innerHTML = "";
  const rows = [
    [r.unit === "records" ? "Records screened" : "Patients screened", r.patients],
    ["Records", r.records],
    ["Flagged", `${r.flagged} (${r.flagRate}%)`],
    ["Self-consistency", r.consistency != null ? r.consistency.toFixed(3) : "—"],
    ...(r.ladder ? [["Fail-safe", r.ladder]] : []),
    ["Proxies", r.arch ? `${r.arch} × ${r.epochs} epoch(s)` : "—"],
    ["Compute", r.minutes ? `${r.minutes} min` : "—"],
    ["Energy", r.energy || "—"],
  ];
  for (const [k, v] of rows) {
    stats.appendChild(el("dt", null, k));
    stats.appendChild(el("dd", null, String(v)));
  }
  stats.classList.remove("is-hidden");

  if (r.consistency == null) return;
  if (r.failed) {
    setVerdict("fail",
      `Self-consistency ${r.consistency.toFixed(3)} — these flags are weaker than a passing screen`,
      "The fail-safe retrained the proxies for as many epochs as it was allowed and the two " +
      "halves still disagree about who is atypical. The flags come from the most-trained " +
      "attempt and are marked below_floor in the CSV. Use them to prioritise, not to " +
      "conclude: try 8 proxies, check that the patient column really groups one person's " +
      "records, and consider a full audit.");
  } else {
    setVerdict("pass", `Self-consistency ${r.consistency.toFixed(3)} — the ranking holds`,
      "Two independent halves of the proxy models rank patients the same way. That is " +
      "necessary for the flags to mean anything; it does not tell you how much of the true " +
      "tail they contain. An unflagged patient is not certified safe.");
  }
  showArtifacts();
}

function showArtifacts() {
  const box = $("artifacts");
  box.innerHTML = "";
  if (state.reportHtml) {
    const url = `/api/file?path=${encodeURIComponent(state.reportHtml)}&t=${TOKEN}`;
    const open = el("button", "artifact", "Read the report");
    open.onclick = () => {
      $("report-frame").src = url;
      $("report-newtab").href = url;
      $("reportview").showModal();
    };
    box.appendChild(open);
  }
  if (state.summaryJson) {
    const a = el("a", "artifact", "Summary JSON");
    a.href = `/api/file?path=${encodeURIComponent(state.summaryJson)}&t=${TOKEN}`;
    a.target = "_blank";
    a.rel = "noopener";
    box.appendChild(a);
  }
  if (state.patientsCsv) {
    const sealed = el("span", "artifact sealed");
    sealed.appendChild(el("span", null, "Flagged cohort"));
    sealed.appendChild(el("small", null, "stays on this machine"));
    sealed.title = state.patientsCsv +
      " — this file ranks individuals by privacy risk, so it is not served to this page.";
    box.appendChild(sealed);
  }
  const reveal = el("button", "artifact", "Show in folder");
  reveal.onclick = () => api("/api/reveal", { path: state.workdir }).catch(() => {});
  box.appendChild(reveal);
  box.classList.remove("is-hidden");
}

/* ── tabs ───────────────────────────────────────────────────────────────── */

document.querySelectorAll(".tab").forEach((tab) => {
  tab.onclick = () => {
    document.querySelectorAll(".tab").forEach((t) => {
      t.classList.toggle("is-on", t === tab);
      t.setAttribute("aria-selected", String(t === tab));
    });
    $("panel-assistant").classList.toggle("is-hidden", tab.dataset.mode !== "assistant");
    $("panel-manual").classList.toggle("is-hidden", tab.dataset.mode !== "manual");
  };
});

/* ── the folder picker ──────────────────────────────────────────────────── */

let browseTarget = null, browseFiles = false, browsePath = "";

document.querySelectorAll("[data-browse]").forEach((btn) => {
  btn.onclick = async () => {
    browseTarget = btn.dataset.browse;
    browseFiles = btn.dataset.files === "1";
    const seed = $(browseTarget).value || state.workdir;
    await paintBrowser(seed);
    $("browser").showModal();
  };
});

async function paintBrowser(path) {
  const data = await api("/api/browse", { path });
  browsePath = data.path;
  $("browser-path").textContent = data.path;
  $("browser-up").disabled = !data.parent;
  const list = $("browser-list");
  list.innerHTML = "";
  for (const entry of data.entries) {
    if (!entry.dir && !browseFiles) continue;
    if (!entry.dir && browseFiles && !/\.(csv|tsv|txt)$/i.test(entry.name)) continue;
    const row = el("button", "row" + (entry.dir ? "" : " is-file"));
    row.type = "button";
    row.appendChild(el("span", null, entry.dir ? "▸" : "·"));
    row.appendChild(el("span", null, entry.name));
    if (!entry.dir) row.appendChild(el("span", "tag", `${Math.ceil(entry.size / 1024)} KB`));
    row.onclick = () => {
      if (entry.dir) return paintBrowser(entry.path);
      pick(entry.path);
    };
    list.appendChild(row);
  }
  if (!list.children.length) list.appendChild(el("p", "note", "nothing to choose in here"));
}

function pick(path) {
  $(browseTarget).value = path;
  $("browser").close();
  $(browseTarget).dispatchEvent(new Event("change"));
}
$("browser-up").onclick = async () => {
  const data = await api("/api/browse", { path: browsePath });
  if (data.parent) paintBrowser(data.parent);
};
$("browser-pick").onclick = () => pick(browsePath);
$("reportview").addEventListener("close", () => ($("report-frame").src = "about:blank"));

/* ── manual mode ────────────────────────────────────────────────────────── */

const GUESS = {
  "col-image": ["image", "file", "name", "id", "path"],
  "col-label": ["target", "label", "class", "diagnos", "malign", "benign",
                "finding", "outcome", "grade", "severity"],
  "col-patient": ["patient", "mrn", "subject", "person", "case", "participant"],
};

$("labels-csv").addEventListener("change", async () => {
  const path = $("labels-csv").value.trim();
  if (!path) return;
  try {
    const { columns } = await api("/api/headers", { path });
    state.columns = columns;
    for (const [id, hints] of Object.entries(GUESS)) {
      const sel = $(id);
      sel.innerHTML = "<option value=''>auto</option>";
      columns.forEach((c) => sel.appendChild(new Option(c, c)));
      const hit = hints.map((h) => columns.find((c) => c.toLowerCase().includes(h))).find(Boolean);
      if (hit) sel.value = hit;
    }
    $("patient-warning").textContent = $("col-patient").value
      ? `Patients will be grouped by "${$("col-patient").value}". Every record of one person must carry the same value there — nothing downstream can check this.`
      : "No column looks like a patient identifier. Without one the screen reports per record, not per patient.";
  } catch (err) {
    $("patient-warning").textContent = `Could not read that CSV: ${err.message}`;
  }
});

function openStep(n) {
  document.querySelectorAll(".step").forEach((s) => {
    const i = Number(s.dataset.step);
    s.classList.toggle("is-open", i === n);
    s.classList.toggle("is-done", i < n);
  });
}
document.querySelectorAll(".step h2").forEach((h) =>
  (h.onclick = () => openStep(Number(h.parentElement.dataset.step))));

/** The last ERROR line a command printed, which is what the user needs to read. */
function failure(text, fallback) {
  const line = (text || "").split(/\r?\n/).reverse().find((l) => /^ERROR|error:/i.test(l));
  return line ? line.replace(/^ERROR:\s*/, "") : fallback;
}

function quote(value) {
  return /[ "]/.test(value) ? `"${value}"` : value;
}

async function guard(button, work) {
  button.disabled = true;
  try {
    await ready;
    if (!state.workdir) {
      throw new Error("the page could not reach the tool; reload it before running anything");
    }
    await work();
  } catch (err) {
    logLine(`ERROR: ${err.message}`, true);
    setVerdict("fail", "That did not run", err.message);
  } finally {
    button.disabled = false;
  }
}

$("do-init").onclick = () =>
  guard($("do-init"), async () => {
    const dir = $("data-dir").value.trim();
    if (!dir) throw new Error("choose the folder that holds the records first");
    // Someone who already has a manifest -- written by hand, or by a script for a dataset
    // this tool cannot crawl, like PTB-XL -- should not be made to rebuild it.
    if (/\.csv$/i.test(dir)) {
      state.manifest = dir;
      logLine(`using the manifest at ${dir}`);
      openStep(2);
      return;
    }
    state.manifest = `${state.workdir}${SEP}manifest.csv`;
    let cmd = `dataconfessor init ${quote(dir)} --out ${quote(state.manifest)}`;
    const csv = $("labels-csv").value.trim();
    if (csv) cmd += ` --labels-csv ${quote(csv)}`;
    for (const [id, flag] of [["col-image", "--image-col"], ["col-label", "--label-col"],
                              ["col-patient", "--patient-col"]]) {
      if ($(id).value) cmd += ` ${flag} ${quote($(id).value)}`;
    }
    logLine(`$ ${cmd}`);
    const { exit, text } = await run(cmd);
    if (exit !== 0) throw new Error(failure(text, "the manifest was not built"));
    openStep(2);
  });

$("do-check").onclick = () =>
  guard($("do-check"), async () => {
    if (!state.manifest) throw new Error("build the manifest first");
    const cmd = `dataconfessor check ${quote(state.manifest)}${kindFlag()}`;
    logLine(`$ ${cmd}`);
    const { text, exit } = await run(cmd);
    
    try {
      const data = JSON.parse(text.slice(text.indexOf("{"), text.lastIndexOf("}") + 1));
      const rpp = data.records_per_patient;
      $("cohort").innerHTML =
        `<b>${data.n_patients.toLocaleString()}</b> patients · ` +
        `<b>${data.n_records.toLocaleString()}</b> records · ` +
        `${rpp.median} records per patient (max ${rpp.max}) · ` +
        `${Object.keys(data.classes).length} classes` +
        (data.warnings.length ? `<br><span class="bad">${data.warnings.join("<br>")}</span>` : "");
      if (data.per_record_only)
        $("cohort").innerHTML += "<br>No patient identifiers: results will be per record.";
    } catch { /* the console already has the output */ }
    if (exit !== 0) throw new Error(failure(text, "the manifest did not pass its checks"));
    openStep(3);
  });

$("do-screen").onclick = () =>
  guard($("do-screen"), async () => {
    if (!state.manifest) throw new Error("build the manifest first");
    state.patientsCsv = `${state.workdir}${SEP}flagged_patients.csv`;
    state.summaryJson = `${state.workdir}${SEP}screen_summary.json`;
    state.reportHtml = `${state.workdir}${SEP}screen_report.html`;
    let cmd = `dataconfessor screen ${quote(state.manifest)}` +
      ` --patients ${quote(state.patientsCsv)}` +
      ` --out ${quote(state.summaryJson)}` +
      ` --report ${quote(state.reportHtml)}` +
      ` --flag ${$("flag").value} --epochs ${$("epochs").value} --models ${$("models").value}` +
      kindFlag();
    if ($("device").value) cmd += ` --device ${$("device").value}`;
    if ($("loader").value.trim()) cmd += ` --loader ${quote($("loader").value.trim())}`;
    setVerdict("busy", "Screening…",
      "Four proxy models are being trained on complementary halves of your patients. The " +
      "first run on a dataset also decodes every record into a cache.");
    logLine(`$ ${cmd}`);
    const { text } = await run(cmd);
    paintResult(readScreen(text));
  });

function kindFlag() {
  return $("kind").value ? ` --kind ${$("kind").value}` : "";
}

/* ── assistant mode ─────────────────────────────────────────────────────── */

const DEFAULT_MODEL = { anthropic: "claude-sonnet-5-5", ollama: "gemma4:31b", openai: "gpt-4.1-mini" };
const KEY_STORE = "dc-provider";

(function restoreProvider() {
  const saved = JSON.parse(sessionStorage.getItem(KEY_STORE) || "{}");
  $("provider-box").open = !saved.key;
  $("provider").value = saved.provider || "anthropic";
  $("model").value = saved.model || DEFAULT_MODEL[$("provider").value];
  $("baseurl").value = saved.baseUrl || "";
  $("apikey").value = saved.key || "";
  summariseProvider();
})();

function providerConfig() {
  return {
    provider: $("provider").value,
    model: $("model").value.trim() || DEFAULT_MODEL[$("provider").value],
    key: $("apikey").value.trim(),
    base_url: $("baseurl").value.trim(),
  };
}

function summariseProvider() {
  const c = providerConfig();
  const local = c.base_url.includes("localhost") || c.base_url.includes("127.0.0.1");
  $("provider-summary").textContent =
    `${c.model} · ${local ? "local, nothing leaves this machine" : c.provider}`;
  sessionStorage.setItem(KEY_STORE, JSON.stringify({ ...c, baseUrl: c.base_url }));
}
["provider", "model", "apikey", "baseurl"].forEach((id) =>
  $(id).addEventListener("change", () => {
    if (id === "provider") $("model").value = DEFAULT_MODEL[$("provider").value];
    summariseProvider();
  }));

document.querySelectorAll(".seed").forEach((s) =>
  (s.onclick = () => {
    $("prompt").value = s.textContent;
    $("prompt").focus();
  }));

const history = [];

function addTurn(node) {
  const intro = document.querySelector(".intro");
  if (intro) intro.remove();
  node.classList.add("turn");
  $("transcript").appendChild(node);
  $("transcript").scrollTop = $("transcript").scrollHeight;
  return node;
}

function commandFrom(reply) {
  const fenced = reply.match(/```(?:bash|sh|shell)?\s*\n([\s\S]*?)```/);
  const line = fenced ? fenced[1] : (reply.match(/^\s*(dataconfessor .+)$/m) || [])[1];
  return line ? line.trim().split("\n")[0].trim() : null;
}

$("composer").onsubmit = async (ev) => {
  ev.preventDefault();
  const text = $("prompt").value.trim();
  if (!text) return;
  $("prompt").value = "";
  addTurn(el("div", "turn-user", text));
  history.push({ role: "user", content: text });
  await converse();
};

$("halt").onclick = () => {
  state.halted = true;
  if (state.job) api("/api/stop", { job: state.job }).catch(() => {});
};

async function converse() {
  await ready;
  const config = providerConfig();
  if (config.provider !== "ollama" && !config.key) {
    $("provider-box").open = true;
    addTurn(el("div", "turn-model", "Add an API key above, or point the Base URL at a local Ollama."));
    return;
  }
  state.halted = false;
  $("send").disabled = true;
  $("halt").classList.remove("is-hidden");

  try {
    for (let step = 0; step < 16 && !state.halted; step++) {
      const thinking = addTurn(el("div", "turn-model", "…"));
      let reply;
      try {
        ({ reply } = await api("/api/chat", {
          ...config,
          messages: history.map(({ role, content }) => ({ role, content })),
        }));
      } catch (err) {
        thinking.textContent = `The model could not be reached. ${err.message}`;
        return;
      }
      history.push({ role: "assistant", content: reply });
      const cmd = commandFrom(reply);
      thinking.textContent = reply.replace(/```[\s\S]*?```/g, "").trim() || (cmd ? "" : reply);
      if (!thinking.textContent) thinking.remove();
      if (!cmd) {
        if (!thinking.isConnected)
          addTurn(el("div", "turn-model",
            "The model replied with nothing. Some models need a shorter system prompt, or " +
            "a non-thinking variant; try another model."));
        return;
      }

      const card = addTurn(el("div", "cmd"));
      card.appendChild(el("div", "cmd-line", cmd));
      const out = el("pre", "cmd-out", "running…");
      card.appendChild(out);

      let result;
      try {
        result = await run(cmd, (line) => {
          out.textContent = out.textContent === "running…" ? line : out.textContent + "\n" + line;
          out.scrollTop = out.scrollHeight;
          logLine(line, /ERROR|\*\*/.test(line));
        });
      } catch (err) {
        out.classList.add("cmd-refused");
        out.textContent = err.message;
        history.push({ role: "user", content: `REFUSED: ${err.message}` });
        continue;
      }
      if (out.textContent === "running…") out.textContent = `(no output, exit ${result.exit})`;
      if (/dataconfessor\s+screen/.test(cmd)) {
        noteArtifacts(cmd);
        paintResult(readScreen(result.text));
      }
      history.push({ role: "user", content: result.text.slice(-6000) || `exit ${result.exit}` });
    }
  } finally {
    $("send").disabled = false;
    $("halt").classList.add("is-hidden");
  }
}

/** Remember the paths the assistant chose, so the artifact row points at the real files. */
function noteArtifacts(cmd) {
  const grab = (flag) => {
    const m = cmd.match(new RegExp(`${flag}\\s+("[^"]+"|\\S+)`));
    return m ? m[1].replace(/"/g, "") : "";
  };
  const absolute = (p) => (!p || /^([a-zA-Z]:|\/|\\\\)/.test(p) ? p : `${state.workdir}${SEP}${p}`);
  state.patientsCsv = absolute(grab("--patients")) || state.patientsCsv;
  state.summaryJson = absolute(grab("--out")) || state.summaryJson;
  state.reportHtml = absolute(grab("--report")) || state.reportHtml;
}

/* ── help dialogs, boot ─────────────────────────────────────────────────── */

for (const id of ["open-help", "open-help-2"]) {
  $(id).onclick = (ev) => {
    ev.preventDefault();
    $("help").showModal();
  };
}

let SEP = "\\";

/** Resolved once the page knows its environment. Every action waits for it: `/api/env`
 *  imports torch to report the device, which on a cold or busy machine takes seconds, and a
 *  run started before it returns would write its outputs to the root of the drive. */
const ready = (async function boot() {
  try {
    const env = await api("/api/env");
    state.workdir = env.workdir;
    SEP = env.separator;
    $("chip-version").textContent = `v${env.version}`;
    $("chip-device").textContent = env.device === "cuda" ? `GPU · ${env.device_detail}` : "CPU only";
    $("chip-device").title = env.device_detail;
    $("chip-workdir").textContent = env.workdir;
    $("chip-workdir").title = "outputs are written here";
    $("data-dir").placeholder = env.home + SEP + "data";
    paintGrid(0, "nothing screened yet", "20% is the calibrated flag rate");
  } catch (err) {
    logLine(`ERROR: ${err.message}`, true);
  }
})();
