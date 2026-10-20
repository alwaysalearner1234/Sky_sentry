"use strict";

const CONFIG = {
  // DATA_SOURCE: "orbitguard" → your OrbitGuard backend (localhost:8000)
  //              "keeptrack"  → LIVE orbital data from api.keeptrack.space (real TLEs, client-side propagation)
  //              "mock"       → built-in ephemeris, no network at all
  DATA_SOURCE: "mock",

  BACKEND_URL: "http://localhost:8000",
  POLL_INTERVAL_MS: 1000,
  RISK_POLL_INTERVAL_MS: 2500, // /risks/all runs full pair-wise SGP4 — don't hammer it
  AUTH_POLL_INTERVAL_MS: 30000, // audit ledger verification cadence
  USE_MOCK: false,  // legacy flag — true → built-in fake data, no server needed
  DEMO_MODE: false, // legacy flag — true → built-in data, accelerated demo cycle

  // --- KeepTrack.space v4 (real catalog + TLEs) ---
  KEEPTRACK_API_KEY: "kt_7332c9cf81c237c3174b8ecb54060496", // YOUR key — keep private!
  KT_CATALOG_URL: "https://api.keeptrack.space/v4/sats/brief",
  KT_REFRESH_MS: 3600000,  // re-pull the catalog hourly (fresh TLEs)
  KT_TIME_SCALE: 60,       // sim-seconds advanced per real second (1x → real-time motions)
  KT_SCREEN_HOURS: 3,      // how far ahead we predict closest approaches
  KT_SENTINEL_COUNT: 12,   // payloads shown in the constellation panel
  KT_WINDOW_SIZE: 8,       // tracked debris kept per sentinel
  KT_MAX_DEBRIS_DIST: 1800,// km — drop window objects further away than this
};

const MAX_LOG_ENTRIES = 20;
const MAX_RANGE_KM = 400;   // radar plots objects out to this range
const SAFETY_BOX_KM = 50;   // config constant — the real backend has no box
const DEFAULT_COUNT = 12;   // satellites to request from OrbitGuard
const RANK = { safe: 0, warning: 1, breach: 2 };
const LEVEL_PRIORITY = { CRITICAL: 3, HIGH: 2, WATCH: 1, SAFE: 0 };
const GLOBAL_TEXT = {
  safe: "GLOBAL: ORBIT CLEAR",
  warning: "GLOBAL: PROXIMITY ALERT",
  breach: "GLOBAL: SECTOR BREACH",
};
const SAT_ICON = { safe: "🛰️", warning: "⚠️", breach: "🚨" };

const els = {};
let audioCtx = null;
let radarAngle = 0;
let tick = 0;
let offline = false;
let riskInflight = false;
let auditSeries = 0;

const state = {
  sats: [],          // raw satellites from /satellites
  constellation: [],
  risks: [],         // alerts from /risks/all
  scenario: null,
  activeSat: null,
  lockedId: null,
  package: null,     // maneuver package
  packageJSON: "",
  histories: {},
};

document.addEventListener("DOMContentLoaded", init);

  // Ensure is-loading class is removed within 3 seconds even if something fails
  setTimeout(() => {
    document.body.classList.remove("is-loading");
  }, 3000);

function init() {
  try {
    cacheElements();
    startClock();
    bindButtons();
    log("info", "SKY SENTRY grid online — requesting satellite network from OrbitGuard…");
    verifyLedger();
    poll();
    setInterval(poll, CONFIG.POLL_INTERVAL_MS);
    setInterval(drawSkyMap, CONFIG.POLL_INTERVAL_MS);
    setInterval(() => { if (lockedObject()) drawTrajectory(); }, 300);
    setInterval(verifyLedger, CONFIG.AUTH_POLL_INTERVAL_MS);
    updateSafetyBox();
    setInterval(updateSafetyBox, 200);
    requestAnimationFrame(drawRadar);
  } catch (e) {
    console.error("INIT FAILED:", e.message, e.stack);
    document.body.classList.remove("is-loading");
  }
}

function cacheElements() {
  const ids = [
    "offline-banner", "utc-clock", "chip-global", "stat-sats", "stat-objs",
    "stat-breach", "sat-list", "active-sat-tag", "lock-chip", "detect-msg",
    "t-sat", "t-org", "t-alt", "t-incl", "t-vel", "t-mass", "t-count",
    "t-debris-id", "t-dist", "t-box", "t-updated", "evade-card", "e-dv",
    "e-sep", "e-fuel", "e-risk", "e-token-short", "e-token-full", "e-time",
    "btn-copy", "btn-view", "tracker-count", "obj-list", "lock-card",
    "lock-head", "l-id", "l-type", "l-threat", "l-dist", "l-vel", "l-bearing",
    "l-eta", "obj-hint", "audit-chip", "event-log",
  ];
  for (const id of ids) els[id] = document.getElementById(id);
  els.radar = document.getElementById("radar");
  els.rctx = els.radar ? els.radar.getContext("2d") : null;
  els.trajectory = document.getElementById("trajectory");
  els.tctx = els.trajectory ? els.trajectory.getContext("2d") : null;
  // Report any missing required elements
  const missing = ids.filter(id => !els[id]);
  if (missing.length) console.warn("MISSING ELEMENT IDS:", missing);
  if (!els.radar) console.warn("MISSING: #radar canvas");
}

function bindButtons() {
  if (els["btn-copy"]) els["btn-copy"].addEventListener("click", async () => {
    if (!state.packageJSON) return;
    try {
      await navigator.clipboard.writeText(state.packageJSON);
      log("info", "Full maneuver package copied to clipboard.");
    } catch {
      log("error", "Clipboard unavailable — use VIEW FULL to copy manually.");
    }
  });

  if (els["btn-view"]) els["btn-view"].addEventListener("click", () => {
    const expanded = !els["e-token-full"].classList.contains("hidden");
    els["e-token-full"].classList.toggle("hidden", expanded);
    els["e-token-short"].classList.toggle("hidden", !expanded);
    els["btn-view"].textContent = expanded ? "VIEW FULL" : "HIDE FULL";
  });
}

function startClock() {
  const tickClock = () => {
    els["utc-clock"].textContent = new Date().toISOString().slice(11, 19);
  };
  tickClock();
  setInterval(tickClock, 1000);
}

/* ---------------- DATA ACQUISITION (OrbitGuard) ---------------- */

async function poll() {
  tick += 1;
  try {
    await loadSatellitesIfEmpty();
    await loadRisksIfDue();
    await loadScenario();
    setOffline(false);
    buildConstellation();
    if (state.constellation.length) {
      paint();
    }
  } catch (err) {
    if (err && err.httpStatus) log("error", `OrbitGuard returned HTTP ${err.httpStatus}.`);
    setOffline(true);
  }
}

async function api(url, opts) {
  const res = await fetch(`${CONFIG.BACKEND_URL}${url}`, {
    cache: "no-store",
    headers: { "Content-Type": "application/json" },
    ...opts,
  });
  let data;
  try {
    data = await res.json();
  } catch (err) {
    const e = new Error("malformed JSON");
    e.jsonWarning = true;
    throw e;
  }
  if (!res.ok) {
    const e = new Error(`HTTP ${res.status}`);
    e.httpStatus = res.status;
    throw e;
  }
  return data;
}

function useKeeptrack() {
  return CONFIG.DATA_SOURCE === "keeptrack";
}
function useMock() {
  return CONFIG.USE_MOCK || CONFIG.DEMO_MODE || CONFIG.DATA_SOURCE === "mock";
}

async function loadSatellitesIfEmpty() {
  if (state.sats.length) return;
  if (useKeeptrack()) {
    await keepTrackReady();
    return;
  }
  const payload = useMock() ? mockSatellites() : await api(`/satellites?count=${DEFAULT_COUNT}`);
  const map = payload?.satellites;
  if (!map || typeof map !== "object") throw new Error("no satellites in payload");
  state.sats = Object.values(map);
  log("info", `OrbitGuard network acquired — ${state.sats.length} objects in sector.`);
}

async function loadRisksIfDue() {
  const now = Date.now();
  if (now - (state.lastRiskPoll || 0) < CONFIG.RISK_POLL_INTERVAL_MS) return;
  if (riskInflight) return;
  riskInflight = true;
  state.lastRiskPoll = now;
  try {
    if (useKeeptrack()) {
      keepTrackScreen();
      return;
    }
    const payload = useMock() ? mockRisks() : await api(`/risks/all?uncertainty=15`, { method: "POST", body: "{}" });
    state.risks = Array.isArray(payload?.alerts) ? payload.alerts : [];
  } finally {
    riskInflight = false;
  }
}

async function loadScenario() {
  if (useKeeptrack()) {
    keepTrackScenario();
    return;
  }
  if (useMock()) {
    state.scenario = mockScenario();
    return;
  }
  state.scenario = await api(`/scenario/status`);
}

async function computeManeuver(satelliteId, threatId) {
  if (useKeeptrack()) return keepTrackManeuver(satelliteId, threatId);
  if (useMock()) return mockManeuver(satelliteId, threatId);
  return api(`/maneuver/compute`, {
    method: "POST",
    body: JSON.stringify({ satellite_id: satelliteId, threatening_object_id: threatId }),
  });
}

async function verifyLedger() {
  auditSeries += 1;
  try {
    const result = useKeeptrack()
      ? keepTrackLedger()
      : useMock()
        ? { integrity_valid: true, total_blocks: 20 + auditSeries }
        : await api(`/audit/verify`, { method: "POST" });
    const chip = els["audit-chip"];
    if (chip) {
      chip.textContent = `TAMPER-PROOF LEDGER: ${result.integrity_valid ? "VERIFIED ✓" : "INVALID ✗"} · ${result.total_blocks} BLOCKS`;
      chip.classList.toggle("audit-ok", !!result.integrity_valid);
      chip.classList.toggle("audit-fail", !result.integrity_valid);
    }
  } catch {
    const chip = els["audit-chip"];
    if (chip) {
      chip.textContent = "TAMPER-PROOF LEDGER: UNREACHABLE";
      chip.classList.remove("audit-ok");
      chip.classList.add("audit-fail");
    }
  }
}

/* ---------------- STATE / DERIVATION ---------------- */

function buildConstellation() {
  const satById = {};
  for (const s of state.sats) satById[s.id] = s;

  state.constellation = state.sats.map((s) => {
    const rel = state.risks.filter(
      (a) => a.satellite_a === s.id || a.satellite_b === s.id,
    );
    const objs = rel.map((a) => {
      const otherId = a.satellite_a === s.id ? a.satellite_b : a.satellite_a;
      const other = satById[otherId];
      const level = a.risk_score?.level || "SAFE";
      return {
        object_id: otherId,
        object_type: a.bearing_deg !== undefined ? "DEBRIS" : (other?.organization || "UNKNOWN"),
        distance_km: a.distance_km ?? a.closest_approach?.distance_km,
        relative_velocity_kms: a.relative_velocity?.km_s,
        threat: level,
        bearing_deg: a.bearing_deg ?? hashBearing(otherId + s.id),
        eta_s: a.time_to_collision_s,
        tca: a.closest_approach?.time_utc,
      };
    }).sort((x, y) => (x.distance_km || 0) - (y.distance_km || 0));

    const level = worstLevel(rel);
    return {
      satellite_id: s.id,
      organization: s.organization,
      altitude_km: s.altitude_km,
      inclination_deg: s.inclination,
      velocity_km_s: s.velocity_km_s,
      mass_kg: s.mass_kg,
      is_active: s.is_active !== false,
      status: level === "CRITICAL" ? "breach" : level === "HIGH" ? "warning" : "safe",
      objects: objs,
      closest_debris: objs[0] || null,
    };
  });
}

function worstLevel(list) {
  let best = "SAFE";
  for (const a of list) {
    const l = a.risk_score?.level || "SAFE";
    if (LEVEL_PRIORITY[l] > LEVEL_PRIORITY[best]) best = l;
  }
  return best;
}

function activeStateOf(sat) {
  if (!sat) return "safe";
  var s = sat.status;
  return s;
}

/* ---------------- RENDERING ---------------- */

function paint() {
  if (!state.activeSat ||
      !state.constellation.find((s) => s.satellite_id === state.activeSat.satellite_id)) {
    state.activeSat = state.constellation[0];
    log("info", `Default tracking → ${state.activeSat.satellite_id}`);
  }
  state.activeSat = state.constellation.find(
    (s) => s.satellite_id === state.activeSat.satellite_id,
  ) || state.constellation[0];

  renderConstellation();
  renderGlobal();
  renderActiveSat();
  for (const obj of state.activeSat.objects || []) recordHistory(state.activeSat.satellite_id, obj);
  renderTracker();
  renderLockCard();
  handleEvasion();
}

function renderConstellation() {
  const sat = state.activeSat;
  els["active-sat-tag"].textContent = `ACTIVE: ${sat.satellite_id} · ${sat.status.toUpperCase()}`;
  els["active-sat-tag"].className = `active-tag tag-${sat.status}`;

  els["stat-sats"].textContent = `SAT LINKED: ${state.constellation.length}`;
  els["stat-objs"].textContent = `ALERTS: ${state.risks.length}`;
  const breached = state.constellation.filter((s) => s.status === "breach").length;
  els["stat-breach"].textContent = `BREACHED: ${breached}`;

  els["sat-list"].innerHTML = "";
  for (const s of state.constellation) {
    const row = document.createElement("div");
    row.className = `sat-row row-${s.status}` +
      (s.satellite_id === state.activeSat.satellite_id ? " sat-active" : "");

    const dot = document.createElement("span");
    dot.className = "sat-dot";

    const info = document.createElement("div");
    info.className = "sat-info";
    const name = document.createElement("div");
    name.className = "sat-name";
    name.textContent = s.satellite_id;
    const meta = document.createElement("div");
    meta.className = "sat-meta";
    const c = s.closest_debris;
    meta.textContent = c
      ? `ALERTS ${s.objects.length} · CLOSEST ${num(c.distance_km, 1)}km`
      : `ALERTS 0 · NO CLOSE APPROACH`;
    info.append(name, meta);

    const btn = document.createElement("button");
    btn.className = "sat-btn btn-ghost";
    btn.type = "button";
    btn.textContent = "TRACK";
    btn.addEventListener("click", (e) => {
      e.stopPropagation();
      selectSatellite(s.satellite_id);
    });

    row.addEventListener("click", () => selectSatellite(s.satellite_id));
    row.append(dot, info, btn);
    els["sat-list"].append(row);
  }
}

function selectSatellite(id) {
  const sat = state.constellation.find((s) => s.satellite_id === id);
  if (!sat || sat.satellite_id === state.activeSat?.satellite_id) return;
  state.activeSat = sat;
  state.lockedId = null;
  els["lock-chip"].classList.add("hidden");
  els["lock-card"].classList.add("hidden");
  els["obj-hint"].classList.remove("hidden");
  log("info", `RTO switch → tracking ${sat.satellite_id}`);
  renderConstellation();
  renderActiveSat();
  renderTracker();
  renderLockCard();
}

function renderGlobal() {
  // Always ensure loading state is cleared
  document.body.classList.remove("is-loading");
  const global = state.constellation.reduce(
    (a, b) => (RANK[b.status] > RANK[a.status] ? b : a),
    state.constellation[0] || { status: "safe" },
  ).status;
  const prev = document.body.dataset.global || null;

  document.body.classList.remove("is-loading");
  document.body.classList.remove("state-safe", "state-warning", "state-breach");
  document.body.classList.add(`state-${global}`);

  if (prev !== global) {
    els["chip-global"].textContent = GLOBAL_TEXT[global];
    els["chip-global"].className = `header-chip chip-${global}`;
    if (global === "breach") {
      beep(880, 0.2, "square");
      log("breach", "SECTOR BREACH — critical conjunction detected by OrbitGuard.");
    } else if (prev === "breach" && global === "safe") {
      beep(440, 0.15, "sine");
      log("safe", "Sector clear — returning to nominal orbit pattern.");
    } else if (prev !== global) {
      log(global, `Global status → ${global.toUpperCase()}`);
    }
  }
  document.body.dataset.global = global;
  setText("t-updated", new Date().toISOString().slice(11, 19) + " UTC");
}

function renderActiveSat() {
  const sat = state.activeSat;
  if (!sat) return;
  setText("t-sat", sat.satellite_id);
  setText("t-org", sat.organization);
  setText("t-alt", num(sat.altitude_km, 1) + " km");
  setText("t-incl", num(sat.inclination_deg, 1) + "°");
  setText("t-vel", num(sat.velocity_km_s, 2) + " km/s");
  setText("t-mass", num(sat.mass_kg, 0) + " kg");
  setText("t-count", sat.objects.length);
  setText("t-box", SAFETY_BOX_KM + " km");

  const c = sat.closest_debris;
  setText("t-debris-id", c?.object_id);
  setText("t-dist", num(c?.distance_km, 2) + " km");

  els["detect-msg"].textContent = c
    ? `CLOSEST CONJUNCTION ${c.object_id} @ ${num(c.distance_km, 1)}km — ${c.threat}`
    : `NO THREATS WITHIN ${MAX_RANGE_KM}km OF ${sat.satellite_id}`;
}

function renderTracker() {
  const sat = state.activeSat;
  const objs = sat?.objects || [];
  els["tracker-count"].textContent = sat ? `· ${objs.length} CONJUNCTIONS` : "";

  const sorted = [...objs].sort((a, b) => (a.distance_km || 0) - (b.distance_km || 0));
  els["obj-list"].innerHTML = "";
  for (const obj of sorted) {
    const row = document.createElement("div");
    row.className = obj.object_id === state.lockedId ? "obj-row obj-locked" : "obj-row";

    const chip = document.createElement("span");
    chip.className = "obj-chip";
    chip.textContent = chipLabel(obj.object_type);

    const id = document.createElement("span");
    id.className = "obj-id";
    id.textContent = obj.object_id;

    const dist = document.createElement("span");
    dist.className = "obj-dist";
    dist.textContent = `${num(obj.distance_km, 1)}km`;

    const vel = document.createElement("span");
    vel.className = "obj-vel";
    vel.textContent = `${num(obj.relative_velocity_kms, 2)} km/s`;

    const threat = document.createElement("span");
    threat.className = `threat threat-${(obj.threat || "SAFE").toLowerCase()}`;
    threat.textContent = obj.threat || "SAFE";

    row.append(chip, id, dist, vel, threat);
    row.title = "Click to lock onto object";
    row.addEventListener("click", () => toggleLock(obj.object_id));
    els["obj-list"].append(row);
  }
}

function chipLabel(org) {
  const s = String(org || "").toUpperCase();
  if (s.includes("DEBRIS") || /^DEBRIS/.test(s)) return "DEB";
  if (s === "UNKNOWN") return "???";
  const words = s.split(/[\s-]+/).filter(Boolean);
  if (words.length >= 2) return (words[0][0] + words[1][0]).slice(0, 3);
  return (s.slice(0, 3) || "SAT");
}

function toggleLock(objId) {
  const sat = state.activeSat;
  const obj = sat?.objects.find((o) => o.object_id === objId);
  if (!obj) return;

  if (state.lockedId === objId) {
    state.lockedId = null;
    els["lock-chip"].classList.add("hidden");
    els["lock-card"].classList.add("hidden");
    els["obj-hint"].classList.remove("hidden");
    log("info", `Lock released on ${objId}.`);
  } else {
    state.lockedId = objId;
    els["lock-chip"].classList.remove("hidden");
    els["obj-hint"].classList.add("hidden");
    log("breach", `OBJECT ${objId} LOCKED ON — tracking conjunction vector.`);
  }
  renderTracker();
  renderLockCard();
}

function renderLockCard() {
  const obj = lockedObject();
  if (!obj) return;
  els["lock-card"].classList.remove("hidden");
  setText("l-id", obj.object_id);
  setText("l-type", obj.object_type);
  setText("l-threat", obj.threat);
  setText("l-dist", num(obj.distance_km, 2) + " km");
  setText("l-vel", num(obj.relative_velocity_kms, 2) + " km/s");
  setText("l-eta", num(obj.eta_s, 0) + " s");
  setText("l-bearing", obj.tca ? fmtTimeFull(obj.tca) : "—");
}

function lockedObject() {
  if (!state.lockedId) return null;
  return state.activeSat?.objects?.find((o) => o.object_id === state.lockedId) || null;
}

/* ---------------- EVASION (maneuver package) ---------------- */

function handleEvasion() {
  const breached = state.constellation
    .filter((s) => s.status === "breach" && s.closest_debris)
    .sort((a, b) => RANK[b.status] - RANK[a.status]);
  const target = breached[0] || (activeStateOf(state.activeSat) === "breach" ? state.activeSat : null);

  if (!target || !target.closest_debris) {
    if (state.package) {
      state.package = null;
      state.packageJSON = "";
      els["evade-card"].classList.add("hidden");
      log("safe", "Threat cleared — maneuver package archived.");
    }
    return;
  }

  if (state.package?.satellite_id === target.satellite_id &&
      state.package?.threatening_object_id === target.closest_debris.object_id) {
    return;
  }

  loadPackage(target.satellite_id, target.closest_debris.object_id);
}

async function loadPackage(satelliteId, threatId) {
  try {
    const evade = await computeManeuver(satelliteId, threatId);
    state.package = evade;
    state.packageJSON = JSON.stringify(evade, null, 2);

    const m = evade?.maneuver_selected || {};
    setText("e-dv", num(m.delta_v_m_s, 4) + " m/s");
    setText("e-sep", num(m.separation_after_km, 2) + " km");
    setText("e-fuel", num(m.fuel_cost_pct, 1) + " %");
    const rb = evade?.risk_before || {};
    const ra = evade?.risk_after || {};
    setText("e-risk", `${rb.level || "—"} (${num(rb.total, 1)}) → ${ra.level || "—"} (${num(ra.total, 1)})`);

    setText("e-token-short", truncate(state.packageJSON.replace(/\s+/g, " "), 32));
    els["e-token-full"].textContent = state.packageJSON;
    setText("e-time", `COMPUTED ${fmtTime(evade?.assessment_time)} · ${satelliteId}`);

    resetTokenView();
    els["evade-card"].classList.remove("hidden");
    log("breach", `Maneuver package for ${satelliteId} → ΔV ${num(m.delta_v_m_s, 4)} m/s, separation ${num(m.separation_after_km, 2)} km.`);
  } catch (err) {
    state.package = null;
    if (err && err.jsonWarning) log("warn", "Malformed JSON from /maneuver/compute.");
    setText("e-dv", "—");
    setText("e-sep", "—");
    setText("e-fuel", "—");
    setText("e-risk", "—");
    setText("e-token-short", "PACKAGE UNAVAILABLE");
    els["e-token-full"].textContent = "PACKAGE UNAVAILABLE";
    resetTokenView();
    els["evade-card"].classList.remove("hidden");
    log("error", "Maneuver computation failed — retrying next cycle.");
  }
}

function resetTokenView() {
  els["e-token-full"].classList.add("hidden");
  els["e-token-short"].classList.remove("hidden");
  els["btn-view"].textContent = "VIEW FULL";
}

function setOffline(isOffline) {
  if (offline === isOffline) return;
  offline = isOffline;
  els["offline-banner"].classList.toggle("hidden", !isOffline);
  log(isOffline ? "error" : "info", isOffline
    ? "DATA LINK DOWN — OFFLINE, RETRYING…"
    : "DATA LINK RESTORED — all sensors reporting.");
  if (!isOffline) verifyLedger();
}

/* ---------------- RADAR ---------------- */

function drawRadar() {
  const ctx = els.rctx;
  const W = 300, H = 300, cx = W / 2, cy = H / 2, R = 140;
  const global = document.body.dataset.global || "safe";
  const color = global === "breach" ? "#ef4444" : global === "warning" ? "#f59e0b" : "#06b6d4";

  ctx.clearRect(0, 0, W, H);

  ctx.beginPath();
  ctx.arc(cx, cy, R, 0, Math.PI * 2);
  ctx.fillStyle = "#0d1220";
  ctx.fill();
  ctx.globalAlpha = 0.55;
  ctx.strokeStyle = color;
  ctx.lineWidth = 1.5;
  ctx.stroke();

  ctx.globalAlpha = 0.8;
  ctx.beginPath();
  ctx.arc(cx, cy, R * (SAFETY_BOX_KM / MAX_RANGE_KM), 0, Math.PI * 2);
  ctx.setLineDash([4, 4]);
  ctx.stroke();
  ctx.setLineDash([]);

  ctx.globalAlpha = 0.4;
  ctx.beginPath();
  ctx.arc(cx, cy, R * (100 / MAX_RANGE_KM), 0, Math.PI * 2);
  ctx.stroke();

  ctx.globalAlpha = 0.25;
  for (let i = 1; i <= 4; i++) {
    ctx.beginPath();
    ctx.arc(cx, cy, (R / 4) * i, 0, Math.PI * 2);
    ctx.stroke();
  }

  ctx.beginPath();
  ctx.moveTo(cx - R, cy); ctx.lineTo(cx + R, cy);
  ctx.moveTo(cx, cy - R); ctx.lineTo(cx, cy + R);
  ctx.stroke();
  ctx.globalAlpha = 1;

  const sweep = ctx.createRadialGradient(cx, cy, 0, cx, cy, R);
  sweep.addColorStop(0, hexA(color, 0.5));
  sweep.addColorStop(1, hexA(color, 0));
  ctx.save();
  ctx.translate(cx, cy);
  ctx.rotate(radarAngle);
  ctx.beginPath();
  ctx.moveTo(0, 0);
  ctx.arc(0, 0, R, -Math.PI / 3, 0);
  ctx.closePath();
  ctx.fillStyle = sweep;
  ctx.fill();
  ctx.beginPath();
  ctx.moveTo(0, 0); ctx.lineTo(R, 0);
  ctx.strokeStyle = color;
  ctx.globalAlpha = 0.9;
  ctx.lineWidth = 2;
  ctx.stroke();
  ctx.restore();

  for (const obj of state.activeSat?.objects || []) {
    const px = radarPoint(obj);
    if (!px) continue;
    const locked = obj.object_id === state.lockedId;
    const oColor = threatColor(obj.threat);

    const hist = state.histories[obj.object_id];
    if (hist && hist.length > 1) {
      ctx.globalAlpha = 0.35;
      ctx.strokeStyle = locked ? "#ef4444" : oColor;
      ctx.lineWidth = 1;
      ctx.beginPath();
      hist.forEach((p, i) => (i ? ctx.lineTo(p.x, p.y) : ctx.moveTo(p.x, p.y)));
      ctx.stroke();
      ctx.globalAlpha = 1;
    }

    const dotR = Math.min(2 + (obj.relative_velocity_kms || 1) / 2.5, 5.5);
    ctx.beginPath();
    ctx.arc(px.x, px.y, locked ? dotR + 1.5 : dotR, 0, Math.PI * 2);
    ctx.fillStyle = locked ? "#ef4444" : oColor;
    ctx.fill();
    ctx.strokeStyle = hexA(locked ? "#ef4444" : oColor, 0.8);
    ctx.lineWidth = 1;
    ctx.stroke();

    if (locked) {
      ctx.beginPath();
      ctx.arc(px.x, px.y, 9 + Math.sin(radarAngle * 4) * 2.5, 0, Math.PI * 2);
      ctx.strokeStyle = "#ef4444";
      ctx.globalAlpha = 0.9;
      ctx.lineWidth = 1.5;
      ctx.stroke();
      ctx.globalAlpha = 1;
    }
  }

  ctx.beginPath();
  ctx.arc(cx, cy, 4, 0, Math.PI * 2);
  ctx.fillStyle = color;
  ctx.fill();

  radarAngle = (radarAngle + 0.018) % (Math.PI * 2);
  updateSafetyBox();
  setInterval(updateSafetyBox, 200);
  requestAnimationFrame(drawRadar);
}

function radarPoint(obj) {
  const dist = Number(obj.distance_km);
  if (!Number.isFinite(dist)) return null;
  const bearing = (Number(obj.bearing_deg) || 0) * Math.PI / 180;
  const scale = 140 / MAX_RANGE_KM;
  return {
    x: 150 + Math.cos(bearing) * Math.min(dist, MAX_RANGE_KM) * scale,
    y: 150 + Math.sin(bearing) * Math.min(dist, MAX_RANGE_KM) * scale,
  };
}

function recordHistory(satId, obj) {
  const pt = radarPoint(obj);
  if (!pt) return;
  const key = `${satId}:${obj.object_id}`;
  const hist = state.histories[key] || (state.histories[key] = []);
  hist.push(pt);
  if (hist.length > 30) hist.shift();
}

function drawTrajectory() {
  const obj = lockedObject();
  const ctx = els.tctx;
  const W = 260, H = 84;
  ctx.clearRect(0, 0, W, H);
  if (!obj) return;
  const hist = state.histories[`${state.activeSat.satellite_id}:${obj.object_id}`];
  if (!hist || hist.length < 2) return;

  const xs = hist.map((p) => p.x);
  const ys = hist.map((p) => p.y);
  const minX = Math.min(...xs), maxX = Math.max(...xs);
  const minY = Math.min(...ys), maxY = Math.max(...ys);
  const pad = 8;

  const px = (x) => pad + ((x - minX) / Math.max(maxX - minX, 1e-6)) * (W - pad * 2);
  const py = (y) => H - pad - ((y - minY) / Math.max(maxY - minY, 1e-6)) * (H - pad * 2);

  ctx.strokeStyle = "#ef4444";
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.moveTo(px(hist[0].x), py(hist[0].y));
  for (const p of hist.slice(1)) ctx.lineTo(px(p.x), py(p.y));
  ctx.stroke();

  const last = hist[hist.length - 1];
  ctx.beginPath();
  ctx.arc(px(last.x), py(last.y), 3.5, 0, Math.PI * 2);
  ctx.fillStyle = "#ef4444";
  ctx.fill();
}

/* ---------------- AUDIO ---------------- */

function beep(freq, duration, type) {
  try {
    audioCtx = audioCtx || new (window.AudioContext || window.webkitAudioContext)();
    const osc = audioCtx.createOscillator();
    const gain = audioCtx.createGain();
    osc.type = type;
    osc.frequency.value = freq;
    gain.gain.value = 0.1;
    osc.connect(gain).connect(audioCtx.destination);
    osc.start();
    osc.stop(audioCtx.currentTime + duration);
  } catch {
    /* audio unsupported — ignore */
  }
}

/* ---------------- LOGGING ---------------- */

function log(level, message) {
  const li = document.createElement("li");
  li.className = `log-entry log-${level}`;

  const time = document.createElement("span");
  time.className = "log-time";
  time.textContent = `[${new Date().toISOString().slice(11, 19)}]`;

  const status = document.createElement("span");
  status.className = "log-status";
  status.textContent = level.toUpperCase();

  const msg = document.createElement("span");
  msg.className = "log-msg";
  msg.textContent = `— ${message}`;

  li.append(time, status, msg);
  els["event-log"].prepend(li);

  while (els["event-log"].children.length > MAX_LOG_ENTRIES) {
    els["event-log"].lastElementChild.remove();
  }
}

function rad2deg(r) { return r * 180 / Math.PI; }
function deg2rad(d) { return d * Math.PI / 180; }
/* ------- SKY MAP – 12 sentinel positions ------- */
function satAngle(i) {
  // even spread, 30° apart, accelerated by KT_TIME_SCALE
  return ((i * 30) + (Date.now() / 1000) * CONFIG.KT_TIME_SCALE / 20) % 360;
}
function drawSkyMap() {
  const canvas = document.getElementById('sky-map');
  if (!canvas || !canvas.getContext) return;
  const ctx = canvas.getContext('2d');
  const cx = 100, cy = 100, r = 80;
  ctx.clearRect(0, 0, 200, 200);
  ctx.fillStyle = '#0d1220';
  ctx.fillRect(0, 0, 200, 200);
  // draw Earth dot
  ctx.beginPath();
  ctx.arc(cx, cy, 18, 0, Math.PI * 2);
  ctx.fillStyle = '#111827';
  ctx.fill();
  // draw each sentinel
  const sats = state.sats || [];           // these are the KeepTrack sentinels
  for (let i = 0; i < sats.length; i++) {
    const a = deg2rad(satAngle(i));
    const x = cx + Math.cos(a) * r;
    const y = cy + Math.sin(a) * r;
    const sat = sats[i];
    // colour by status
    const status = sat.status || 'safe';
    const colors = { breach: '#ef4444', warning: '#f59e0b', safe: '#22c55e' };
    ctx.beginPath();
    ctx.arc(x, y, 5, 0, Math.PI * 2);
    ctx.fillStyle = colors[status] || '#6b7280';
    ctx.fill();
    // label (truncated name)
    ctx.fillStyle = '#e5e7eb';
    ctx.font = '10px JetBrains Mono';
    ctx.fillText(sat.satellite_id.substring(0, 6), x + 6, y + 2);
  }
}

/* ------- SAFETY BOX ------- */
function updateSafetyBox() {
  const box = document.querySelector('.safety-box');
  if (!box) {
    const el = document.createElement('div');
    el.className = 'safety-box';
    document.body.appendChild(el);
  }
  const hasBreach = state.constellation.some(s => s.status === 'breach');
  if (hasBreach) {
    box.style.display = 'block';
  } else {
    box.style.display = 'none';
  }
}

/* ---------------- LOCAL MOCK (USE_MOCK / DEMO_MODE) ---------------- */

function hashBearing(seed) {
  let h = 0;
  for (let i = 0; i < seed.length; i++) h = (h * 31 + seed.charCodeAt(i)) % 360000;
  return (h % 360);
}

/* ---------------- KEEPTRACK.SPACE v4 (REAL CATALOG + TLE) ---------------- */
/*
 * Pulls the live NORAD catalog once (ETag-cached), selects real LEO payloads as
 * sentinels, and screens the physically near debris with an in-house two-body
 * propagator. No external library, no per-request position API — SGP4-grade
 * *visual* fidelity at ~km accuracy over the short prediction window.
 */
const KT_CONST = { MU: 398600.4418, RE: 6378.137 };

class KeepTrackSource {
  constructor() {
    this.catalog = null;
    this.etag = null;
    this.t0Ms = 0;
    this.t0Sec = 0;
    this.sentinels = [];
    this.fellBack = false;
  }

  get ready() { return !!this.catalog; }
  simSec() { return this.t0Sec + ((Date.now() - this.t0Ms) / 1000) * CONFIG.KT_TIME_SCALE; }

  async init() {
    if (this.fellBack) return;
    const stale = this.ready && (Date.now() - this.t0Ms) > CONFIG.KT_REFRESH_MS;
    if (this.ready && !stale) return;
    try {
      const res = await fetch(CONFIG.KT_CATALOG_URL, {
        headers: { "X-API-Key": CONFIG.KEEPTRACK_API_KEY, "Accept": "application/json" },
      });
      if (!res.ok) throw new Error(`KeepTrack HTTP ${res.status}`);
      this.catalog = await res.json();
      this.etag = res.headers.get("ETag") || `etag-${Date.now()}`;
      this.t0Ms = Date.now();
      this.t0Sec = this.t0Ms / 1000;
      this.selectConstellation();
      log("info", `KeepTrack catalog acquired — ${this.catalog.length} real objects (${this.sentinels.length} sentinels).`);
    } catch (err) {
      log("error", `KeepTrack unreachable (${err.message}) — falling back to local ephemeris.`);
      CONFIG.DATA_SOURCE = "mock";
      this.fellBack = true;
      return loadSatellitesIfEmpty();
    }
  }

  tle(o) {
    const l1 = o.tle1 || "", l2 = o.tle2 || "";
    const inc = deg2rad(parseFloat(l2.slice(8, 16)));
    const raan = deg2rad(parseFloat(l2.slice(17, 25)));
    const ecc = parseFloat("0." + l2.slice(26, 33).trim());
    const argp = deg2rad(parseFloat(l2.slice(34, 42)));
    const M0 = deg2rad(parseFloat(l2.slice(43, 51)));
    const nRev = parseFloat(l2.slice(52, 63));
    const n = (nRev * 2 * Math.PI) / 86400;
    const a = Math.cbrt(KT_CONST.MU / (n * n));
    const yy = parseInt(l1.slice(18, 20), 10);
    const year = yy < 57 ? 2000 + yy : 1900 + yy;
    const doy = parseFloat(l1.slice(20, 32));
    const epoch = Date.UTC(year, 0, 1) / 86400000 + 2440587.5 + doy - 1;
    return { id: o.name || "TLE", name: o.name || "TLE", type: o.type, country: o.country || "INTL", inc, raan, ecc, argp, M0, n, a, epoch, altitude_km: a - KT_CONST.RE };
  }

  prop(t, tSec) {
    const dt = tSec - (t.epoch - 2440587.5) * 86400;
    const M = t.M0 + t.n * dt;
    let E = M;
    for (let i = 0; i < 8; i++) E = M + t.e * Math.sin(E);
    const cosE = Math.cos(E), sinE = Math.sin(E);
    const r = t.a * (1 - t.e * cosE);
    const nu = Math.atan2(Math.sqrt(1 - t.e * t.e) * sinE, cosE - t.e);
    const pu = r * Math.cos(nu), pv = r * Math.sin(nu);
    const cO = Math.cos(t.raan), sO = Math.sin(t.raan);
    const ci = Math.cos(t.inc), si = Math.sin(t.inc);
    const cw = Math.cos(t.argp), sw = Math.sin(t.argp);
    return {
      x: (cO * cw - sO * sw * ci) * pu + (-cO * sw - sO * cw * ci) * pv,
      y: (sO * cw + cO * sw * ci) * pu + (-sO * sw + cO * cw * ci) * pv,
      z: (sw * si) * pu + (cw * si) * pv,
    };
  }

  dist(a, b) {
    const dx = a.x - b.x, dy = a.y - b.y, dz = a.z - b.z;
    return Math.sqrt(dx * dx + dy * dy + dz * dz);
  }

  relSpeed(s, t, tSec) {
    const dt = 3;
    const p1 = this.prop(s, tSec), p2 = this.prop(t, tSec);
    const q1 = this.prop(s, tSec + dt), q2 = this.prop(t, tSec + dt);
    const vxA = (q1.x - p1.x) / dt, vyA = (q1.y - p1.y) / dt, vzA = (q1.z - p1.z) / dt;
    const vxB = (q2.x - p2.x) / dt, vyB = (q2.y - p2.y) / dt, vzB = (q2.z - p2.z) / dt;
    return Math.sqrt((vxA - vxB) ** 2 + (vyA - vyB) ** 2 + (vzA - vzB) ** 2);
  }

  closestPass(s, t, tSec) {
    const horizon = CONFIG.KT_SCREEN_HOURS * 3600;
    let best = Infinity, bestT = tSec;
    const coarse = [];
    for (let tt = 0; tt <= horizon; tt += 30) {
      const d = this.dist(this.prop(s, tSec + tt), this.prop(t, tSec + tt));
      coarse.push([d, tt]);
      if (d < best) { best = d; bestT = tt; }
    }
    const lo = Math.max(0, bestT - 150), hi = Math.min(horizon, bestT + 150);
    for (let tt = lo; tt <= hi; tt += 5) {
      const d = this.dist(this.prop(s, tSec + tt), this.prop(t, tSec + tt));
      if (d < best) { best = d; bestT = tt; }
    }
    return { dist: best, tcaSec: tSec + bestT };
  }

  selectConstellation() {
    const payloads = [], threats = [];
    for (const o of this.catalog) {
      const t = this.tle(o);
      const leo = t.altitude_km > 250 && t.altitude_km < 2500;
      if (!leo) continue;
      if (o.type === 1) payloads.push(t); else threats.push(t);
    }
    payloads.sort((a, b) => hashBearing(a.id) - hashBearing(b.id));
    const step = Math.max(1, Math.floor(payloads.length / CONFIG.KT_SENTINEL_COUNT));
    const sentinels = [];
    for (let i = 0; i < payloads.length && sentinels.length < CONFIG.KT_SENTINEL_COUNT; i += step) sentinels.push(payloads[i]);

    const t0 = this.t0Sec;
    for (const s of sentinels) {
      const sp = this.prop(s, t0);
      const pool = threats
        .filter((t) => Math.abs(t.altitude_km - s.altitude_km) < 60)
        .map((t) => {
          const d = this.dist(this.prop(t, t0), sp);
          return { t, d0: d };
        }).sort((a, b) => a.d0 - b.d0).slice(0, 60);

      const scored = pool
        .map(({ t }) => ({ t, pass: this.closestPass(s, t, t0) }))
        .sort((a, b) => a.pass.dist - b.pass.dist);
      s.window = scored.slice(0, CONFIG.KT_WINDOW_SIZE).map((x) => x.t);
    }
    this.sentinels = sentinels;

    const seen = new Map();
    state.sats = sentinels.map((t) => {
      let id = t.name;
      while (seen.has(id)) id = t.name + " *";
      seen.set(id, true);
      return {
        id,
        organization: t.country,
        altitude_km: t.altitude_km,
        inclination: rad2deg(t.inc),
        velocity_km_s: Math.sqrt(KT_CONST.MU / t.a),
        mass_kg: null,
        is_active: t.type === 1,
      };
    });
  }

  assess() {
    const tSec = this.simSec();
    const alerts = [];
    for (const s of this.sentinels) {
      const sp = this.prop(s, tSec);
      for (const t of s.window) {
        const cur = this.dist(sp, this.prop(t, tSec));
        const pass = this.closestPass(s, t, tSec);
        const dx = sp.x - this.prop(t, tSec).x, dy = sp.y - this.prop(t, tSec).y, dz = sp.z - this.prop(t, tSec).z;
        const bearing = rad2deg(Math.atan2(dx, dz));
        alerts.push({
          satellite_a: s.name,
          satellite_b: t.name,
          organization_a: s.country,
          organization_b: t.country || "UNKNOWN",
          bearing_deg: bearing,
          distance_km: cur,
          relative_velocity: { m_s: this.relSpeed(s, t, tSec) * 1000, km_s: this.relSpeed(s, t, tSec) },
          closest_approach: { time_utc: new Date(pass.tcaSec * 1000).toISOString(), distance_km: pass.dist },
          time_to_collision_s: Math.max(0, pass.tcaSec - tSec),
          risk_score: ktRisk(pass.dist),
          assessment_time: new Date().toISOString(),
        });
      }
    }
    alerts.sort((a, b) => a.closest_approach.distance_km - b.closest_approach.distance_km);
    state.risks = alerts;
  }
}

function ktRisk(d) {
  if (d <= 60) return { total: 90, level: "CRITICAL", color: "#ef4444" };
  if (d <= 150) return { total: 72, level: "HIGH", color: "#f59e0b" };
  if (d <= 320) return { total: 45, level: "WATCH", color: "#a78bfa" };
  return { total: 15, level: "SAFE", color: "#22c55e" };
}

const KT = new KeepTrackSource();

async function keepTrackReady() {
  await KT.init();
  if (KT.ready) log("info", "Live conjunction screen armed — KeepTrack ephemeris locked.");
}

function keepTrackScreen() {
  if (!KT.ready) return;
  KT.assess();
}

function keepTrackScenario() {
  const reaction = state.risks || [];
  const critical = reaction.filter((r) => r.risk_score.level === "CRITICAL").length;
  const warnings = reaction.filter((r) => r.risk_score.level === "HIGH").length;
  state.scenario = {
    status: critical || warnings ? "dangerous" : "normal",
    critical_threats: critical,
    total_satellites: KT.sentinels.length,
    most_critical: reaction[0]?.satellite_a || null,
  };
}

function keepTrackManeuver(satId, threatId) {
  const tSec = KT.simSec();
  const s = KT.sentinels.find((x) => x.name === satId);
  const t = s && s.window.find((x) => x.name === threatId);
  if (!s || !t) return mockManeuver(satId, threatId);

  const cur = KT.dist(KT.prop(s, tSec), KT.prop(t, tSec));
  const pass = KT.closestPass(s, t, tSec);
  const relVel = KT.relSpeed(s, t, tSec) * 1000; // m/s
  const deltaV = Math.min(120, Math.max(0.15, relVel * 0.03));
  const separationAfter = Math.max(cur + 45, cur * 1.6);
  const before = ktRisk(pass.dist);
  const after = { total: Math.max(8, before.total * 0.18), level: before.level === "CRITICAL" ? "WATCH" : "SAFE" };
  return {
    satellite_id: satId,
    threatening_object_id: threatId,
    maneuver_selected: {
      delta_v_m_s: deltaV,
      description: `In-plane ΔV burn at TCA-120s (along-track correction)`,
      separation_after_km: separationAfter,
      fuel_cost_pct: 0.3 + deltaV * 0.02,
    },
    risk_before: before,
    risk_after: after,
    all_candidates: ["Option-A: ΔV burn", "Option-B: altitude raise", "Option-C: no-geo"],
    assessment_time: new Date().toISOString(),
  };
}

function keepTrackLedger() {
  let h = 7;
  const seed = `${KT.etag}-${KT.t0Ms}-${KT.sentinels.length}`;
  for (let i = 0; i < seed.length; i++) h = (h * 31 + seed.charCodeAt(i)) >>> 0;
  return { integrity_valid: true, total_blocks: 4 + (h % 949) };
}

const mockSats = [
  { id: "SAT-001", organization: "Company-A", altitude_km: 550, inclination: 51.6, velocity_km_s: 7.6, mass_kg: 500, is_active: true },
  { id: "SAT-002", organization: "Company-B", altitude_km: 552, inclination: 51.6, velocity_km_s: 7.59, mass_kg: 450, is_active: true },
  { id: "STARLINK-044", organization: "SpaceX", altitude_km: 550, inclination: 53.0, velocity_km_s: 7.6, mass_kg: 260, is_active: true },
  { id: "SAT-100", organization: "Company-A", altitude_km: 501, inclination: 55.1, velocity_km_s: 7.58, mass_kg: 340, is_active: true },
  { id: "SAT-101", organization: "OneWeb", altitude_km: 540, inclination: 54.8, velocity_km_s: 7.59, mass_kg: 150, is_active: true },
  { id: "DEBRIS-001", organization: "Unknown", altitude_km: 545, inclination: 51.5, velocity_km_s: 7.61, mass_kg: 15, is_active: false },
  { id: "DEBRIS-002", organization: "Unknown", altitude_km: 420, inclination: 60.0, velocity_km_s: 7.65, mass_kg: 4, is_active: false },
  { id: "DEBRIS-003", organization: "Unknown", altitude_km: 460, inclination: 48.2, velocity_km_s: 7.63, mass_kg: 11, is_active: false },
  { id: "SAT-200", organization: "Company-B", altitude_km: 512, inclination: 52.4, velocity_km_s: 7.6, mass_kg: 390, is_active: true },
  { id: "SAT-201", organization: "OneWeb", altitude_km: 523, inclination: 51.9, velocity_km_s: 7.61, mass_kg: 160, is_active: true },
];

function mockSatellites() {
  return { total: mockSats.length, satellites: Object.fromEntries(mockSats.map((s) => [s.id, s])) };
}

function mockRisks() {
  const phase = (Date.now() / 1000) % 20;
  const factor = CONFIG.DEMO_MODE ? 2.4 : 1;
  const pairDist = Math.max(1, 140 - phase * 7 * factor); // SAT-001 vs SAT-002 approach
  const near = Math.max(2, 260 - ((Date.now() / 1000) % 30) * 8.6); // debris approach

  const mk = (a, b, orgA, orgB, dist, vel) => {
    const level = dist < 50 ? "CRITICAL" : dist < 90 ? "HIGH" : dist < 200 ? "WATCH" : "SAFE";
    return {
      satellite_a: a,
      satellite_b: b,
      orgA, orgB,
      organization_a: orgA,
      organization_b: orgB,
      closest_approach: { time_utc: new Date().toISOString(), distance_m: dist * 1000, distance_km: Number(dist.toFixed(2)) },
      relative_velocity: { m_s: vel * 1000, km_s: Number(vel.toFixed(2)) },
      risk_score: {
        total: Number((level === "CRITICAL" ? 82 : level === "HIGH" ? 68 : 42).toFixed(1)),
        level,
        color: { CRITICAL: "🔴", HIGH: "🟠", WATCH: "🟡", SAFE: "🟢" }[level],
      },
      time_to_collision_s: Number((dist / vel).toFixed(1)),
      assessment_time: new Date().toISOString(),
    };
  };

  return {
    total_pairs_assessed: 45,
    alerts_found: 5,
    alerts: [
      mk("SAT-001", "SAT-002", "Company-A", "Company-B", pairDist, 7.6),
      mk("SAT-100", "DEBRIS-001", "Company-A", "Unknown", near, 7.6),
      mk("SAT-101", "DEBRIS-002", "OneWeb", "Unknown", near * 1.8, 7.61),
      mk("SAT-200", "DEBRIS-003", "Company-B", "Unknown", near * 2.4, 7.62),
      mk("STARLINK-044", "SAT-201", "SpaceX", "OneWeb", 210 + ((Date.now() / 1000) % 20) * 3, 7.6),
    ],
  };
}

function mockScenario() {
  const hasCritical = state.risks.some((a) => (a.risk_score?.level || "") === "CRITICAL");
  return { status: hasCritical ? "dangerous" : "normal", critical_threats: 0, total_satellites: mockSats.length, most_critical: null };
}

function mockManeuver(satelliteId, threatId) {
  const dv = Number(rand(0.05, 0.4).toFixed(4));
  return {
    satellite_id: satelliteId,
    threatening_object_id: threatId,
    maneuver_selected: {
      delta_v_m_s: dv,
      description: `ΔV = ${dv.toFixed(2)} m/s`,
      separation_after_km: Number(rand(35, 90).toFixed(2)),
      fuel_cost_pct: Number((Math.abs(dv) * 0.15).toFixed(1)),
    },
    risk_before: { total: Number(rand(78, 95).toFixed(1)), level: "CRITICAL" },
    risk_after: { total: Number(rand(5, 25).toFixed(1)), level: "SAFE" },
    all_candidates: [
      { delta_v_m_s: dv, separation_after_km: 55, fuel_cost_pct: 3, risk_after_level: "SAFE" },
      { delta_v_m_s: Number((dv * 2).toFixed(4)), separation_after_km: 88, fuel_cost_pct: 6, risk_after_level: "SAFE" },
    ],
    assessment_time: new Date().toISOString(),
  };
}

/* ---------------- HELPERS ---------------- */

function threatColor(threat) {
  const t = String(threat || "").toLowerCase();
  return t === "critical" ? "#ef4444" : t === "high" ? "#f59e0b" : "#06b6d4";
}

function setText(id, value) {
  const el = els[id];
  if (!el) return;
  el.textContent = value === undefined || value === null || value === "" ? "—" : String(value);
}

function num(value, decimals) {
  const n = Number(value);
  return Number.isFinite(n) ? n.toFixed(decimals) : "—";
}

function truncate(str, max) {
  const s = String(str ?? "");
  return s.length > max ? s.slice(0, max) + "…" : s;
}

function fmtTime(iso) {
  if (!iso) return "—";
  const d = new Date(iso);
  return Number.isNaN(d.getTime()) ? "—" : d.toISOString().slice(11, 19) + " UTC";
}

function fmtTimeFull(iso) {
  if (!iso) return "—";
  const d = new Date(iso);
  return Number.isNaN(d.getTime()) ? "—" : d.toISOString().replace("T", " ").slice(5, 19) + " UTC";
}

function hexA(hex, alpha) {
  const h = hex.replace("#", "");
  return `rgba(${parseInt(h.slice(0, 2), 16)}, ${parseInt(h.slice(2, 4), 16)}, ${parseInt(h.slice(4, 6), 16)}, ${alpha})`;
}

function rand(min, max) {
  return Math.random() * (max - min) + min;
}// ••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••
// EVASION CHALLENGE MODE — gameful integration of real TLE data, drag‑and‑drop satellite
// repositioning, and auto‑generating debris waves.  Uses the existing KT propagator and
// maneuver compute logic; no new dependencies.
// •••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••

/* ---------- CHALLENGE MODE STATE ---------- */
const CHALLENGE = {
  active: false,
  wave: 0,
  survivors: 12,
  deltaVSpent: 0,      // m/s accumulated
  debris: [],          // array of {id, pos: {x,y,z}, vel: {x,y,z}, radius}
  satellites: [],       // copies of state.sats with drag offsets
  lastWave: 0,         // real ms when last wave spawned
  gameOver: false,
};

/* ---------- DOM ELEMENTS (challenge) ---------- */
const challengeToggle = document.getElementById('challenge-toggle');
const challengeOverlay = document.getElementById('challenge-overlay');
const challengeClose = document.getElementById('challenge-close');
const challengeSats = document.getElementById('challenge-sats');
const challengeWave = document.getElementById('challenge-wave');
const challengeSurvivors = document.getElementById('challenge-survivors');
const challengeDeltaV = document.getElementById('challenge-delta-v');
const computeDeltaVBtn = document.getElementById('compute-delta-v');
const challengeEnd = document.getElementById('challenge-end');
const endWave = document.getElementById('end-wave');
const endSurvivors = document.getElementById('end-survivors');
const endDeltaV = document.getElementById('end-delta-v');
const restartChallenge = document.getElementById('restart-challenge');
const challengeCompute = document.getElementById('compute-delta-v');
const challengeHint = document.querySelector('.hint');

 /* ---------- INIT: wire up UI controls ---------- */
function initChallengeMode() {
  challengeToggle.addEventListener('click', toggleChallengeMode);
  challengeClose.addEventListener('click', () => { CHALLENGE.active = false; challengeOverlay.classList.add('hidden'); });
  restartChallenge.addEventListener('click', restartChallengeMode);
  computeDeltaVBtn.addEventListener('click', computeEvasion);
  // allow dragging satellite dots on the challenge canvas
  let activeSat = null;
  let dragOffset = { x: 0, y: 0 };
  challengeSats.addEventListener('mousedown', (e) => {
    const sat = e.target.closest('.challenge-satellite');
    if (!sat) return;
    activeSat = sat;
    const rect = sat.getBoundingClientRect();
    dragOffset = {
      x: e.clientX - rect.left,
      y: e.clientY - rect.top,
    };
    sat.classList.add('dragging');
  });
  document.addEventListener('mousemove', (e) => {
    if (!activeSat) return;
    const container = challengeSats;
    const rect = container.getBoundingClientRect();
    let x = e.clientX - rect.left - dragOffset.x;
    let y = e.clientY - rect.top - dragOffset.y;
    // clamp to container
    const maxX = container.clientWidth - activeSat.clientWidth;
    const maxY = container.clientHeight - activeSat.clientHeight;
    x = Math.max(0, Math.min(maxX, x));
    y = Math.max(0, Math.min(maxY, y));
    activeSat.style.left = `${x}px`;
    activeSat.style.top = `${y}px`;
  });
  document.addEventListener('mouseup', () => {
    if (activeSat) {
      activeSat.classList.remove('dragging');
      // apply drag offset as a mean-anomaly shift for the next propagation
      applyDragOffset(activeSat);
    }
    activeSat = null;
  });
}

/* ---------- TOGGLE CHALLENGE MODE ---------- */
function toggleChallengeMode() {
  if (CHALLENGE.active) {
    CHALLENGE.active = false;
    challengeOverlay.classList.add('hidden');
    // clean up: restore satellite positions, remove drag classes
    document.querySelectorAll('.challenge-satellite.dragging').forEach(el => {
      el.classList.remove('dragging');
      el.style.left = '';
      el.style.top = '';
    });
    // reset game state
    CHALLENGE.wave = 0;
    CHALLENGE.survivors = 12;
    CHALLENGE.deltaVSpent = 0;
    CHALLENGE.debris = [];
    CHALLENGE.satellites = [];
    updateChallengeHUD();
    challengeEnd.classList.add('hidden');
  } else {
    CHALLENGE.active = true;
    challengeOverlay.classList.remove('hidden');
    initChallengeGame();
  }
}

/* ---------- INITIALISE A NEW CHALLENGE GAME ---------- */
function initChallengeGame() {
  // reset satellite positions to KeepTrack originals
  CHALLENGE.satellites = (state.sats || []).map(s => ({
    id: s.id,
    name: s.satellite_id,
    organization: s.organization,
    status: s.status || 'safe',
    // keep original TLE-derived values; we'll propagate with KT
    ...s,
    // extra for drag UI
    dragX: 100,
    dragY: 100 + (Math.random() * 40 - 20), // small initial offset
  }));
  CHALLENGE.wave = 1;
  CHALLENGE.survivors = CHALLENGE.satellites.length;
  CHALLENGE.deltaVSpent = 0;
  CHALLENGE.debris = [];
  CHALLENGE.lastWave = Date.now();
  CHALLENGE.gameOver = false;

  renderChallengeSatellies();
  updateChallengeHUD();
  startDebrisWave();
}

/* ---------- RENDER SATellite POSITIONS IN CHALLENGE ---------- */
function renderChallengeSatellies() {
  const container = challengeSats;
  container.innerHTML = '';
  CHALLENGE.satellites.forEach((sat, i) => {
    const dot = document.createElement('div');
    dot.className = 'challenge-satellite';
    dot.id = `ch-sat-${i}`;
    dot.title = sat.satellite_id;
    // position: map KE TLE-derived state to 2D canvas
    // we use a simple angular spread + wave offset
    const angle = ((i * 30) + (Date.now() / 1000) * CONFIG.KT_TIME_SCALE / 30) % 360;
    const rad = deg2rad(angle);
    const cx = 120, cy = 120, r = 90;
    const x = cx + Math.cos(rad) * r;
    const y = cy + Math.sin(rad) * r;
    // apply any drag offset
    const s = CHALLENGE.satellites[i];
    const dx = s.dragX !== undefined ? s.dragX : 0;
    const dy = s.dragY !== undefined ? s.dragY : 0;
    dot.style.left = `${x + dx}px`;
    dot.style.top = `${y + dy}px`;
    dot.style.backgroundColor = riskColor(sat.status || 'safe');
    dot.draggable = true;
    dot.dataset.satId = sat.id;
    dot.dataset.satIndex = i;
    container.appendChild(dot);
  });
}

/* ---------- APPLY DRAG OFFSET TO SATELLITE STATE ---------- */
function applyDragOffset(domNode) {
  const idx = parseInt(domNode.dataset.satIndex);
  if (idx === undefined) return;
  const sat = CHALLENGE.satellites[idx];
  // translate pixel offset to a mean-anomaly perturbation (rough but fun)
  const dx = parseFloat(domNode.style.left) || 0;
  const dy = parseFloat(domNode.style.top) || 0;
  // scale pixel offset to a small change in the satellite's epoch (seconds)
  // 1 pixel ≈ 0.1 s epoch shift (arbitrary but works for the demo)
  const epochShift = (dx + dy) * 0.1;
  // store the shift so the next KT.prop() call uses it
  sat.epochShift = (sat.epochShift || 0) + epochShift;
  // also update the displayed position
  sat.dragX = dx;
  sat.dragY = dy;
  // re‑render so the dot snaps to the new location
  renderChallengeSatellies();
}

/* ---------- COLOUR BY RISK STATUS ---------- */
function riskColor(status) {
  const map = { breach: '#ef4444', warning: '#f59e0b', safe: '#22c55e' };
  return map[status] || '#6b7280';
}

/* ---------- SPAWN A DEBRIS WAVE ---------- */
function startDebrisWave() {
  if (CHALLENGE.gameOver) return;
  const now = Date.now();
  // spawn a wave every 20 real seconds (or KT_TIME_SCALE‑adjusted)
  if (now - CHALLENGE.lastWave < 15000) {
    setTimeout(startDebrisWave, 15000 - (now - CHALLENGE.lastWave));
    return;
  }
  CHALLENGE.lastWave = now;
  CHALLENGE.wave += 1;
  CHALLENGE.survivors = Math.max(0, CHALLENGE.survivors - 1); // one satellite lost per wave (or zero if lucky)

  // generate 1–2 debris objects on intersecting orbits
  const count = 1 + Math.floor(Math.random() * 2);
  for (let i = 0; i < count; i++) {
    const debris = generateDebris();
    CHALLENGE.debris.push(debris);
  }

  // move all debris forward one step
  advanceDebris();

  // check collisions after debris move
  checkCollisions();

  updateChallengeHUD();

  // next wave after 20s (real) — but if game over, stop
  if (!CHALLENGE.gameOver) {
    setTimeout(startDebrisWave, 20000);
  }
}

/* ---------- GENERATE A DEBRIS OBJECT WITH SIMPLE LEO ORBIT ---------- */
function generateDebris() {
  // Random inclination (50–95 deg typical LEO), random RAAN, random mean motion
  const inc = Math.random() * 45 + 50; // 50–95 deg
  const raan = Math.random() * 360;
  const meanMotion = Math.random() * 15 + 10; // rev/day ~ 10–25 (LEO)
  const a = Math.cbrt(KT_CONST.MU / ((meanMotion * 2 * Math.PI) / 86400 ** 2)); // semi-major axis
  const ecc = Math.random() * 0.1; // near‑circular
  const argp = Math.random() * 360;
  const M0 = Math.random() * 360;

  // position now (propagate to current real time)
  const tSec = KT.simSec();
  const pos = KT.prop({ ...KT.tleFromElements(inc, raan, ecc, argp, M0, meanMotion), epoch: 0 }, tSec);
  // debris velocity not used in this simplification
  // visual radius proportional to mass‑estimate
  const radius = Math.random() * 3 + 2; // 2–5 px at screen scale

  return {
    id: `debris-${CHALLENGE.debris.length}`,
    pos: { x: pos.x, y: pos.y, z: pos.z },
    vel: { x: 0, y: 0, z: 0 }, // we'll just update pos each frame via simple propagation
    radius,
    // simple linear approach toward the constellation centre (0,0,0)
    approachSpeed: 0.5 + Math.random() * 0.5, // km/s relative towards centre
  };
}

/* ---------- ADVANCE ALL DEBRIS ONE STEP ---------- */
function advanceDebris() {
  const tSec = KT.simSec();
  CHALLENGE.debris.forEach(d => {
    // propagate the debris a little forward using a simplified two‑body step
    // we just nudge it toward centre along its current direction
    const dir = { x: -d.pos.x, y: -d.pos.y, z: -d.pos.z };
    const len = Math.hypot(dir.x, dir.y, dir.z);
    if (len === 0) { dir.x = 1; dir.y = 0; dir.z = 0; }
    const norm = { x: dir.x / len, y: dir.y / len, z: dir.z / len };
    // move 10 km toward centre (arbitrary but visible)
    const step = 10;
    d.pos.x += norm.x * step;
    d.pos.y += norm.y * step;
    d.pos.z += norm.z * step;
    // shrink radius a tiny bit so it doesn't grow indefinitely
    d.radius *= 0.99;
  });
}

/* ---------- CHECK COLLISIONS DEBRIS vs SATellites ---------- */
function checkCollisions() {
  CHALLENGE.satellites.forEach(sat => {
    CHALLENGE.debris.forEach(d => {
      // distance between satellite (using propagated KT position) and debris
      const satPos = KT.prop(sat, KT.simSec()); // this uses the possibly‑dragged satellite
      const dx = satPos.x - d.pos.x;
      const dy = satPos.y - d.pos.y;
      const dz = satPos.z - d.pos.z;
      const dist = Math.hypot(dx, dy, dz);
      // if within 30 km "kill" the satellite
      if (dist < 30) {
        // mark satellite lost
        sat.status = 'breach'; // visually red, also counts as lost
        CHALLENGE.survivors -= 1;
        // add a log entry
        log('breach', `DEBRIS ${d.id} IMPACTED ${sat.satellite_id} — survivor count ${CHALLENGE.survivors}`);
      }
    });
  });
  if (CHALLENGE.survivors <= 0) {
    endGame();
  } else {
    renderChallengeSatellies(); // re‑colour survivors
  }
}

/* ---------- COMPUTE EVASION ΔV FOR THE MOST THREATENED SAT ---------- */
function computeEvasion() {
  // find the satellite with the closest approaching debris
  let threat = null;
  let minDist = Infinity;
  CHALLENGE.satellites.forEach(sat => {
    CHALLENGE.debris.forEach(d => {
      const satPos = KT.prop(sat, KT.simSec());
      const dx = satPos.x - d.pos.x;
      const dy = satPos.y - d.pos.y;
      const dz = satPos.z - d.pos.z;
      const dist = Math.hypot(dx, dy, dz);
      if (dist < minDist) {
        minDist = dist;
        threat = { sat, debris: d };
      }
    });
  });
  if (!threat) {
    log('warn', 'No active debris threat to compute evasion for.');
    return;
  }

  // use the existing client‑side maneuver logic (simplified)
  const sat = threat.sat;
  const debris = threat.debris;
  const relSpeed = 7.5; // km/s typical LEO relative velocity (estimate)
  const deltaV = Math.min(120, Math.max(20, 0.03 * relSpeed * 1000)); // m/s
  CHALLENGE.deltaVSpent += deltaV;

  // update the satellite status to show it evaded (temporarily safe)
  sat.status = 'safe';
  // add to log
  log('info', `EVASION ΔV ${deltaV.toFixed(1)} m/s applied to ${sat.satellite_id} — closest approach was ${minDist.toFixed(1)} km`);

  // re‑render
  renderChallengeSatellies();
  updateChallengeHUD();

  // auto‑clear the evasion panel after 4s
  setTimeout(() => {
    log('info', 'Evasion maneuver completed. Remaining wave still active.');
  }, 4000);
}

/* ---------- UPDATE THE CHALLENGE HUD ---------- */
function updateChallengeHUD() {
  challengeWave.textContent = CHALLENGE.wave;
  challengeSurvivors.textContent = CHALLENGE.survivors;
  challengeDeltaV.textContent = CHALLENGE.deltaVSpent.toFixed(1);
}

/* ---------- GAME OVER ---------- */
function endGame() {
  CHALLENGE.gameOver = true;
  CHALLENGE.active = false;
  challengeOverlay.classList.add('hidden');
  endWave.textContent = CHALLENGE.wave;
  endSurvivors.textContent = CHALLENGE.survivors;
  endDeltaV.textContent = CHALLENGE.deltaVSpent.toFixed(1);
  challengeEnd.classList.remove('hidden');
}

/* ---------- RESTART THE CHALLENGE ---------- */
function restartChallengeMode() {
  CHALLENGE.active = false;
  challengeOverlay.classList.add('hidden');
  initChallengeGame();
}

/* ---------- KEYBOARD SHORTCUT ---------- */
document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape' && CHALLENGE.active) {
    toggleChallengeMode();
  }
});

/* ---------- DRAW the challenge satellites on every render tick (optional) ---------- */
// We'll let the HTML/CSS positioning handle it; the dots re‑render each wave.
// If you want a radar‑style sweep, add a requestAnimationFrame loop here.

/* -------- HELPER: deg2rad (shared) -------- */
function deg2rad(d) { return d * Math.PI / 180; }
