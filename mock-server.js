"use strict";

/**
 * SKY SENTRY mock — mirrors the OrbitGuard FastAPI backend contract
 * (github.com/alwaysalearner1234/Sky_sentry) so the frontend can run the
 * full demo with zero dependencies. Port 8000, CORS enabled.
 *
 * GET  /satellites?count=N
 * GET  /scenario/status
 * POST /scenario/dangerous
 * POST /risks/all?uncertainty=15
 * POST /risks/assess
 * POST /maneuver/compute
 * POST /audit/verify
 * GET  /audit/ledger
 * POST /privacy/share
 */

const http = require("http");
const crypto = require("crypto");

const PORT = 8000;

const START = Date.now();
const CORS = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
  "Access-Control-Allow-Headers": "Content-Type",
};

function rand(min, max) {
  return Math.random() * (max - min) + min;
}

const SATELLITES = [
  { id: "SAT-001", organization: "Company-A", altitude_km: 550, inclination: 51.6, velocity_km_s: 7.6, mass_kg: 500, is_active: true },
  { id: "SAT-002", organization: "Company-B", altitude_km: 552, inclination: 51.6, velocity_km_s: 7.59, mass_kg: 450, is_active: true },
  { id: "STARLINK-044", organization: "SpaceX", altitude_km: 550, inclination: 53.0, velocity_km_s: 7.6, mass_kg: 260, is_active: true },
  { id: "SAT-100", organization: "Company-A", altitude_km: 501, inclination: 55.1, velocity_km_s: 7.58, mass_kg: 340, is_active: true },
  { id: "SAT-101", organization: "OneWeb", altitude_km: 540, inclination: 54.8, velocity_km_s: 7.59, mass_kg: 150, is_active: true },
  { id: "SAT-102", organization: "Company-B", altitude_km: 480, inclination: 53.4, velocity_km_s: 7.62, mass_kg: 300, is_active: true },
  { id: "SAT-103", organization: "SpaceX", altitude_km: 505, inclination: 52.0, velocity_km_s: 7.6, mass_kg: 280, is_active: true },
  { id: "SAT-104", organization: "OneWeb", altitude_km: 519, inclination: 51.7, velocity_km_s: 7.61, mass_kg: 145, is_active: true },
  { id: "DEBRIS-001", organization: "Unknown", altitude_km: 545, inclination: 51.5, velocity_km_s: 7.61, mass_kg: 15, is_active: false },
  { id: "DEBRIS-002", organization: "Unknown", altitude_km: 420, inclination: 60.0, velocity_km_s: 7.65, mass_kg: 4, is_active: false },
  { id: "DEBRIS-003", organization: "Unknown", altitude_km: 460, inclination: 48.2, velocity_km_s: 7.63, mass_kg: 11, is_active: false },
  { id: "DEBRIS-004", organization: "Unknown", altitude_km: 500, inclination: 55.0, velocity_km_s: 7.62, mass_kg: 7, is_active: false },
];

let ledgerChain = [];
let ledgerSeed = "0".repeat(64);

function ledgerAdd(eventType, details) {
  const block = {
    event_id: `evt_${Date.now()}_${ledgerChain.length}`,
    timestamp: new Date().toISOString(),
    event_type: eventType,
    previous_hash: ledgerSeed,
    hash: "",
    details,
  };
  const data = `${ledgerSeed}${eventType}${JSON.stringify(details)}${block.timestamp}`;
  block.hash = crypto.createHash("sha256").update(data).digest("hex");
  ledgerSeed = block.hash;
  ledgerChain.push(block);
  return block;
}

ledgerAdd("system_init", { message: "OrbitGuard (mock) initialized" });

function levelFor(dist) {
  if (dist < 50) return "CRITICAL";
  if (dist < 90) return "HIGH";
  if (dist < 200) return "WATCH";
  return "SAFE";
}

function scoreFor(level) {
  if (level === "CRITICAL") return 82;
  if (level === "HIGH") return 68;
  if (level === "WATCH") return 42;
  return 15;
}

const COLOR = { CRITICAL: "\u{1F534}", HIGH: "\u{1F7E0}", WATCH: "\u{1F7E1}", SAFE: "\u{1F7E2}" };

function riskEntry(a, b, dist, vel) {
  const level = levelFor(dist);
  return {
    satellite_a: a.id,
    satellite_b: b.id,
    organization_a: a.organization,
    organization_b: b.organization,
    closest_approach: {
      time_utc: new Date().toISOString(),
      distance_m: Number((dist * 1000).toFixed(0)),
      distance_km: Number(dist.toFixed(2)),
    },
    relative_velocity: {
      m_s: Number(vel.toFixed(2)),
      km_s: Number((vel / 1000).toFixed(2)),
    },
    risk_score: {
      total: scoreFor(level),
      level,
      color: COLOR[level],
    },
    time_to_collision_s: Number((dist / Math.max(vel, 0.1)).toFixed(1)),
    assessment_time: new Date().toISOString(),
  };
}

/** Advance the sector and compute current risk alerts. */
function risksAll() {
  const t = (Date.now() - START) / 1000;
  // SAT-001 vs SAT-002: slow approach into a critical conjunction (20s cycle)
  const pair = Math.max(1, 140 - (t % 20) * 7);
  // debris objects creep inward on a 30s cycle
  const drift = (t % 30) * 8.6;

  const byId = Object.fromEntries(SATELLITES.map((s) => [s.id, s]));
  const SAT = (id) => byId[id];

  const mk = (aId, bId, distFactor) => {
    const a = SAT(aId);
    const b = SAT(bId);
    const d = Math.max(2, 260 - drift * distFactor);
    const vel = (a.velocity_km_s + b.velocity_km_s) * 1000 * 0.02; // relative ~ disturbance
    return riskEntry(a, b, d, vel);
  };

  const alerts = [
    riskEntry(SAT("SAT-001"), SAT("SAT-002"), pair, 7600),
    mk("SAT-100", "DEBRIS-001", 1.0),
    mk("SAT-101", "DEBRIS-002", 1.7),
    mk("SAT-102", "DEBRIS-003", 2.3),
    mk("SAT-103", "DEBRIS-004", 3.0),
  ];

  return {
    total_pairs_assessed: (SATELLITES.length * (SATELLITES.length - 1)) / 2,
    alerts_found: alerts.length,
    alerts,
  };
}

function assessPair(body) {
  const a = body.satellite_a_id;
  const b = body.satellite_b_id;
  const sa = SATELLITES.find((s) => s.id === a);
  const sb = SATELLITES.find((s) => s.id === b);
  if (!sa || !sb) return null;
  const dist = rand(3, 220);
  const vel = rand(20, 9000);
  return riskEntry(sa, sb, dist, vel);
}

function computeManeuver(body) {
  const dv = Number(rand(0.05, 0.4).toFixed(4));
  const result = {
    satellite_id: body.satellite_id,
    threatening_object_id: body.threatening_object_id,
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
  ledgerAdd("maneuver_selected", {
    satellite_id: body.satellite_id,
    threatening_id: body.threatening_object_id,
    delta_v_m_s: dv,
  });
  return result;
}

function auditLedger() {
  return {
    ledger: {
      total_events: ledgerChain.length,
      integrity_valid: true,
      last_event_type: ledgerChain[ledgerChain.length - 1]?.event_type || null,
    },
    events: ledgerChain.map((e) => ({
      id: e.event_id,
      timestamp: e.timestamp,
      event_type: e.event_type,
      hash: e.hash.slice(0, 16),
      previous_hash: e.previous_hash.slice(0, 16),
      details: e.details,
    })),
  };
}

function privacyShare(body) {
  const safe = {
    object_hash: 12345678,
    risk_zone: "ZONE-55",
    tca: 12,
    risk_level: "CRITICAL",
    timestamp: new Date().toISOString(),
  };
  ledgerAdd("privacy_share", { satellite_id: body.satellite_id, compliant: true });
  return { compliant: true, safe_summary: safe, shared_message: safe };
}

function send(res, status, body) {
  res.writeHead(status, {
    "Content-Type": "application/json",
    "Cache-Control": "no-store",
    ...CORS,
  });
  res.end(JSON.stringify(body));
}

function readBody(req) {
  return new Promise((resolve) => {
    let data = "";
    req.on("data", (c) => (data += c));
    req.on("end", () => {
      try {
        resolve(data ? JSON.parse(data) : {});
      } catch {
        resolve({});
      }
    });
  });
}

const server = http.createServer(async (req, res) => {
  if (req.method === "OPTIONS") {
    res.writeHead(204, CORS);
    return res.end();
  }

  const url = new URL(req.url, `http://${req.headers.host || "localhost"}`);
  const path = url.pathname;
  const count = Number(url.searchParams.get("count")) || 12;

  try {
    switch (path) {
      case "/":
        return send(res, 200, {
          message: "OrbitGuard Backend API (mock)",
          version: "1.0.0",
          status: "operational",
          endpoints: ["/satellites", "/risks/all", "/risks/assess", "/maneuver/compute", "/audit/verify", "/audit/ledger", "/privacy/share", "/scenario/status"],
        });

      case "/satellites": {
        const sats = SATELLITES.slice(0, count);
        return send(res, 200, {
          total: sats.length,
          satellites: Object.fromEntries(sats.map((s) => [s.id, s])),
        });
      }

      case "/scenario/status": {
        const alerts = risksAll().alerts;
        const critical = alerts.filter((a) => a.risk_score.level === "CRITICAL");
        return send(res, 200, {
          status: critical.length ? "dangerous" : "normal",
          critical_threats: critical.length,
          total_satellites: SATELLITES.length,
          most_critical: critical[0] || null,
        });
      }

      case "/scenario/dangerous":
        ledgerAdd("scenario_start", { scenario: "SAT-001 approaching SAT-002" });
        return send(res, 200, {
          message: "Dangerous scenario loaded",
          satellites: SATELLITES.length,
          description: "SAT-001 and SAT-002 on collision course",
        });

      case "/risks/all":
        return send(res, 200, risksAll());

      case "/risks/assess": {
        const body = await readBody(req);
        const result = assessPair(body);
        return result
          ? send(res, 200, result)
          : send(res, 404, { detail: "One or both satellites not found" });
      }

      case "/maneuver/compute": {
        const body = await readBody(req);
        return send(res, 200, computeManeuver(body));
      }

      case "/audit/verify":
        return send(res, 200, {
          integrity_valid: true,
          total_blocks: ledgerChain.length,
          message: "Ledger integrity verified",
        });

      case "/audit/ledger":
        return send(res, 200, auditLedger());

      case "/privacy/share": {
        const body = await readBody(req);
        return send(res, 200, privacyShare(body));
      }

      default:
        return send(res, 404, { detail: "not found" });
    }
  } catch (err) {
    return send(res, 500, { detail: String(err && err.message) });
  }
});

server.listen(PORT, () => {
  console.log(`SKY SENTRY mock (OrbitGuard contract) → http://localhost:${PORT}`);
  console.log(`  ${SATELLITES.length} objects · SAT-001 vs SAT-002 approaches every 20s`);
  console.log(`  run the real backend instead: pip install fastapi uvicorn numpy scipy sgp4 && uvicorn main:app --port 8000`);
});