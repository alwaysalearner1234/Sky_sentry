# SKY SENTRY — Orbital Defense Grid (OrbitGuard-linked)

Real-time satellite collision-avoidance dashboard wired to the **OrbitGuard** FastAPI
backend (`github.com/alwaysalearner1234/Sky_sentry`). Plain HTML5 / CSS3 / vanilla ES6+ —
no frameworks, no build step.

## Run against the real OrbitGuard backend

```bash
# 1. clone + install
git clone https://github.com/alwaysalearner1234/Sky_sentry.git
cd Sky_sentry
pip install fastapi uvicorn numpy scipy sgp4

# 2. start the backend (repo root has main.py)
uvicorn main:app --host 0.0.0.0 --port 8000

# 3. serve the frontend
cd frontend
python3 -m http.server 5500
# open http://localhost:5500
```

The frontend polls the real endpoints every second:

| Call | Purpose |
| --- | --- |
| `GET /satellites?count=12` | load the constellation (also loads the dangerous scenario) |
| `POST /risks/all?uncertainty=15` | pair-wise collision risk → drives SAFE/WARNING/BREACH |
| `GET /scenario/status` | global sector status |
| `POST /maneuver/compute` | on breach → optimal evasion package (ΔV, separation, fuel, risk before/after) |
| `POST /audit/verify` | tamper-proof ledger check (footer chip) |

## Demo without the backend (zero deps)

```bash
node mock-server.js   # mirrors the OrbitGuard contract on :8000
```

or flip `USE_MOCK: true` in `app.js` `CONFIG` for a fully offline build-in demo.

### CONFIG (`top of app.js`)

```js
const CONFIG = {
  BACKEND_URL: "http://localhost:8000",
  POLL_INTERVAL_MS: 1000,
  RISK_POLL_INTERVAL_MS: 2500, // /risks/all is SGP4-heavy; throttle it
  AUTH_POLL_INTERVAL_MS: 30000,// ledger verification cadence
  USE_MOCK: false,
  DEMO_MODE: false,
};
```

## The detect → find → evade loop

1. **Detect** — `/risks/all` flags conjunctions (`CRITICAL / HIGH / WATCH / SAFE`).
   Per satellite: `CRITICAL` → BREACH, `HIGH` → WARNING, else SAFE.
2. **Find** — the Constellation panel lists every tracked object (satellite or debris);
   the radar plots each conjunction by distance/bearing with approach trails; click any
   object in the tracker to **lock on** (reticle + trajectory plot + live distance / TCA).
3. **Evade** — on breach the backend computes the optimal ΔV maneuver; the evasion card
   shows `ΔV`, separation-after, fuel cost, and risk before → after, with the full JSON
   package available via `COPY PACKAGE` / `VIEW FULL`. Screen flashes red at 2 Hz and beeps
   (880 Hz square on breach, 440 Hz sine back to safe).

## Notes

- Risk levels are the OrbitGuard `RiskScore` levels (`>=80 CRITICAL`, `>=60 HIGH`, `>=30 WATCH`).
- `SAFETY_BOX_KM` is a front-end display constant (`50 km`) for the radar rings — the real
  backend has no box; thresholding is driven by the backend's own risk levels.
- Malformed JSON → warning logged, tick skipped. Backend down → `⚠ OFFLINE — RETRYING…` banner.
- Mock server emits the same shapes as OrbitGuard, so the demo is byte-compatible.