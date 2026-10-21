# 🌍 WeatherTrust: The Verifiable Trust Layer for Climate Data

**IEEE ClimateChain Global Hackathon 2026 Submission**

> *"Don't just share climate data. Prove how far to trust it."*

## 📌 The Problem
Climate decisions—from carbon markets to early warning systems—depend on satellite data. However, these readings can be corrupted by sensor drift, cloud contamination, and cyberattacks. Currently, users have no way to verify if the data they are using is authentic, and centralized databases can be silently altered.

## 🚀 The Solution
**WeatherTrust** is an AI-powered, blockchain-anchored system that determines exactly which satellite observations are reliable. 
1. **AI Reliability Engine:** Evaluates satellite data and assigns a Trust Score (0-1) with SHAP-based explanations for anomalies.
2. **ZKP & Blockchain Ledger:** Uses Zero-Knowledge Proofs and a custom Solidity Smart Contract (`TrustRegistry.sol`) with Merkle Proofs to anchor these trust scores on-chain.
3. **3D WebGL Dashboard:** A cinematic, 12K-quality Three.js digital twin of Earth where users can visualize climate anomalies, filter bad data, and test cryptographic tamper-proofing in real-time.

## ✨ Key Features (What to try in the Demo)
*   **3D Earth & Real Satellites:** Track real orbital paths (GOES-16, NOAA-20) rendered via SGP4 physics.
*   **AI Explainability:** Click any red anomaly dot to see exactly *why* the AI flagged it (e.g., "Cloud edge").
*   **Trust Gate Slider:** Drag the slider to instantly filter out low-trust data on the globe.
*   **Carbon Credit Verification:** Click "Verify Carbon" to simulate a carbon market rejecting credits due to bad data.
*   **Tamper-Proof Ledger:** Click "TAMPER" to simulate a cyberattack. Watch the Merkle hashes mismatch and the cryptographic proof fail instantly.
*   **God's Eye View:** A cinematic, automated camera sweep of the orbital environment.

## 🛠️ Technology Stack
*   **Frontend:** Vanilla JS, Three.js (WebGL), CSS3, HTML5
*   **AI Backend (Mocked in Demo):** FastAPI, Python, LightGBM (simulated in `app-trust.js`)
*   **Cryptography:** Circom (Groth16 ZK-SNARKs), Merkle Trees
*   **Blockchain:** Solidity, OpenZeppelin, Hardhat

## 💻 How to Run Locally

The frontend is fully self-contained. The mock AI data generator is built into the JavaScript, so **no backend or API keys are required** to run the demo.

1. **Clone the repo:**
   ```bash
   git clone https://github.com/alwaysalearner1234/Sky_sentry.git
   cd Sky_sentry/frontend
