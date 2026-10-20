// UNVERIFIED — USER MUST COMPILE
// Appended to frontend/index.html AFTER app.js.
// Wires the audit chip to the ZKP proof-server + ClimateSentinel contract.

(function () {
  'use strict';

  async function anchorCurrentConjunction() {
    const conjId = (window.CURRENT_CONJ_ID != null) ? window.CURRENT_CONJ_ID : 0;
    const chip = document.querySelector('.audit-chip') || document.querySelector('#audit-chip');
    if (!chip) { console.warn('audit chip not found'); return; }
    const original = chip.textContent;

    try {
      chip.textContent = 'PROVING…';
      chip.style.opacity = '0.6';

      const res = await fetch(`http://localhost:3001/api/proof?conjId=${conjId}`);
      if (!res.ok) throw new Error(`proof server: ${res.status}`);
      const data = await res.json();

      if (!window.Chain || !window.Chain.isReady || !window.Chain.isReady()) {
        chip.textContent = 'NO WALLET';
        chip.style.opacity = '1';
        console.warn('Chain not ready — set window.CLIMATE_SENTINEL_ADDRESS and load ethers CDN');
        return;
      }

      chip.textContent = 'ANCHORING…';
      const { txHash, etherscanUrl } = await window.Chain.anchorProof(
        data.proof.pA, data.proof.pB, data.proof.pC, data.publicSignals
      );

      chip.textContent = `0x${txHash.slice(2, 8)}…${txHash.slice(-4)}`;
      chip.dataset.etherscan = etherscanUrl;
      chip.style.opacity = '1';
      chip.style.cursor = 'pointer';
      chip.title = `View on Etherscan: ${etherscanUrl}`;
      chip.onclick = () => window.open(etherscanUrl, '_blank');
    } catch (err) {
      console.error('Anchor failed:', err);
      chip.textContent = 'ANCHOR FAILED';
      chip.style.opacity = '1';
      chip.title = err.message;
      setTimeout(() => { chip.textContent = original; }, 3000);
    }
  }

  window.anchorCurrentConjunction = anchorCurrentConjunction;

  // Auto-bind: if there's a button with id="anchor-button", wire it.
  // Otherwise, bind to the audit chip itself on click.
  document.addEventListener('DOMContentLoaded', () => {
    const btn = document.getElementById('anchor-button');
    if (btn) btn.addEventListener('click', anchorCurrentConjunction);
  });
})();
