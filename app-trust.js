setInterval(() => {
    document.getElementById('utc-clock').textContent = new Date().toISOString().slice(11, 19);
}, 1000);

function generateMockData() {
    const data = [];
    for (let i = 0; i < 500; i++) {
        const isFlagged = Math.random() > 0.8;
        data.push({
            obs_id: `T-${i}`,
            lat: 25 + Math.random() * 11,
            lon: -106 + Math.random() * 13,
            trust_score: isFlagged ? Math.random() * 0.4 : 0.8 + Math.random() * 0.2,
            flag: isFlagged,
            top_reasons: isFlagged ? 'Cloud edge' : 'Clear',
            uncertainty_low: 20, uncertainty_high: 30
        });
    }
    return data;
}

let originalData = []; // Store the full dataset

function updateUI(data) {
    let flaggedCount = 0; let totalTrust = 0;
    data.forEach(obs => {
        totalTrust += obs.trust_score;
        if (obs.flag) flaggedCount++;
    });

    document.getElementById('stat-sats').textContent = `TILES: ${data.length}`;
    document.getElementById('stat-objs').textContent = `FLAGGED: ${flaggedCount}`;
    document.getElementById('stat-breach').textContent = `MEAN TRUST: ${(totalTrust / data.length || 0).toFixed(2)}`;
    
    const chip = document.getElementById('chip-global');
    chip.textContent = `GLOBAL TRUST: ${(totalTrust / data.length * 100 || 0).toFixed(1)}%`;
    chip.style.color = flaggedCount > 0 ? '#ef4444' : '#22c55e';

    window.dispatchEvent(new CustomEvent('climate-data-loaded', { detail: data }));
}

// Initialize Data
originalData = generateMockData();
updateUI(originalData);

// Trust Gate Slider Logic
document.getElementById('trust-slider').addEventListener('input', (e) => {
    const threshold = e.target.value / 100;
    document.getElementById('trust-val').textContent = `${e.target.value}%`;
    
    // Filter the data based on the slider
    const filteredData = originalData.filter(obs => obs.trust_score >= threshold);
    updateUI(filteredData);
    
    if (filteredData.length === 0) {
        logEvent("warning", `Trust Gate at ${e.target.value}% - NO DATA PASSES THIS THRESHOLD.`);
    }
});

// Carbon Credit Logic
document.getElementById('carbon-btn').addEventListener('click', () => {
    logEvent("info", "Initiating Carbon Credit Verification...");
    
    const currentData = originalData; 
    const hasAnomalies = currentData.some(obs => obs.flag);
    
    setTimeout(() => {
        const btn = document.getElementById('carbon-btn');
        if (hasAnomalies) {
            logEvent("breach", "❌ CARBON CREDIT REJECTED: Low-trust satellite data detected. Cannot verify emissions.");
            btn.textContent = "❌ REJECTED";
            btn.style.background = "rgba(239, 68, 68, 0.3)";
            btn.style.borderColor = "#ef4444";
            btn.style.color = "#ef4444";
        } else {
            logEvent("safe", "✅ CARBON CREDIT APPROVED: All data verified.");
            btn.textContent = "✅ APPROVED";
        }
    }, 1200);
});

// Click on Earth
window.addEventListener('earth-obs-select', (e) => {
    const obs = e.detail;
    document.getElementById('explain-hint').style.display = 'none';
    document.getElementById('explain-card').style.display = 'block';
    
    document.getElementById('l-id').textContent = obs.obs_id;
    document.getElementById('l-trust').textContent = obs.trust_score.toFixed(3);
    document.getElementById('l-trust').style.color = obs.flag ? '#ef4444' : '#22c55e';
    document.getElementById('l-reason').textContent = obs.top_reasons || 'Unknown';
    document.getElementById('l-unc').textContent = `[${obs.uncertainty_low} to ${obs.uncertainty_high}]`;

    document.getElementById('btn-verify').onclick = () => {
        logEvent("safe", `✅ MERKLE PROOF VERIFIED for ${obs.obs_id}`);
        document.getElementById('audit-chip').className = 'audit-chip audit-ok';
        document.getElementById('audit-chip').textContent = 'TAMPER-PROOF LEDGER: VERIFIED ✓';
    };
});

// Tamper Demo
document.getElementById('tamper-btn').addEventListener('click', () => {
    logEvent("warning", "⚠️ INITIATING CYBERATTACK SIMULATION...");
    const validHash = "0x8f2a9b1c...4e5d";
    const tamperedHash = "0x3c9b7d2a...1a2b";
    
    setTimeout(() => {
        logEvent("breach", "🚨 DATA ALTERED! RECALCULATING HASH...");
        setTimeout(() => {
            logEvent("breach", `❌ MERKLE VERIFICATION FAILED.`);
            logEvent("breach", `   Valid Hash:    ${validHash}`);
            logEvent("breach", `   Tampered Hash: ${tamperedHash}`);
            document.body.style.animation = "flash-breach 0.5s ease-in-out 3";
            document.getElementById('audit-chip').className = 'audit-chip audit-fail';
            document.getElementById('audit-chip').textContent = 'TAMPER-PROOF LEDGER: INVALID ✗';
            setTimeout(() => { document.body.style.animation = ""; }, 1500);
        }, 800);
    }, 1000);
});

function logEvent(level, msg) {
    const log = document.getElementById('event-log');
    const li = document.createElement('li');
    li.className = `log-item log-${level}`;
    li.innerHTML = `<span class="log-time">${new Date().toISOString().slice(11,19)}</span> <span class="log-tag">${level.toUpperCase()}</span> ${msg}`;
    log.prepend(li);
}
