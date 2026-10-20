// UNVERIFIED — USER MUST RUN (browser ES module, no build step)
/* globals ethers */
// Ethers v6 loaded from CDN via importmap in index.html
// Window globals set by index.html:
//   window.CLIMATE_SENTINEL_ADDRESS  — contract address
//   window.GROTH16_VERIFIER_ADDRESS  — verifier address

// ---------- Provider ----------
function getProvider() {
  if (!window.ethereum) {
    throw new Error('No Ethereum provider found. Please install MetaMask.');
  }
  // BrowserProvider is ethers v6
  return new ethers.BrowserProvider(window.ethereum);
}

// ---------- anchorProof ----------
/**
 * Sends a verifyAndAnchor transaction to the ClimateSentinel contract.
 * proof: [pA, pB, pC] each as [low, high] uint256 pairs
 * pubSignals: [commitmentA, commitmentB, proximity, dist_sq_m, R_m]
 * Returns { txHash, etherscanUrl }
 */
async function anchorProof(pA, pB, pC, pubSignals) {
  const provider = getProvider();
  const signer = await provider.getSigner();

  const sentinelAddr = window.CLIMATE_SENTINEL_ADDRESS;
  if (!sentinelAddr) throw new Error('CLIMATE_SENTINEL_ADDRESS not set');

  const climate = new ethers.Contract(sentinelAddr, [
    "function verifyAndAnchor(
      uint256[2] calldata _pA,
      uint256[2][2] calldata _pB,
      uint256[2] calldata _pC,
      uint256[5] calldata _pubSignals
    ) external returns (bool accepted)",
  ], provider);

  // Normalise pA/pB/pC into the flat calldata format the contract expects:
  // pA: two uint256 values (low, high)
  // pB: four uint256 values (simulating uint256[2][2])
  // pC: two uint256 values
  const tx = await climate.verifyAndAnchor(
    pA[0], pA[1],
    pB[0], pB[1], pB[2], pB[3],
    pC[0], pC[1],
    pubSignals
  );

  const receipt = await tx.wait();
  const etherscan = window.ETHERSCAN_API_URL || 'https://sepolia.etherscan.io';
  const etherscanUrl = `${etherscan}/tx/${receipt.hash}`;

  return { txHash: receipt.hash, etherscanUrl };
}

// ---------- getRecentEvents ----------
/**
 * Fetches recent DataAnchored events from the ClimateSentinel contract.
 * Returns array of { proofId, commitmentA, commitmentB, dist_sq_m, R_m, timestamp }
 */
async function getRecentEvents(fromBlock) {
  const sentinelAddr = window.CLIMATE_SENTINEL_ADDRESS;
  if (!sentinelAddr) throw new Error('CLIMATE_SENTINEL_ADDRESS not set');

  const climate = new ethers.Contract(sentinelAddr, [
    "event DataAnchored(
      uint256 indexed proofId,
      uint256 commitmentA,
      uint256 commitmentB,
      uint256 dist_sq_m,
      uint256 R_m,
      uint256 timestamp
    )",
  ], provider);

  const events = await climate.queryFilter('DataAnchored', fromBlock, 'latest');
  return events.map((ev) => ({
    proofId: ev.args.proofId.toNumber(),
    commitmentA: ev.args.commitmentA.toString(),
    commitmentB: ev.args.commitmentB.toString(),
    dist_sq_m: ev.args.dist_sq_m.toString(),
    R_m: ev.args.R_m.toString(),
    timestamp: ev.args.timestamp.toNumber(),
  }));
}

// ---------- Export via window.Chain ----------
if (typeof window !== 'undefined') {
  window.Chain = { anchorProof, getRecentEvents };
}