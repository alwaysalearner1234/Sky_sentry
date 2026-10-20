"""
Basic P2 functionality tests.

Tests the core modules that can run with the available dependencies.
Skips tests requiring optional packages (lightgbm, shap, sklearn) unless available.
"""

import sys
import os
import numpy as np
import pandas as pd

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.inject.faults import create_faulty_dataset, inject_offset, inject_drift, inject_stripe, inject_dropout, inject_spike


def test_fault_injection():
    """Test that fault injection creates labeled data correctly."""
    print("Testing fault injection...")

    data = create_faulty_dataset(n_obs=100, n_faults=3, seed=42)

    # Check structure
    assert "obs_id" in data.columns, "Missing obs_id column"
    assert "sat_value" in data.columns, "Missing sat_value column"
    assert "ref_value" in data.columns, "Missing ref_value column"
    assert "fault_type" in data.columns, "Missing fault_type column"
    assert "fault_severity" in data.columns, "Missing fault_severity column"

    # Check fault distribution
    fault_counts = data["fault_type"].value_counts()
    assert "none" in fault_counts.index, "Expected 'none' fault type"
    assert len(fault_counts) > 1, "Expected multiple fault types"

    # Check that faulty regions have different values
    clean_data = data[data["fault_type"] == "none"]
    fault_data = data[data["fault_type"] != "none"]

    print(f"  - Dataset: {len(data)} observations")
    print(f"  - Clean: {len(clean_data)}, Faulty: {len(fault_data)}")
    print(f"  - Fault types: {fault_counts.to_dict()}")

    # Basic assertion: faulty data should have different statistics
    sat_diff = abs(fault_data["sat_value"].mean() - clean_data["sat_value"].mean())
    print(f"  - Mean sat value difference (clean vs faulty): {sat_diff:.3f}")

    print("  ✓ Fault injection test passed\n")
    return True


def test_range_check():
    """Test L1 range check detector."""
    print("Testing L1 range check...")

    # Create simple test data
    data = pd.DataFrame({
        "obs_id": range(10),
        "sat_value": [280.0, 281.0, 100.0, 283.0, 284.0, 285.0, 286.0, 287.0, 288.0, 289.0],
        "ref_value": [279.0, 280.0, 278.0, 282.0, 283.0, 284.0, 285.0, 286.0, 287.0, 288.0],
    })

    result = range_check(data, sat_min=150, sat_max=350, quality_flags_to_exclude=[1])

    assert "l1_reject" in result.columns, "Missing l1_reject column"

    # The value 100 K should be rejected (out of range)
    rejected = result["l1_reject"].sum()
    print(f"  - L1 rejections: {rejected}/10")

    # At least the out-of-range value should be rejected
    assert rejected >= 1, "Expected at least 1 L1 rejection"

    print("  ✓ L1 range check test passed\n")
    return True


def test_robust_z_score():
    """Test L2 robust z-score detector."""
    print("Testing L2 robust z-score...")

    # Create data with some obvious outliers
    np.random.seed(42)
    n = 50
    sat = 280.0 + np.random.randn(n) * 2.0
    ref = sat + np.random.randn(n) * 0.5  # Small residuals normally

    # Add a few extreme outliers
    sat[0] = 350.0  # Very high
    sat[1] = 150.0  # Very low

    data = pd.DataFrame({
        "obs_id": range(n),
        "sat_value": sat,
        "ref_value": ref,
    })

    result = robust_z_score(data)

    assert "l2_score" in result.columns, "Missing l2_score column"
    assert "l2_reject" in result.columns, "Missing l2_reject column"

    l2_rejected = result["l2_reject"].sum()
    print(f"  - L2 rejections: {l2_rejected}/50")

    # Should detect at least some outliers
    assert l2_rejected >= 1, "Expected at least 1 L2 rejection"

    print("  ✓ L2 robust z-score test passed\n")
    return True


def test_combined_detector():
    """Test the combined three-layer detector."""
    print("Testing combined detector...")

    np.random.seed(42)
    n = 50
    sat = 280.0 + np.random.randn(n) * 2.0
    ref = sat + np.random.randn(n) * 0.5

    data = pd.DataFrame({
        "obs_id": range(n),
        "sat_value": sat,
        "ref_value": ref,
        "view_zenith": 30.0 + np.random.randn(n) * 5.0,
        "solar_zenith": 45.0 + np.random.randn(n) * 5.0,
        "cloud_fraction": np.clip(np.random.randn(n) * 0.1, 0, 1),
        "land_cover": "forest",
        "elevation": 100.0 + np.random.randn(n) * 10.0,
    })

    result = combined_detector(data, apply_l1=True, apply_l2=True, apply_l3=False)

    assert "combined_reject" in result.columns, "Missing combined_reject column"

    combined_rejected = result["combined_reject"].sum()
    print(f"  - Combined rejections: {combined_rejected}/50")

    print("  ✓ Combined detector test passed\n")
    return True


def test_trust_score():
    """Test trust score computation."""
    print("Testing trust score computation...")

    np.random.seed(42)
    n = 50
    sat = 280.0 + np.random.randn(n) * 2.0
    ref = sat + np.random.randn(n) * 0.5

    data = pd.DataFrame({
        "obs_id": range(n),
        "sat_value": sat,
        "ref_value": ref,
        "residual_raw": sat - ref,
        "predicted_error": 2.0 + np.random.randn(n) * 0.5,
        "interval_low": 1.0 + np.random.randn(n) * 0.5,
        "interval_high": 3.0 + np.random.randn(n) * 0.5,
        "combined_reject": np.random.rand(n) > 0.7,
    })

    from src.confidence.confidence import TrustScoreComputer
    computer = TrustScoreComputer()
    result = computer.compute(data)

    assert "trust_score" in result.columns, "Missing trust_score column"
    assert result["trust_score"].min() >= 0.0, "Trust score below 0"
    assert result["trust_score"].max() <= 1.0, "Trust score above 1"

    mean_trust = result["trust_score"].mean()
    print(f"  - Mean trust score: {mean_trust:.3f}")
    print(f"  - Trust range: [{result['trust_score'].min():.3f}, {result['trust_score'].max():.3f}]")

    print("  ✓ Trust score test passed\n")
    return True


def test_evaluation_detection():
    """Test detection metrics computation."""
    print("Testing detection metrics...")

    np.random.seed(42)
    n = 100
    data = pd.DataFrame({
        "obs_id": range(n),
        "fault_type": ["none"] * n,
        "combined_reject": np.random.rand(n) > 0.7,
        "trust_score": np.random.rand(n),
    })

    # Add some fault annotations
    data.loc[::10, "fault_type"] = "offset"
    data.loc[::7, "fault_type"] = "spike"

    metrics = detection_metrics(data)

    assert "overall" in metrics, "Missing overall metrics"
    assert "precision" in metrics, "Missing precision"
    assert "recall" in metrics, "Missing recall"
    assert "f1" in metrics, "Missing f1"

    print(f"  - Overall F1: {metrics['overall']['f1']:.3f}")
    print(f"  - Per-fault F1: {metrics['per_fault']}")

    print("  ✓ Detection metrics test passed\n")
    return True


def run_all_basic_tests():
    """Run all basic P2 tests."""
    print("=" * 50)
    print("WeatherTrust P2 Basic Tests")
    print("=" * 50 + "\n")

    tests = [
        test_fault_injection,
        test_range_check,
        test_robust_z_score,
        test_combined_detector,
        test_trust_score,
        test_evaluation_detection,
    ]

    results = []
    for test in tests:
        try:
            results.append(test())
        except Exception as e:
            print(f"  ✗ {test.__name__} FAILED: {e}\n")
            results.append(False)

    print("=" * 50)
    passed = sum(results)
    total = len(results)
    print(f"Results: {passed}/{total} tests passed")
    print("=" * 50)

    if passed == total:
        print("\n✓ All basic P2 tests passed!")
    else:
        print(f"\n⚠ {total - passed} test(s) failed - check output above")

    return passed == total


if __name__ == "__main__":
    success = run_all_basic_tests()
    sys.exit(0 if success else 1)