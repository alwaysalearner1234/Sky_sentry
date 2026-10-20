"""
WeatherTrust - AI-Powered Weather Satellite Data Reliability System.

P2 Module: ML and Validation Engine
"""

from src.inject.faults import create_faulty_dataset, inject_offset, inject_drift, inject_stripe, inject_dropout, inject_spike, inject_fault
from src.detect.rules import range_check, robust_z_score, isolation_forest_detect, combined_detector
from src.confidence.confidence import LightGBMErrorModel, ConformalIntervals, TrustScoreComputer
from src.explain.shap_reasons import shap_feature_importance, extract_top_reasons, dbscan_region_clustering
from src.correct.bias import BiasModel, ValidationGate
from src.evaluation.evaluation import (
    detection_metrics, trust_score_correlation, calibration_check,
    before_after_metrics, generate_evaluation_report
)


def run_full_pipeline(
    n_obs: int = 2000,
    inject_faults: bool = True,
    fault_types: list = None,
    region_of_interest: str = "CONUS",
    **kwargs
) -> dict:
    """Run the complete P2 pipeline from raw data to trust scores.

    End-to-end workflow coordinating all P2 components:
      1. Fault injection (synthetic anomalies)
      2. Three-layer detection (L1/L2/L3)
      3. LightGBM error model training + prediction
      4. Conformal prediction intervals
      5. Trust score computation
      6. SHAP explanations + top-k reasons
      7. DBSCAN region clustering
      8. Validation gate evaluation

    Args:
        n_obs: Number of observations to process
        inject_faults: Whether to inject synthetic faults for evaluation
        fault_types: Which fault types to inject; if None, all types
        region_of_interest: Region name (for logging/metadata)
        **kwargs: Passed to sub-pipelines

    Returns:
        Dict with full pipeline results including:
            - data: Final DataFrame with all computed columns
            - detection: Detection results
            - error_model: Trained LightGBM model
            - intervals: Conformal intervals
            - trust: Trust scores
            - explanations: SHAP and region explanations
            - gate: Validation gate result
            - report: Full evaluation report
    """
    print("=" * 60)
    print(f"WeatherTrust P2 Pipeline -- Region: {region_of_interest}")
    print("=" * 60)

    # Step 1: Data creation / fault injection
    if inject_faults:
        if fault_types is None:
            fault_types = ["offset", "drift", "stripe", "dropout", "spike"]
        if not isinstance(fault_types, list):
            fault_types = [fault_types] if fault_types else ["offset", "drift", "stripe", "dropout", "spike"]
        data = _run_fault_injection(n_obs=n_obs, fault_types=fault_types, seed=kwargs.get("seed", 42))
    else:
        import numpy as np
        import pandas as pd
        np.random.seed(kwargs.get("seed", 42))
        n = n_obs
        hours = np.arange(n)
        baseline_lst = 280.0 + np.random.randn(n) * 5.0
        baseline_lst = np.clip(baseline_lst, 220, 330)
        time_labels = pd.date_range("2024-01-01", periods=n, freq="h")

        data = pd.DataFrame({
            "obs_id": range(n),
            "time_utc": time_labels,
            "sat_value": baseline_lst,
            "ref_value": baseline_lst + np.random.randn(n) * 1.0,
        })

    print(f"\n[P2] Starting with {len(data)} observations\n")

    # Step 2: Detection pipeline (L1 -> L2 -> L3)
    print("[P2] Running detection pipeline (L1 -> L2 -> L3)...")

    # Layer 1: Rule-based range checks
    if kwargs.get("apply_l1", True):
        data = range_check(data, **kwargs)
        l1_rejected = data["l1_reject"].sum()
        print(f"  L1 rejected {l1_rejected} observations")

    # Layer 2: Robust z-score (MAD-based)
    if kwargs.get("apply_l2", True):
        data = robust_z_score(data, **kwargs)
        l2_rejected = data["l2_reject"].sum()
        print(f"  L2 rejected {l2_rejected} observations (outliers via MAD)")

    # Layer 3: Isolation Forest
    if kwargs.get("apply_l3", True):
        # Ensure residual column exists for IF
        if "residual" not in data.columns and "sat_value" in data.columns and "ref_value" in data.columns:
            data["residual"] = data["sat_value"] - data["ref_value"]
        data = isolation_forest_detect(data, **kwargs)
        l3_rejected = data["l3_reject"].sum()
        print(f"  L3 rejected {l3_rejected} observations (via IF)")

    # Combined flag: reject if any layer rejects
    data = combined_detector(data, apply_l1=False, apply_l2=False, apply_l3=False)
    combined_rejected = data["combined_reject"].sum()
    print(f"  Combined layer: {combined_rejected} total rejections")

    # Step 3: LightGBM error model
    print("[P2] Training LightGBM error model...")

    # Engineer features for error prediction
    data_engineered = LightGBMErrorModel.engineer_features(data, **kwargs)

    # Split into train/test
    if "time_utc" in data_engineered.columns:
        data_sorted = data_engineered.sort_values("time_utc").reset_index(drop=True)
    else:
        data_sorted = data_engineered.sample(frac=1, random_state=42).reset_index(drop=True)

    n = len(data_sorted)
    train_size = int(n * 0.8)
    train_data = data_sorted.iloc[:train_size]
    test_data = data_sorted.iloc[train_size:]

    # Train the error model
    error_model = LightGBMErrorModel()
    train_with_features = error_model.engineer_features(train_data, **kwargs)
    error_model.train(train_with_features)

    # Predict on test data
    test_with_pred = error_model.predict(test_data)

    # Predict on full dataset (for dashboard use)
    full_with_pred = error_model.predict(data_engineered)

    n_train = len(train_data)
    n_test = len(test_data)
    print(f"  - Trained on {n_train} observations, predicting on {n_test} test observations")

    # Step 4: Conformal / quantile intervals
    print(f"[P2] Computing {1 - 0.1*100}% conformal prediction intervals...")

    data_with_intervals = ConformalIntervals.quantile_interval(
        data_engineered,
        predicted_error=full_with_pred["predicted_error"],
        alpha=0.1,
        method="residual"
    )

    n_intervals = data_with_intervals.shape[0]
    print(f"  - Computed intervals for {n_intervals} observations")

    # Step 5: Trust score computation
    print("[P2] Computing trust scores...")

    computer = TrustScoreComputer()
    data_with_trust = computer.compute(data_with_intervals, **kwargs)

    n_high = (data_with_trust["trust_score"] >= 0.7).sum()
    n_medium = ((data_with_trust["trust_score"] >= 0.4) & (data_with_trust["trust_score"] < 0.7)).sum()
    n_low = (data_with_trust["trust_score"] < 0.4).sum()

    print(f"  - Trust distribution: High({n_high}) Medium({n_medium}) Low({n_low})")
    print(f"  - Mean trust: {data_with_trust['trust_score'].mean():.3f}")

    # Step 6: Explanations (SHAP + region clustering)
    print("[P2] Generating explanations (SHAP + region clustering)...")

    feature_cols = kwargs.get("feature_cols", ["view_zenith", "solar_zenith", "cloud_fraction",
                                               "land_cover", "elevation", "predicted_error"])

    available_features = [c for c in feature_cols if c in data_with_trust.columns]
    print(f"  - Using {len(available_features)} features for SHAP")

    try:
        shap_importance = shap_feature_importance(
            error_model.model,
            available_features,
            data_sample=data_with_trust.head(min(50, len(data_with_trust))),
            max_samples=50
        )
        top_features = sorted(shap_importance.items(), key=lambda x: x[1], reverse=True)[:3]
        print(f"  - Top SHAP features: {[(f, w) for f, w in top_features]}")
    except Exception as e:
        print(f"  - SHAP computation warning: {e}")
        top_features = [(f, 0.0) for f in available_features[:3]]

    flagged_threshold = kwargs.get("flagged_threshold", 0.3)
    flagged_data = data_with_trust[data_with_trust["trust_score"] < flagged_threshold]
    top_reasons = extract_top_reasons(
        flagged_data,
        trust_score_col="trust_score",
        top_k=3,
        feature_cols=available_features
    )
    print(f"  - Extracted reasons for {len(top_reasons)} flagged observations")

    region_result = dbscan_region_clustering(
        data_with_trust,
        trust_score_col="trust_score",
        flag_threshold=flagged_threshold,
        eps=0.3,
        min_samples=5
    )
    flagged_with_regions = region_result[0]
    region_summaries = region_result[1]
    n_regions = len(region_summaries)
    print(f"  - Detected {n_regions} flagged regions via DBSCAN")

    # Step 7: Validation gate
    print("[P2] Running validation gate...")

    gate_mode = kwargs.get("gate_mode", "flag-only")
    gate = ValidationGate(mode=gate_mode)

    result = _correction_pipeline(
        data_with_trust,
        train_frac=kwargs.get("train_frac", 0.8),
        bias_model_type="linear",
        gate_mode=gate_mode,
    )

    mode = result["mode"]
    gate_result = result["gate_result"]
    corrected_data = result["data"]

    n_flagged = corrected_data["trust_score"].lt(0.5).sum() if "trust_score" in corrected_data.columns else 0

    print(f"  - Gate mode: {mode}")
    print(f"  - Passes: {gate_result['passes']}")
    print(f"  - RMSE improvement: {gate_result['improvement']*100:.1f}% "
          f"(CI: {gate_result['improvement_ci'][0]*100:.1f}% to "
          f"{gate_result['improvement_ci'][1]*100:.1f}%)")

    # Step 8: Generate comprehensive evaluation report
    print("[P2] Generating evaluation report...")

    report = _generate_evaluation_report(
        data_with_trust,
        combined_reject_col="combined_reject",
        trust_score_col="trust_score",
        interval_low_col="interval_low",
        interval_high_col="interval_high",
        ref_col="ref_value",
        error_col="residual_raw",
    )

    # Compile final results
    results = {
        "data": data_with_trust,
        "detection": {
            "combined_reject_col": "combined_reject",
            "l1_rejected": int(data_with_trust.get("l1_reject", pd.Series([False])).sum()),
            "l2_rejected": int(data_with_trust.get("l2_reject", pd.Series([False])).sum()),
            "l3_rejected": int(data_with_trust.get("l3_reject", pd.Series([False])).sum()),
            "combined_rejected": combined_rejected,
        },
        "error_model": error_model,
        "intervals": {
            "interval_low": "interval_low",
            "interval_high": "interval_high",
            "empirical_coverage": report.get("calibration", {}).get("empirical_coverage", None),
        },
        "trust": {
            "trust_score_col": "trust_score",
            "mean_trust": float(data_with_trust["trust_score"].mean()),
            "high_trust_frac": float((data_with_trust["trust_score"] >= 0.7).mean()),
            "low_trust_frac": float((data_with_trust["trust_score"] < 0.4).mean()),
        },
        "explanations": {
            "shap_importance": dict(top_features),
            "top_reasons": top_reasons,
            "flagged_with_regions": flagged_with_regions,
            "region_summaries": region_summaries,
            "n_regions_detected": n_regions,
        },
        "gate": {
            "mode": mode,
            "passes": gate_result["passes"],
            "rmse_improvement_pct": float(gate_result["improvement"] * 100),
        },
        "report": report,
        "pipeline_config": {
            "n_obs": n_obs,
            "fault_types": fault_types,
            "region": region_of_interest,
            "seed": kwargs.get("seed", 42),
        },
    }

    print("\n" + "=" * 60)
    print("P2 Pipeline complete!")
    print("=" * 60)
    print(f"\nKey results:")
    print(f"  - Detection: {combined_rejected} rejections out of {n_obs}")
    print(f"  - Trust mean: {results['trust']['mean_trust']:.3f}")
    print(f"  - Gate passes: {gate_result['passes']}")
    if report.get('before_after'):
        ba = report['before_after']
        print(f"  - RMSE improvement: {ba.get('rmse_reduction', 0)*100:.1f}% "
              f"(significant: {ba.get('improvement_significant', False)})")
    print(f"  - Regions detected: {n_regions}")

    return results


def _run_fault_injection(n_obs: int = 2000, fault_types: list = None, seed: int = 42) -> pd.DataFrame:
    """Run synthetic fault injection to create labeled anomaly data."""
    print(f"[P2] Injecting {len(fault_types or 5)} fault types into {n_obs} observations...")

    if fault_types is None:
        fault_types = ["offset", "drift", "stripe", "dropout", "spike"]

    data = create_faulty_dataset(
        n_obs=n_obs,
        n_faults=len(fault_types),
        fault_types=fault_types,
        seed=seed
    )

    n_faults = len(data[data["fault_type"] != "none"])
    n_clean = len(data[data["fault_type"] == "none"])

    print(f"  - Injected {n_faults} faulty observations, {n_clean} clean")
    print(f"  - Fault types: {data['fault_type'].value_counts().to_dict()}")

    return data


def _correction_pipeline(data, train_frac=0.8, bias_model_type="linear", gate_mode="flag-only", **kwargs):
    """Run the validation gate and bias correction pipeline."""
    from src.correct.gate import correction_pipeline
    return correction_pipeline(
        data,
        train_frac=train_frac,
        bias_model_type=bias_model_type,
        gate_mode=gate_mode,
        **kwargs
    )


def _generate_evaluation_report(data, combined_reject_col="combined_reject", trust_score_col="trust_score",
                                 interval_low_col="interval_low", interval_high_col="interval_high",
                                 ref_col="ref_value", error_col="residual_raw") -> dict:
    """Generate comprehensive evaluation report."""
    from src.evaluation.evaluation import detection_metrics, trust_score_correlation, calibration_check, before_after_metrics, generate_evaluation_report as gen_report

    report = gen_report(
        data,
        fault_type_col=data.get("fault_type", "fault_type") if "fault_type" in data.columns else "fault_type",
        combined_reject_col=combined_reject_col,
        trust_score_col=trust_score_col,
        interval_low_col=interval_low_col,
        interval_high_col=interval_high_col,
        ref_col=ref_col,
        error_col=error_col,
    )

    # Add before/after summary
    ba = before_after_metrics(
        data_original=data,
        data_corrected=data,
        ref_col=ref_col,
        sat_col="sat_value",
    )
    report["before_after"] = ba

    return report


def quick_pipeline_demo(n_obs: int = 500, **kwargs) -> dict:
    """Run a quick demo of the P2 pipeline with fewer observations."""
    print("\n--- Quick P2 Demo ---\n")
    return run_full_pipeline(n_obs=n_obs, inject_faults=True, **kwargs)