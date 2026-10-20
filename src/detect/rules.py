"""
Rule-based and statistical detectors for anomalous satellite observations.

Provides layered detection:
  - L1: Range checks and consistency rules (hard rejects)
  - L2: Robust z-score using Median Absolute Deviation (MAD)
  - L3: Isolation Forest on engineered features
"""

import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Any, Tuple


def range_check(data: pd.DataFrame, sat_col: str = "sat_value",
                ref_col: str = "ref_value", sat_min: float = None,
                sat_max: float = None, ref_quality_flag_col: str = None,
                quality_flags_to_exclude: List[int] = None) -> pd.DataFrame:
    """L1: Range and quality flag checks.

    Hard rejects observations that:
    - Have sat_value outside physically plausible range
    - Have quality flags indicating known problems
    - Show sat-ref difference exceeding threshold

    Args:
        data: Input DataFrame
        sat_col: Satellite value column name
        ref_col: Reference value column name
        sat_min: Lower bound for sat_value; if None, uses default ~150 K
        sat_max: Upper bound for sat_value; if None, uses default ~350 K
        ref_quality_flag_col: Column name for reference quality flags
        quality_flags_to_exclude: List of quality flag values to reject

    Returns:
        DataFrame with 'l1_reject' column (True = rejected)
    """
    data = data.copy()

    # Set default ranges if not provided
    if sat_min is None:
        sat_min = 150.0  # K typical lower bound for LST
    if sat_max is None:
        sat_max = 350.0  # K typical upper bound for LST

    # Initialize rejection flag
    data["l1_reject"] = False

    # Check 1: Satellite value out of physical range
    out_of_range = (data[sat_col] < sat_min) | (data[sat_col] > sat_max)
    data.loc[out_of_range, "l1_reject"] = True

    # Check 2: Quality flag exclusions
    if ref_quality_flag_col and ref_quality_flag_col in data.columns:
        if quality_flags_to_exclude:
            bad_flags = data[ref_quality_flag_col].isin(quality_flags_to_exclude)
            data.loc[bad_flags, "l1_reject"] = True

    # Check 3: Sat-reference difference too large
    # Typical threshold: > 20 K indicates potential problem
    diff = np.abs(data[sat_col] - data[ref_col])
    large_diff = diff > 20.0
    data.loc[large_diff, "l1_reject"] = True

    return data


def robust_z_score(data: pd.DataFrame, sat_col: str = "sat_value",
                   ref_col: str = "ref_value", mad_constant: float = 0.6745,
                   z_threshold: float = 3.5) -> pd.DataFrame:
    """L2: Robust z-score detector using Median Absolute Deviation.

    Computes residual = sat - ref, then standardizes using MAD.
    Flags observations where |residual| > z_threshold * MAD.

    Args:
        data: Input DataFrame
        sat_col: Satellite value column
        ref_col: Reference value column
        mad_constant: MAD scaling factor (0.6745 for consistency with normal dist)
        z_threshold: z-score threshold for flagging

    Returns:
        DataFrame with 'l2_score' (robust z-score) and 'l2_reject' columns
    """
    data = data.copy()

    # Compute residual
    data["residual"] = data[sat_col] - data[ref_col]

    # Compute robust z-score: 0.6745 * residual / MAD
    residuals = data["residual"]
    median_res = np.median(residuals)
    mad = np.median(np.abs(residuals - median_res))

    # Avoid division by zero
    if mad == 0 or np.isnan(mad):
        data["l2_score"] = 0.0
        data["l2_reject"] = False
        return data

    data["l2_score"] = 0.6745 * (residuals - median_res) / mad
    data["l2_reject"] = np.abs(data["l2_score"]) > z_threshold

    # Also store the raw residual for downstream use
    data["residual_raw"] = residuals

    return data


def isolation_forest_detect(data: pd.DataFrame, feature_cols: List[str] = None,
                           contamination: float = 0.1,
                           random_state: int = 42) -> pd.DataFrame:
    """L3: Isolation Forest anomaly detection.

    Uses scikit-learn's Isolation Forest on engineered features to detect
    anomalous observations that passed L1 and L2 but are still unusual.

    Args:
        data: Input DataFrame
        feature_cols: List of feature column names to use; if None, uses
                     engineered features from available columns
        contamination: Expected proportion of anomalies (default 0.1)
        random_state: Random seed for reproducibility

    Returns:
        DataFrame with 'l3_score' (anomaly score, lower = more anomalous)
                     and 'l3_reject' columns
    """
    try:
        from sklearn.ensemble import IsolationForest
    except ImportError:
        raise ImportError("scikit-learn required for Isolation Forest detection")

    data = data.copy()

    # Default feature set if none specified
    if feature_cols is None:
        feature_cols = ["view_zenith", "solar_zenith", "cloud_fraction",
                        "land_cover", "elevation", "residual"]
        # Remove columns not in data
        feature_cols = [c for c in feature_cols if c in data.columns]

    if len(feature_cols) == 0:
        # Fallback: use residual and view angles if available
        feature_cols = [c for c in ["residual", "view_zenith", "solar_zenith"]
                        if c in data.columns]

    if len(feature_cols) < 2:
        # Not enough features; return neutral results
        data["l3_score"] = 0.5
        data["l3_reject"] = False
        return data

    # Prepare feature matrix
    X = data[feature_cols].copy()

    # Handle missing values - fill with median
    for col in feature_cols:
        if col in X.columns:
            X[col] = X[col].fillna(X[col].median())

    # Handle infinite values
    for col in feature_cols:
        if col in X.columns:
            X[col] = X[col].replace([np.inf, -np.inf], np.nan)
            X[col] = X[col].fillna(X[col].median())

    # Standardize features
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.fillna(X.mean()))

    # Train Isolation Forest
    iforest = IsolationForest(
        contamination=contamination,
        random_state=random_state,
        n_estimators=100
    )
    iforest.fit(X_scaled)

    # Get anomaly scores: decision_function returns lower for more anomalous
    scores = iforest.decision_function(X_scaled)
    # Normalize to 0-1 range (reverse: higher = more anomalous)
    # decision_function can be negative; map to [0, 1]
    min_score, max_score = scores.min(), scores.max()
    if max_score - min_score > 0:
        normalized = (scores - min_score) / (max_score - min_score)
    else:
        normalized = np.zeros_like(scores)

    # We want higher score = more anomalous
    data["l3_score"] = 1.0 - normalized  # invert so higher = more anomalous
    data["l3_reject"] = normalized < (1.0 - contamination)

    # Store used features
    data["_detected_features"] = feature_cols

    return data


def rule_qc_pipeline(data: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """Run the full L1 rule QC pipeline.

    Convenience function that applies range checks and quality flag exclusions.

    Args:
        data: Input DataFrame
        **kwargs: Passed to range_check()

    Returns:
        DataFrame with l1_reject column
    """
    return range_check(data, **kwargs)


def robust_zscore_pipeline(data: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """Run the full L2 robust z-score pipeline.

    Args:
        data: Input DataFrame
        **kwargs: Passed to robust_z_score()

    Returns:
        DataFrame with l2_score and l2_reject columns
    """
    return robust_z_score(data, **kwargs)


def iforest_pipeline(data: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """Run the full L3 Isolation Forest pipeline.

    Args:
        data: Input DataFrame
        **kwargs: Passed to isolation_forest_detect()

    Returns:
        DataFrame with l3_score and l3_reject columns
    """
    return isolation_forest_detect(data, **kwargs)


def combined_detector(data: pd.DataFrame,
                      apply_l1: bool = True,
                      apply_l2: bool = True,
                      apply_l3: bool = True,
                      **kwargs) -> pd.DataFrame:
    """Run the full three-layer detection pipeline.

    The layers are applied sequentially; each subsequent layer only sees
    data that passed previous layers (though this implementation marks
    rejections independently for clarity).

    Args:
        data: Input DataFrame
        apply_l1: Whether to apply L1 range checks
        apply_l2: Whether to apply L2 robust z-score
        apply_l3: Whether to apply L3 Isolation Forest
        **kwargs: Passed to each sub-pipeline

    Returns:
        DataFrame with l1_reject, l2_score/l2_reject, l3_score/l3_reject
    """
    data = data.copy()

    if apply_l1:
        data = range_check(data, **kwargs)

    if apply_l2:
        data = robust_z_score(data, **kwargs)

    if apply_l3:
        # For IF, we need residual from L2; make sure it exists
        if "residual" not in data.columns and "sat_value" in data.columns and "ref_value" in data.columns:
            data["residual"] = data["sat_value"] - data["ref_value"]
        data = isolation_forest_detect(data, **kwargs)

    # Compute combined flag: reject if any layer rejects
    data["combined_reject"] = (
        data.get("l1_reject", False) |
        data.get("l2_reject", False) |
        data.get("l3_reject", False)
    )

    return data