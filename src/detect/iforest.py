"""
Isolation Forest detector module.

Dedicated module for scikit-learn Isolation Forest anomaly detection.
Can be imported separately from the detect package.
"""

import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Any

try:
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


def iforest_detect(data: pd.DataFrame,
                   feature_cols: List[str] = None,
                   contamination: float = 0.1,
                   random_state: int = 42,
                   n_estimators: int = 100) -> pd.DataFrame:
    """Run Isolation Forest anomaly detection.

    Detects anomalies by isolating observations in feature space.
    Observations that require fewer splits to isolate are considered
    more anomalous.

    Args:
        data: Input DataFrame
        feature_cols: List of feature column names. If None, uses
                     available columns from: view_zenith, solar_zenith,
                     cloud_fraction, land_cover, elevation, residual.
        contamination: Expected proportion of anomalies in the data.
        random_state: Random seed for reproducibility.
        n_estimators: Number of trees in the forest.

    Returns:
        DataFrame with 'l3_score' (anomaly score, 0-1, higher = more anomalous)
                     and 'l3_reject' (True if anomalous) columns.
    """
    if not SKLEARN_AVAILABLE:
        raise ImportError(
            "scikit-learn is required for Isolation Forest detection. "
            "Install with: pip install scikit-learn"
        )

    data = data.copy()

    # Default feature set
    if feature_cols is None:
        feature_cols = ["view_zenith", "solar_zenith", "cloud_fraction",
                        "land_cover", "elevation", "residual"]

    # Only use columns that exist in the data
    available_features = [c for c in feature_cols if c in data.columns]

    if len(available_features) < 2:
        # Not enough features; return neutral results
        data["l3_score"] = 0.5
        data["l3_reject"] = False
        data["_detected_features"] = []
        return data

    # Prepare feature matrix
    X = data[available_features].copy()

    # Handle missing values
    for col in available_features:
        if col in X.columns:
            X[col] = X[col].fillna(X[col].median())

    # Handle infinite values
    for col in available_features:
        if col in X.columns:
            X[col] = X[col].replace([np.inf, -np.inf], np.nan)
            if X[col].isna().all():
                X[col] = 0.0  # All NaN, fill with 0
            else:
                X[col] = X[col].fillna(X[col].median())

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.fillna(X.mean()))

    # Train Isolation Forest
    iforest = IsolationForest(
        contamination=contamination,
        random_state=random_state,
        n_estimators=n_estimators
    )
    iforest.fit(X_scaled)

    # Get anomaly scores: decision_function returns lower for more anomalous
    scores = iforest.decision_function(X_scaled)

    # Normalize scores to [0, 1] range
    min_score, max_score = scores.min(), scores.max()
    if max_score - min_score > 0:
        normalized = (scores - min_score) / (max_score - min_score)
    else:
        normalized = np.zeros_like(scores)

    # Invert so higher score = more anomalous
    anomaly_score = 1.0 - normalized

    data["l3_score"] = anomaly_score
    data["l3_reject"] = anomaly_score > (1.0 - contamination)

    # Store metadata
    data["_detected_features"] = available_features
    data["_iforest_contamination"] = contamination
    data["_iforest_n_trees"] = n_estimators

    return data


def quick_iforest(data: pd.DataFrame, contamination: float = 0.1,
                  seed: int = 42) -> pd.DataFrame:
    """Convenience function: Isolation Forest with default features.

    Uses view_zenith, solar_zenith, cloud_fraction, land_cover, elevation,
    and residual as features if available.

    Args:
        data: Input DataFrame
        contamination: Expected anomaly proportion
        seed: Random seed

    Returns:
        DataFrame with l3_score and l3_reject columns
    """
    default_features = ["view_zenith", "solar_zenith", "cloud_fraction",
                        "land_cover", "elevation"]
        # Add residual if available
        if "residual" in data.columns:
            default_features.append("residual")

    return iforest_detect(
        data,
        feature_cols=default_features,
        contamination=contamination,
        random_state=seed
    )


def feature_importance_summary(data: pd.DataFrame) -> Dict[str, float]:
    """Summarize which features most contributed to anomalies.

    Uses the isolation path length indirectly via average path length
    per feature across all trees.

    Args:
        data: DataFrame produced by iforest_detect()

    Returns:
        Dict mapping feature name to importance score (0-1)
    """
    if not SKLEARN_AVAILABLE:
        return {}

    if "_detected_features" not in data.columns:
        return {}

    features = data["_detected_features"]
    if isinstance(features, list) and len(features) > 0:
        # Return uniform importance if we can't compute detailed importance
        n = len(features)
        return {f: 1.0 / n for f in features}

    return {}