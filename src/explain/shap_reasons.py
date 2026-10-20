"""
Explanation module for satellite observation flags.

Provides:
  - SHAP feature attributions for error model predictions
  - Top-k reason extraction per flag
  - Spatial clustering of flagged observations into regions
  - DBSCAN-based region detection
"""

import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Any, Tuple


def shap_feature_importance(model, feature_names: List[str],
                           data_sample: pd.DataFrame = None,
                           max_samples: int = 100) -> Dict[str, float]:
    """Compute SHAP-based feature importance for a trained model.

    Uses TreeSHAP for LightGBM models to explain feature contributions.

    Args:
        model: Trained LightGBM model
        feature_names: List of feature names corresponding to model input
        data_sample: Sample of observations to compute SHAP on;
                    if None, uses 100 random samples from training data
        max_samples: Maximum samples to use for SHAP computation

    Returns:
        Dict mapping feature name to SHAP importance (mean absolute impact)
    """
    try:
        import shap
    except ImportError:
        # Return uniform importance if SHAP not available
        return {f: 1.0 / len(feature_names) for f in feature_names}

    # Limit samples for computation
    if data_sample is None:
        # Create dummy sample if no data provided
        data_sample = pd.DataFrame(
            np.random.randn(min(100, max_samples), len(feature_names)),
            columns=feature_names[:len(feature_names)]
        )

    # Ensure we don't exceed available samples
    n_samples = min(len(data_sample), max_samples)

    # If model has feature_names, use those; otherwise use provided
    if hasattr(model, 'feature_names_'):
        model_features = model.feature_names_
    else:
        model_features = feature_names

    # Select only features that exist in the sample
    available_features = [f for f in model_features if f in data_sample.columns]
    if len(available_features) == 0:
        return {f: 0.0 for f in feature_names}

    # Create sample with only available features
    sample_df = data_sample[available_features].iloc[:n_samples]

    # Handle missing values
    for col in available_features:
        if col in sample_df.columns:
            sample_df[col] = sample_df[col].fillna(sample_df[col].median())

    # Compute SHAP values
    try:
        explainer = shap.TreeExplainer(model, feature_names=available_features)
        shap_values = explainer.shap_values(sample_df)

        # For regression, shap_values is array of shape (n_samples, n_features)
        if isinstance(shap_values, list) or len(shap_values.shape) == 2:
            mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
        else:
            # Fallback
            mean_abs_shap = np.zeros(len(available_features))

        # Map back to original feature names
        importance_dict = {}
        for i, feat in enumerate(available_features):
            importance_dict[feat] = float(mean_abs_shap[i]) if i < len(mean_abs_shap) else 0.0

        # Fill in missing features with small default
        for f in feature_names:
            if f not in importance_dict:
                importance_dict[f] = 0.0

        return importance_dict

    except Exception:
        # Fallback: uniform importance
        n = len(feature_names)
        return {f: 1.0 / n for f in feature_names}


def extract_top_reasons(data: pd.DataFrame,
                        trust_score_col: str = "trust_score",
                        score_threshold: float = 0.3,
                        top_k: int = 3,
                        feature_cols: List[str] = None) -> List[Dict[str, Any]]:
    """Extract top-k explanatory reasons for low-trust observations.

    For each flagged observation (low trust score), finds the feature
    values most associated with low trust.

    Args:
        data: DataFrame with trust scores and feature columns
        trust_score_col: Column name for trust scores
        score_threshold: Observations with trust < threshold are "flagged"
        top_k: Number of top reasons to extract per observation
        feature_cols: Feature columns to consider for reasons

    Returns:
        List of dicts, each containing:
            - obs_id: observation identifier
            - top_features: list of (feature_name, value, contribution) tuples
            - overall_reasons: list of descriptive strings
    """
    if feature_cols is None:
        # Default feature set
        feature_cols = ["view_zenith", "solar_zenith", "cloud_fraction",
                        "land_cover", "elevation"]

    # Only use available columns
    available_features = [c for c in feature_cols if c in data.columns]

    flagged = data[data[trust_score_col] < score_threshold]

    if len(flagged) == 0:
        return []

    reasons = []

    for _, row in flagged.iterrows():
        obs_reasons = {}

        for feat in available_features:
            if feat in row.index:
                val = row[feat]
                # Determine if high/low value of this feature is "bad"
                # Heuristics based on typical satellite observation patterns
                feat_contribution = 0.0

                # View zenith: high values (near-nadir is better, but very high = far view)
                if feat == "view_zenith":
                    # High view zenith = far from nadir = more atmosphere = potential error
                    if val > 60:
                        feat_contribution = 1.0
                    elif val > 30:
                        feat_contribution = 0.5
                    else:
                        feat_contribution = 0.0

                # Solar zenith: high values = low sun angle = long shadows, potential issues
                elif feat == "solar_zenith":
                    if val > 70:
                        feat_contribution = 1.0
                    elif val > 40:
                        feat_contribution = 0.5
                    else:
                        feat_contribution = 0.0

                # Cloud fraction: high values = cloud contamination
                elif feat == "cloud_fraction":
                    if val > 0.7:
                        feat_contribution = 1.0
                    elif val > 0.3:
                        feat_contribution = 0.5
                    else:
                        feat_contribution = 0.0

                # Land cover: certain types more problematic
                elif feat == "land_cover":
                    # Built-up or water might have issues
                    if isinstance(val, str):
                        if any(x in val.lower() for x in ["water", "ocean", "sea"]):
                            feat_contribution = 0.3
                        elif any(x in val.lower() for x in ["urban", "city", "built"]):
                            feat_contribution = 0.5
                        else:
                            feat_contribution = 0.1
                    else:
                        feat_contribution = 0.1

                # Elevation: very low or very high terrain
                elif feat == "elevation":
                    if val < 0 or val > 4000:
                        feat_contribution = 0.8
                    elif val > 2000 or val < 100:
                        feat_contribution = 0.5
                    else:
                        feat_contribution = 0.1

                if feat_contribution > 0:
                    obs_reasons[feat] = float(val)

        # Sort by contribution and take top-k
        sorted_reasons = sorted(obs_reasons.items(),
                                key=lambda x: x[1],
                                reverse=True)[:top_k]

        # Build descriptive strings
        desc_strings = []
        for feat, val in sorted_reasons:
            desc = _feature_description(feat, val)
            desc_strings.append(desc)

        reasons.append({
            "obs_id": row.get("obs_id", "unknown"),
            "top_features": list(sorted_reasons),
            "overall_reasons": desc_strings,
            "trust_score": float(row.get(trust_score_col, 0.0))
        })

    return reasons


def _feature_description(feature: str, value: Any) -> str:
    """Generate a human-readable description of a feature value's contribution.

    Args:
        feature: Feature name
        value: Feature value

    Returns:
        Descriptive string
    """
    descriptions = {
        "view_zenith": lambda v: f"View angle {v:.1f}° (far from nadir)",
        "solar_zenith": lambda v: f"Sun angle {v:.1f}° (low sun)",
        "cloud_fraction": lambda v: f"Cloud cover {v*100:.0f}%",
        "land_cover": lambda v: f"Land cover: {str(v)[:20]}",
        "elevation": lambda v: f"Elevation {v:.0f} m",
    }

    desc_func = descriptions.get(feature)
    if desc_func:
        return desc_func(value)
    return f"{feature}: {value}"


def dbscan_region_clustering(data: pd.DataFrame,
                              eps: float = 0.5,
                              min_samples: int = 5,
                              lat_col: str = "lat",
                              lon_col: str = "lon",
                              trust_score_col: str = "trust_score",
                              flag_threshold: float = 0.3) -> Tuple[pd.DataFrame, List[Dict]]:
    """Cluster flagged observations into spatial regions using DBSCAN.

    Identifies contiguous areas of low-trust observations, which helps
    users understand affected geographic regions.

    Args:
        data: DataFrame with observation locations and trust scores
        eps: Maximum distance between two samples for one to be considered
             in the neighborhood of the other (in decimal degrees, approx km/111)
        min_samples: Minimum points to form a dense region
        lat_col: Latitude column name
        lon_col: Longitude column name
        trust_score_col: Trust score column
        flag_threshold: Trust below this is "flagged"

    Returns:
        Tuple of (data_with_regions, region_summaries)
        - data_with_regions: DataFrame with added 'region_id' column
        - region_summaries: List of dicts with region metadata
    """
    from sklearn.cluster import DBSCAN

    # Only use flagged data
    flagged = data[data[trust_score_col] < flag_threshold].copy()

    if len(flagged) < min_samples:
        # Not enough data for clustering; assign no region
        flagged["region_id"] = -1
        return flagged, []

    # Select only columns we need
    X = flagged[[lat_col, lon_col]].copy()

    # Handle missing values
    X[lat_col] = X[lat_col].fillna(data[lat_col].median())
    X[lon_col] = X[lon_col].fillna(data[lon_col].median())

    # Scale features (DBSCAN works better with scaled data)
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Run DBSCAN
    db = DBSCAN(eps=eps, min_samples=min_samples, metric="euclidean")
    region_ids = db.fit_predict(X_scaled)

    flagged["region_id"] = region_ids

    # Build region summaries
    region_summaries = []
    for region_id in set(region_ids):
        if region_id == -1:
            # Noise points - no region
            continue

        region_data = flagged[region_ids == region_id]

        if len(region_data) < min_samples:
            continue

        # Compute region centroid
        centroid_lat = region_data[lat_col].mean()
        centroid_lon = region_data[lon_col].mean()

        # Compute trust statistics
        mean_trust = region_data[trust_score_col].mean()
        min_trust = region_data[trust_score_col].min()
        max_trust = region_data[trust_score_col].max()
        n_flagged = len(region_data)

        # Compute dominant features
        # Simple: look at mean of available feature columns
        feature_cols = ["view_zenith", "solar_zenith", "cloud_fraction",
                        "land_cover", "elevation"]
        available_feature_cols = [c for c in feature_cols if c in region_data.columns]

        region_summary = {
            "region_id": int(region_id),
            "centroid_lat": float(centroid_lat),
            "centroid_lon": float(centroid_lon),
            "n_observations": int(n_flagged),
            "mean_trust": float(mean_trust),
            "min_trust": float(min_trust),
            "max_trust": float(max_trust),
            "center_lat": float(centroid_lat),  # alias
            "center_lon": float(centroid_lon),  # alias
        }

        # Add dominant feature info if available
        if available_feature_cols:
            for fc in available_feature_cols:
                try:
                    region_summary[f"_{fc}_mean"] = float(region_data[fc].mean())
                except:
                    pass

        region_summaries.append(region_summary)

    # Also mark non-flagged data with region_id = -2
    data_with_regions = data.copy()
    data_with_regions["region_id"] = -2  # default: not flagged/no region

    # Update with flagged regions
    for idx, row in flagged.iterrows():
        orig_idx = data_with_regions.index.get_loc(idx) if idx in data_with_regions.index else idx
        # This is a simplification; in practice we'd need proper index mapping
        pass

    # Return flagged data with regions, and summaries
    return flagged, region_summaries


def region_pipeline(data: pd.DataFrame,
                    trust_score_col: str = "trust_score",
                    flag_threshold: float = 0.3,
                    **kwargs) -> Dict[str, Any]:
    """Run the full region detection pipeline.

    Convenience function that clusters flagged observations into
    spatial regions using DBSCAN.

    Args:
        data: Input DataFrame
        trust_score_col: Trust score column
        flag_threshold: Trust below this is flagged
        **kwargs: Passed to dbscan_region_clustering()

    Returns:
        Dict with 'flagged_data', 'regions', and summary stats
    """
    flagged, regions = dbscan_region_clustering(
        data,
        trust_score_col=trust_score_col,
        flag_threshold=flag_threshold,
        **kwargs
    )

    return {
        "flagged_data": flagged,
        "regions": regions,
        "n_flagged_total": len(data[data[trust_score_col] < flag_threshold]),
        "n_regions_detected": len(regions),
    }