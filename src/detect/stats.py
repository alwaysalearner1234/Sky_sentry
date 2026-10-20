"""
Additional statistical detectors for satellite observation anomalies.

Includes percentile-based detectors, seasonal adjusted residuals,
and spatial coherence checks.
"""

import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Any


def percentile_detector(data: pd.DataFrame, sat_col: str = "sat_value",
                        ref_col: str = "ref_value", percentile: float = 95.0) -> pd.DataFrame:
    """Detect observations based on extreme residual percentiles.

    Flags observations where the absolute residual exceeds a given
    percentile threshold of the residual distribution.

    Args:
        data: Input DataFrame
        sat_col: Satellite value column
        ref_col: Reference value column
        percentile: Percentile threshold (e.g., 95 means top 5% of residuals)

    Returns:
        DataFrame with 'perc_score' (percentile rank of |residual|)
                     and 'perc_reject' columns
    """
    data = data.copy()

    residuals = np.abs(data[sat_col] - data[ref_col])

    if residuals.std() == 0 or np.isnan(residuals.std()):
        data["perc_score"] = 0.0
        data["perc_reject"] = False
        return data

    threshold = np.percentile(residuals, percentile)
    data["perc_score"] = residuals / threshold  # normalize by threshold
    data["perc_reject"] = residuals > threshold

    return data


def modified_z_score(data: pd.DataFrame, sat_col: str = "sat_value",
                    ref_col: str = "ref_value") -> pd.DataFrame:
    """Compute Modified z-score for each observation.

    Modified z-score = 0.6745 * (x - median) / MAD
    More robust than standard z-score for heavy-tailed distributions.

    Args:
        data: Input DataFrame
        sat_col: Satellite value column
        ref_col: Reference value column

    Returns:
        DataFrame with 'modz_score' column
    """
    data = data.copy()

    residuals = data[sat_col] - data[ref_col]
    median_res = np.median(residuals)
    mad = np.median(np.abs(residuals - median_res))

    if mad == 0 or np.isnan(mad):
        data["modz_score"] = 0.0
        return data

    data["modz_score"] = 0.6745 * (residuals - median_res) / mad

    return data


def spatial_coherence(data: pd.DataFrame, sat_col: str = "sat_value",
                      lat_col: str = "lat", lon_col: str = "lon",
                      window_km: float = 50.0) -> pd.DataFrame:
    """Check spatial coherence: flag observations whose value differs
    significantly from neighbors within a moving window.

    Note: This is a simplified implementation that uses a grid-based
    approximation rather than true geographic neighbors.

    Args:
        data: Input DataFrame
        sat_col: Satellite value column
        lat_col: Latitude column
        lon_col: Longitude column
        window_km: Spatial window size in km

    Returns:
        DataFrame with 'spatial_score' and 'spatial_reject' columns
    """
    data = data.copy()

    # Simple grid-based approximation
    # Bucket observations into spatial bins
    lat_bins = np.floor(data[lat_col] / window_km * 10).astype(int)
    lon_bins = np.floor(data[lon_col] / window_km * 10).astype(int)
    data["_grid_id"] = lat_bins * 1000 + lon_bins

    # Compute mean sat_value per grid cell
    grid_means = data.groupby("_grid_id")[sat_col].mean()
    grid_counts = data.groupby("_grid_id")[sat_col].count()

    # Map back to each observation
    data["_grid_mean"] = data["_grid_id"].map(grid_means)
    data["_grid_count"] = data["_grid_id"].map(grid_counts)

    # Compute deviation from grid mean
    data["_grid_dev"] = np.abs(data[sat_col] - data["_grid_mean"])

    # Flag if deviation exceeds 2*MAD within grid cell
    # Only for grid cells with enough observations
    data["spatial_score"] = 0.0
    data["spatial_reject"] = False

    for grid_id in data["_grid_id"].unique():
        n_in_grid = grid_counts.get(grid_id, 0)
        if n_in_grid < 3:
            continue
        grid_data = data[data["_grid_id"] == grid_id]
        residuals = grid_data[sat_col] - grid_data["_grid_mean"]
        mad = np.median(np.abs(residuals - np.median(residuals)))

        if mad > 0:
            scores = 0.6745 * np.abs(residuals) / mad
            data.loc[grid_data.index, "spatial_score"] = scores
            data.loc[grid_data.index, "spatial_reject"] = scores > 3.5

    # Clean up temporary columns
    data.drop(["_grid_id", "_grid_mean", "_grid_count", "_grid_dev"], axis=1, inplace=True, errors="ignore")

    return data


def seasonal_residual(data: pd.DataFrame, sat_col: str = "sat_value",
                      ref_col: str = "ref_value", month_col: str = "month") -> pd.DataFrame:
    """Compute seasonally-adjusted residuals.

    Removes the mean residual per month so that seasonal patterns
    don't trigger false anomalies.

    Args:
        data: Input DataFrame
        sat_col: Satellite value column
        ref_col: Reference value column
        month_col: Column containing month (1-12) for each observation

    Returns:
        DataFrame with 'seasonal_residual' and 'seasonal_score' columns
    """
    data = data.copy()

    if month_col not in data.columns:
        # Extract month from time_utc if available
        if "time_utc" in data.columns:
            data[month_col] = pd.to_datetime(data["time_utc"]).dt.month
        else:
            # No month info; just compute raw residuals
            data["seasonal_residual"] = data[sat_col] - data[ref_col]
            data["seasonal_score"] = 0.0
            return data

    # Compute residual
    data["raw_residual"] = data[sat_col] - data[ref_col]

    # Compute mean residual per month
    monthly_mean_residual = data.groupby(month_col)["raw_residual"].transform("mean")

    # Subtract monthly mean to get seasonally-adjusted residual
    data["seasonal_residual"] = data["raw_residual"] - monthly_mean_residual

    # Compute robust z-score on seasonal residuals
    residuals = data["seasonal_residual"]
    median_res = np.median(residuals)
    mad = np.median(np.abs(residuals - median_res))

    if mad == 0 or np.isnan(mad):
        data["seasonal_score"] = 0.0
        data["seasonal_reject"] = False
        return data

    data["seasonal_score"] = 0.6745 * (residuals - median_res) / mad
    data["seasonal_reject"] = np.abs(data["seasonal_score"]) > 3.5

    return data