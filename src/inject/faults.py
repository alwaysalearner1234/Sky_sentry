"""
Synthetic fault injection for satellite LST observations.

Injects various fault types into clean data to create labeled anomalies.
Fault types:
  - offset: constant bias shift
  - drift: linear time trend
  - stripe: spatial striping pattern
  - dropout: random missing values
  - spike: isolated extreme values
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, Any


def inject_offset(data: pd.DataFrame, col: str = "sat_value",
                   offset: float = None, severity: str = "medium") -> pd.DataFrame:
    """Inject a constant offset bias into satellite values.

    Args:
        data: DataFrame with sat_value column
        col: column name to apply offset to
        offset: absolute offset value; if None, computed from severity
        severity: "low", "medium", "high" -> auto-scale offset

    Returns:
        DataFrame with offset injected
    """
    if offset is None:
        if severity == "low":
            offset = 0.5
        elif severity == "medium":
            offset = 2.0
        elif severity == "high":
            offset = 5.0
        else:
            raise ValueError(f"Unknown severity: {severity}")

    data = data.copy()
    data[col] = data[col] + offset
    return data


def inject_drift(data: pd.DataFrame, col: str = "sat_value",
                  drift_per_hour: float = None, severity: str = "medium",
                  time_col: str = "time_utc") -> pd.DataFrame:
    """Inject a linear drift over time into satellite values.

    Args:
        data: DataFrame with sat_value and time_col
        col: column name to apply drift to
        drift_per_hour: drift per hour; if None, computed from severity
        severity: "low", "medium", "high"
        time_col: name of time column (used to compute hour offsets)

    Returns:
        DataFrame with drift injected
    """
    if drift_per_hour is None:
        if severity == "low":
            drift_per_hour = 0.01
        elif severity == "medium":
            drift_per_hour = 0.05
        elif severity == "high":
            drift_per_hour = 0.15
        else:
            raise ValueError(f"Unknown severity: {severity}")

    data = data.copy()
    # Convert time to hours since start
    if time_col in data.columns:
        t = pd.to_datetime(data[time_col])
        hours_from_start = (t - t.min()).dt.total_seconds() / 3600
        data[col] = data[col] + drift_per_hour * hours_from_start.values
    else:
        # No time column, inject drift based on index
        hours = np.arange(len(data))
        data[col] = data[col] + drift_per_hour * hours

    return data


def inject_stripe(data: pd.DataFrame, col: str = "sat_value",
                 stripe_width: int = 10, severity: str = "medium") -> pd.DataFrame:
    """Inject a spatial stripe fault (affects contiguous rows).

    Args:
        data: DataFrame with sat_value
        col: column name to apply stripe to
        stripe_width: number of rows affected per stripe
        severity: "low", "medium", "high"

    Returns:
        DataFrame with stripe injected
    """
    if severity == "low":
        magnitude = 1.0
    elif severity == "medium":
        magnitude = 3.0
    elif severity == "high":
        magnitude = 7.0
    else:
        raise ValueError(f"Unknown severity: {severity}")

    data = data.copy()
    n = len(data)
    # Create stripes every stripe_width rows
    for i in range(0, n, stripe_width):
        end = min(i + stripe_width, n)
        data[col].iloc[i:end] += magnitude

    return data


def inject_dropout(data: pd.DataFrame, col: str = "sat_value",
                   dropout_ratio: float = None, severity: str = "medium") -> pd.DataFrame:
    """Inject random dropout (NaN) into satellite values.

    Args:
        data: DataFrame with sat_value
        col: column name to apply dropout to
        dropout_ratio: fraction of values to drop; if None, uses severity default
        severity: "low", "medium", "high"

    Returns:
        DataFrame with dropouts (NaN) injected
    """
    if dropout_ratio is None:
        if severity == "low":
            dropout_ratio = 0.05
        elif severity == "medium":
            dropout_ratio = 0.15
        elif severity == "high":
            dropout_ratio = 0.30
        else:
            raise ValueError(f"Unknown severity: {severity}")

    data = data.copy()
    n = len(data)
    n_drop = int(n * dropout_ratio)
    drop_indices = np.random.choice(n, n_drop, replace=False)
    data = data.copy()
    data[col].iloc[drop_indices] = np.nan

    return data


def inject_spike(data: pd.DataFrame, col: str = "sat_value",
                 severity: str = "medium") -> pd.DataFrame:
    """Inject random spikes (extreme outliers) into satellite values.

    Args:
        data: DataFrame with sat_value
        col: column name to apply spikes to
        severity: "low", "medium", "high"

    Returns:
        DataFrame with spikes injected
    """
    if severity == "low":
        multiplier = 3.0
    elif severity == "medium":
        multiplier = 5.0
    elif severity == "high":
        multiplier = 10.0
    else:
        raise ValueError(f"Unknown severity: {severity}")

    data = data.copy()
    n = len(data)
    n_spikes = max(1, n // 50)  # ~2% of data
    spike_indices = np.random.choice(n, n_spikes, replace=False)

    # Get the range of normal values for scaling
    sat_min = data[col].min()
    sat_max = data[col].max()
    range_val = sat_max - sat_min if sat_max > sat_min else 1.0

    for idx in spike_indices:
        # Spike above normal range
        data[col].iloc[idx] = max(sat_max, sat_max + multiplier * range_val)

    return data


def inject_fault(data: pd.DataFrame, fault_type: str = "offset",
                 severity: str = "medium", **kwargs) -> pd.DataFrame:
    """Inject a specified fault type into satellite data.

    Args:
        data: Input DataFrame with sat_value column
        fault_type: One of "offset", "drift", "stripe", "dropout", "spike"
        severity: "low", "medium", "high"
        **kwargs: Additional arguments passed to specific fault functions

    Returns:
        DataFrame with fault injected
    """
    fault_generators = {
        "offset": lambda d: inject_offset(d, severity=severity, **kwargs),
        "drift": lambda d: inject_drift(d, severity=severity, **kwargs),
        "stripe": lambda d: inject_stripe(d, severity=severity, **kwargs),
        "dropout": lambda d: inject_dropout(d, severity=severity, **kwargs),
        "spike": lambda d: inject_spike(d, severity=severity, **kwargs),
    }

    if fault_type not in fault_generators:
        raise ValueError(f"Unknown fault type: {fault_type}. "
                         f"Choose from {list(fault_generators.keys())}")

    return fault_generators[fault_type](data)


def create_faulty_dataset(n_obs: int = 1000, n_faults: int = 5,
                          fault_types: list = None, seed: int = 42) -> pd.DataFrame:
    """Create a synthetic dataset with multiple injected faults.

    Args:
        n_obs: Number of observations
        n_faults: Number of different fault types to inject
        fault_types: List of fault types to inject; if None, random choice
        seed: Random seed for reproducibility

    Returns:
        DataFrame with clean baseline + injected faults labeled
    """
    if fault_types is None:
        fault_types = ["offset", "drift", "stripe", "dropout", "spike"]

    np.random.seed(seed)

    # Create baseline LST data (typical CONUS LST values ~ 250-320 K)
    baseline_lst = 280.0 + np.random.randn(n_obs) * 5.0
    baseline_lst = np.clip(baseline_lst, 220, 330)

    # Create time column (hourly over 3 months ~ 2000 hours)
    hours = np.arange(n_obs)
    time_labels = pd.date_range("2024-01-01", periods=n_obs, freq="h")

    data = pd.DataFrame({
        "obs_id": range(n_obs),
        "time_utc": time_labels,
        "sat_value": baseline_lst,
        "ref_value": baseline_lst + np.random.randn(n_obs) * 1.0,  # slightly different
    })

    # Inject faults
    fault_idx = 0
    for ft in fault_types:
        n_fault_obs = n_obs // (len(fault_types) + 1)  # split data among faults
        start = fault_idx * n_fault_obs
        end = min((fault_idx + 1) * n_fault_obs, n_obs)

        # Choose severity based on fault type order
        severity_options = ["low", "medium", "high"]
        severity = severity_options[fault_idx % 3]

        faulted = inject_fault(
            data.iloc[start:end], fault_type=ft, severity=severity
        )
        data.loc[data["obs_id"].between(start, end - 1), "sat_value"] = faulted["sat_value"]
        fault_idx += 1

    # Add fault labels
    data["fault_type"] = "none"
    data["fault_severity"] = "none"

    # Re-assign fault labels after baseline modification
    fault_idx = 0
    for ft in fault_types:
        n_fault_obs = n_obs // (len(fault_types) + 1)
        start = fault_idx * n_fault_obs
        end = min((fault_idx + 1) * n_fault_obs, n_obs)
        data.loc[data["obs_id"].between(start, end - 1), "fault_type"] = ft
        severity_options = ["low", "medium", "high"]
        data.loc[data["obs_id"].between(start, end - 1), "fault_severity"] = severity_options[fault_idx % 3]
        fault_idx += 1

    return data


if __name__ == "__main__":
    # Quick test: create dataset and verify faults are injectable
    data = create_faulty_dataset(n_obs=200, n_faults=3, seed=123)
    print(f"Created dataset with {len(data)} observations")
    print(f"Fault distribution:\n{data['fault_type'].value_counts()}")
    print(f"\nSample offset fault region:")
    offset_data = data[data["fault_type"] == "offset"]
    print(offset_data.head(10))
    print(f"\nSample spike fault region:")
    spike_data = data[data["fault_type"] == "spike"]
    print(spike_data.head(10))