"""
Confidence estimation for satellite observations.

Components:
  - LightGBM error model: predicts expected error from observation features
  - Conformal / quantile intervals: provides uncertainty intervals per observation
  - Trust score: combines anomaly score, predicted error, interval width,
    and reference agreement into a 0-1 trustworthiness score
"""

import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Any, Tuple


class LightGBMErrorModel:
    """LightGBM-based model predicting expected error from observation features.

    Trains on satellite-minus-reference residuals to predict |residual|
    as a function of viewing geometry, cloud cover, surface type, etc.
    """

    def __init__(self,
                 max_depth: int = -1,
                 learning_rate: float = 0.1,
                 n_estimators: int = 100,
                 min_data_in_leaf: int = 20,
                 random_state: int = 42):
        self.params = {
            "objective": "regression",
            "metric": "mae",
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "num_leaves": 31,
            "min_data_in_leaf": min_data_in_leaf,
            "random_state": random_state,
            "verbosity": -1,
        }
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.model = None
        self.feature_names = None
        self.trained = False

    def engineer_features(self, data: pd.DataFrame,
                          sat_col: str = "sat_value",
                          ref_col: str = "ref_value",
                          view_zenith_col: str = "view_zenith",
                          solar_zenith_col: str = "solar_zenith",
                          cloud_fraction_col: str = "cloud_fraction",
                          land_cover_col: str = "land_cover",
                          elevation_col: str = "elevation") -> pd.DataFrame:
        """Engineer features for the error model.

        Creates features that help predict observation error:
        - Absolute residual
        - View/solar geometry angles
        - Cloud fraction
        - Land cover type
        - Elevation

        Also creates interaction features.
        """
        data = data.copy()

        # Basic residual
        data["abs_residual"] = np.abs(data[sat_col] - data[ref_col])

        # View angle features
        if view_zenith_col in data.columns:
            data["view_zenith"] = data[view_zenith_col]
        else:
            # Default: compute approximate view zenith from geometry
            # (placeholder - in practice would need sun-sat geometry)
            data["view_zenith"] = 30.0

        if solar_zenith_col in data.columns:
            data["solar_zenith"] = data[solar_zenith_col]
        else:
            data["solar_zenith"] = 45.0

        # Cloud fraction
        if cloud_fraction_col in data.columns:
            data["cloud_fraction"] = data[cloud_fraction_col]
        else:
            data["cloud_fraction"] = 0.0

        # Land cover (one-hot encode if categorical)
        if land_cover_col in data.columns:
            data["land_cover"] = data[land_cover_col].astype(str).cat.codes \
                if data[land_cover_col].dtype == 'category' else data[land_cover_col]
        else:
            data["land_cover"] = 0

        # Elevation
        if elevation_col in data.columns:
            data["elevation"] = data[elevation_col]
        else:
            data["elevation"] = 0.0

        # Interaction features
        if "view_zenith" in data.columns and "cloud_fraction" in data.columns:
            data["view_cloud_interaction"] = data["view_zenith"] * data["cloud_fraction"]

        if "solar_zenith" in data.columns and "cloud_fraction" in data.columns:
            data["solar_cloud_interaction"] = data["solar_zenith"] * data["cloud_fraction"]

        # Sin/cos encoding of angles (cyclical features)
        for angle_col in ["view_zenith", "solar_zenith"]:
            if angle_col in data.columns:
                data[f"{angle_col}_sin"] = np.sin(np.radians(data[angle_col]))
                data[f"{angle_col}_cos"] = np.cos(np.radians(data[angle_col]))

        return data

    def train(self,
              data: pd.DataFrame,
              feature_cols: List[str] = None,
              target_col: str = "abs_residual") -> "LightGBMErrorModel":
        """Train the LightGBM regression model.

        Args:
            data: Training DataFrame with features and target
            feature_cols: List of feature column names. If None, uses
                         engineered features.
            target_col: Column name for the target (expected error).

        Returns:
            Self, for method chaining
        """
        try:
            import lightgbm as lgb
        except ImportError:
            raise ImportError(
                "LightGBM is required for error model. "
                "Install with: pip install lightgbm"
            )

        if feature_cols is None:
            # Use engineered features
            feature_cols = ["view_zenith", "solar_zenith", "cloud_fraction",
                            "land_cover", "elevation", "view_cloud_interaction",
                            "solar_cloud_interaction", "view_zenith_sin",
                            "view_zenith_cos", "solar_zenith_sin", "solar_zenith_cos"]

        # Only use available columns
        available_features = [c for c in feature_cols if c in data.columns]
        self.feature_names = available_features

        if len(available_features) == 0:
            raise ValueError("No valid feature columns found in training data")

        X = data[available_features].copy()
        y = data[target_col].copy()

        # Handle missing values
        for col in available_features:
            if col in X.columns:
                X[col] = X[col].fillna(X[col].median())

        # Train-test split not needed here since we use time-based split externally
        # But we'll still fit on all provided data for the pipeline

        # LightGBM Dataset
        train_data = lgb.Dataset(X, label=y)

        self.model = lgb.train(
            self.params,
            train_data,
            num_boost_round=self.n_estimators,
            valid_sets=[train_data],
        )

        self.trained = True
        return self

    def predict(self, data: pd.DataFrame,
                feature_cols: List[str] = None) -> pd.DataFrame:
        """Predict expected error for new observations.

        Args:
            data: DataFrame with observation features
            feature_cols: Feature columns; if None, uses trained model's features

        Returns:
            DataFrame with 'predicted_error' column
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call .train() first.")

        if feature_cols is None:
            feature_cols = self.feature_names

        # Only use available columns
        available_features = [c for c in feature_cols if c in data.columns]

        if len(available_features) == 0:
            # Return neutral predictions
            data["predicted_error"] = 2.0  # default default
            return data

        X = data[available_features].copy()

        # Handle missing values
        for col in available_features:
            if col in X.columns:
                X[col] = X[col].fillna(X[col].median())

        # Predict
        predictions = self.model.predict(X)

        data["predicted_error"] = predictions
        return data


class ConformalIntervals:
    """Conformal prediction intervals for satellite observations.

    Provides finite-sample valid uncertainty intervals regardless of
    the underlying distribution, using the MAPIE (Model-Agnostic
    Prediction with Inductive Empowerment) principle.

    Two modes:
      - Quantile regression: Train two models to predict lower/upper quantiles
      - Residual-based: Use residuals from a point predictor to build intervals
    """

    @staticmethod
    def quantile_interval(data: pd.DataFrame,
                          predicted_error: pd.Series,
                          alpha: float = 0.1,
                          method: str = "residual") -> pd.DataFrame:
        """Compute prediction intervals using quantile approach.

        Args:
            data: Input DataFrame
            predicted_error: Series of predicted absolute errors from a model
            alpha: Significance level (e.g., 0.1 for 90% intervals)
            method: "residual" use residuals around predictions,
                   "regression" use quantile regression models

        Returns:
            DataFrame with 'interval_low' and 'interval_high' columns
                         representing prediction intervals
        """
        data = data.copy()

        if method == "residual":
            # Use predicted error + residual-based margin
            residuals = np.abs(data["sat_value"] - data["ref_value"]) if "sat_value" in data.columns and "ref_value" in data.columns else np.zeros(len(data))

            # Available residuals if we have ground truth
            if "residual_raw" in data.columns:
                residuals = data["residual_raw"]
            elif "sat_value" in data.columns and "ref_value" in data.columns:
                residuals = data[sat_col] - data[ref_col] if "sat_value" in data.columns and "ref_value" in data.columns else np.zeros(len(data))

            # Compute margin: high quantile of |residual| / predicted_error
            # Avoid division by zero
            pred_err_safe = predicted_error.replace(0, np.nan)

            if method == "residual" and len(residuals) > 0:
                # Use absolute residuals to compute scale factor
                valid_mask = ~np.isnan(residuals) & ~np.isnan(pred_err_safe) & (pred_err_safe > 0)
                if valid_mask.sum() > 10:
                    scale_factors = np.abs(residuals[valid_mask]) / pred_err_safe[valid_mask]
                    # Use high quantile of scale factors
                    margin_quantile = np.percentile(scale_factors, 100 * (1 - alpha / 2))
                    margin = margin_quantile * predicted_error
                else:
                    # Fallback: use fixed multiplier
                    margin = 1.96 * predicted_error  # ~95% if Gaussian
            else:
                margin = 1.96 * predicted_error

            data["interval_low"] = predicted_error - margin
            data["interval_high"] = predicted_error + margin

        elif method == "regression":
            # Placeholder: would need quantile regression models
            # For now, use residual-based approach
            data = ConformalIntervals.quantile_interval(
                data, predicted_error, alpha=alpha, method="residual"
            )

        # Ensure low < high
        data["interval_low"], data["interval_high"] = \
            np.minimum(data["interval_low"], data["interval_high"]), \
            np.maximum(data["interval_low"], data["interval_high"])

        return data

    @staticmethod
    def coverage_probability(data: pd.DataFrame,
                             sat_col: str = "sat_value",
                             ref_col: str = "ref_value",
                             interval_low_col: str = "interval_low",
                             interval_high_col: str = "interval_high") -> float:
        """Compute empirical coverage probability of prediction intervals.

        Coverage = fraction of observations where ref_value falls
        within [interval_low, interval_high].

        Args:
            data: DataFrame with interval columns and reference values
            sat_col: Satellite value column (not used in coverage calc)
            ref_col: Reference/ground truth value column
            interval_low_col: Name of lower interval bound
            interval_high_col: Name of upper interval bound

        Returns:
            Empirical coverage fraction (should be near 1 - alpha)
        """
        if ref_col not in data.columns or interval_high_col not in data.columns:
            return np.nan

        ref_vals = data[ref_col]
        lows = data[interval_low_closes]
        highs = data[interval_high_col]

        # Coverage: ref is within [low, high]
        within = (ref_vals >= lows) & (ref_vals <= highs)
        coverage = within.mean()

        return float(coverage)


class TrustScoreComputer:
    """Computes per-observation trust scores (0-1).

    Combines multiple signals into a single trustworthiness score:
    - Anomaly score from detectors (L1/L2/L3)
    - Predicted error from error model
    - Uncertainty interval width
    - Reference agreement (sat-ref difference relative to predicted error)
    - Quality flags

    The trust score answers: "How much can we trust this observation?"
    Higher = more trustworthy.
    """

    def __init__(self,
                 anomaly_weight: float = 0.3,
                 error_weight: float = 0.3,
                 interval_weight: float = 0.2,
                 agreement_weight: float = 0.2,
                 # How each component contributes
                 ):
        self.anomaly_weight = anomaly_weight
        self.error_weight = error_weight
        self.interval_weight = interval_weight
        self.agreement_weight = agreement_weight

    def compute(self,
                data: pd.DataFrame,
                anomaly_score_col: str = "combined_reject",
                predicted_error_col: str = "predicted_error",
                interval_low_col: str = "interval_low",
                interval_high_col: str = "interval_high",
                residual_col: str = "residual_raw",
                **kwargs) -> pd.DataFrame:
        """Compute trust score for each observation.

        Components:
          1. Anomaly signal: Higher anomaly -> lower trust
          2. Predicted error: Larger predicted error -> lower trust
             (relative to typical values, normalized)
          3. Interval width: Wider intervals -> lower trust (less certain)
          4. Reference agreement: sat-ref difference within interval -> higher trust

        Each component is normalized to [0, 1] and combined with weights.

        Args:
            data: Input DataFrame with all required columns
            anomaly_score_col: Column with anomaly scores (higher = more anomalous)
            predicted_error_col: Column with predicted absolute errors
            interval_low_col: Column with lower interval bounds
            interval_high_col: Column with upper interval bounds
            residual_col: Column with sat-residual differences

        Returns:
            DataFrame with 'trust_score' column (0-1, higher = more trustworthy)
        """
        data = data.copy()

        # Component 1: Anomaly signal
        # Higher anomaly -> lower trust, so we invert
        if anomaly_score_col in data.columns:
            anomaly = data[anomaly_score_col].astype(float)
            # Normalize anomaly to [0, 1] (if already a score like L3 score)
            anomaly_min, anomaly_max = anomaly.min(), anomaly.max()
            if anomaly_max - anomaly_min > 0:
                anomaly_norm = (anomaly - anomaly_min) / (anomaly_max - anomaly_min)
            else:
                anomaly_norm = np.zeros_like(anomaly)
            # Invert: high anomaly = low trust
            anomaly_trust = 1.0 - anomaly_norm
        else:
            anomaly_trust = 1.0  # No anomaly info = full trust (neutral)
            data[anomaly_score_col] = 0.0

        # Component 2: Predicted error
        # Larger predicted error -> lower trust, normalize
        if predicted_error_col in data.columns:
            pred_err = data[predicted_error_col]
            p_min, p_max = pred_err.min(), pred_err.max()
            if p_max - p_min > 0:
                pred_err_norm = (pred_err - p_min) / (p_max - p_min)
            else:
                pred_err_norm = np.zeros_like(pred_err)
            # Invert: high error = low trust
            error_trust = 1.0 - pred_err_norm
        else:
            error_trust = 1.0
            data[predicted_error_col] = 2.0  # default

        # Component 3: Interval width
        # Wider intervals -> less trust (more uncertainty)
        if interval_low_col in data.columns and interval_high_col in data.columns:
            interval_width = data[interval_high_col] - data[interval_low_col]
            iw_min, iw_max = interval_width.min(), interval_width.max()
            if iw_max - iw_min > 0:
                iw_norm = (interval_width - iw_min) / (iw_max - iw_min)
            else:
                iw_norm = np.zeros_like(interval_width)
            # Invert: wide interval = low trust
            interval_trust = 1.0 - iw_norm
        else:
            interval_trust = 1.0
            data[interval_high_col] = data.get(interval_low_col, 2.0) + 1.0

        # Component 4: Reference agreement
        # How well does sat agree with ref relative to predicted error?
        if residual_col in data.columns and predicted_error_col in data.columns:
            residuals = np.abs(data[residual_col])
            pred_err = data[predicted_error_col].replace(0, np.nan)

            # Agreement: ratio of residual to predicted error
            # Small ratio = good agreement = high trust
            valid = ~np.isnan(residuals) & ~np.isnan(pred_err) & (pred_err > 0)
            agreement_ratios = np.zeros(len(data))
            agreement_ratios[valid] = np.abs(residuals[valid]) / pred_err[valid]

            # Normalize: ratio < 1 means residual < predicted error (good)
            # ratio > 2 means residual > 2x predicted error (poor)
            # Map [0, 2] to [1, 0] trust scale
            arg_min, arg_max = 0.0, 2.0
            if arg_max - arg_min > 0:
                agreement_norm = np.clip((agreement_ratios - arg_min) / (arg_max - arg_min), 0, 1)
            else:
                agreement_norm = np.zeros_like(agreement_ratios)
            # Invert: low ratio = high trust
            agreement_trust = 1.0 - agreement_norm
        else:
            agreement_trust = 1.0
            data[residual_col] = 0.0

        # Combine with weights
        # Default weights: anomaly=0.3, error=0.3, interval=0.2, agreement=0.2
        trust_score = (
            self.anomaly_weight * anomaly_trust +
            self.error_weight * error_trust +
            self.interval_weight * interval_trust +
            self.agreement_weight * agreement_trust
        )

        # Clip to [0, 1]
        trust_score = np.clip(trust_score, 0.0, 1.0)

        data["trust_score"] = trust_score

        # Also compute sub-component scores for explanation
        data["_anomaly_trust"] = anomaly_trust
        data["_error_trust"] = error_trust
        data["_interval_trust"] = interval_trust
        data["_agreement_trust"] = agreement_trust

        return data

    @staticmethod
    def trust_category(trust_score: float) -> str:
        """Categorize trust score into descriptive bands.

        Args:
            trust_score: 0-1 value

        Returns:
            Category label
        """
        if trust_score >= 0.7:
            return "high"
        elif trust_score >= 0.4:
            return "medium"
        else:
            return "low"

    @staticmethod
    def trust_category_color(trust_score: float) -> str:
        """Get a color code for visual display.

        Args:
            trust_score: 0-1 value

        Returns:
            Matplotlib/Plotly color string
        """
        if trust_score >= 0.7:
            return "#2ca02c"  # green
        elif trust_score >= 0.4:
            return "#ff7f0e"  # orange
        else:
            return "#d62728"  # red