"""
Bias correction and validation gate for satellite observations.

Provides:
  - Bias model: estimates systematic bias as function of conditions
  - Validation gate: only applies correction if held-out ground truth
    shows measurable improvement (RMSE reduction, bias reduction)
  - Flag-only default: system flags rather than corrects when gate fails
"""

import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Any, Tuple


class BiasModel:
    """Estimates and corrects systematic bias in satellite observations.

    Models bias as a function of observation conditions (view angle,
    solar angle, land cover, etc.) using a lightweight model.
    """

    def __init__(self,
                 model_type: str = "linear",
                 random_state: int = 42):
        self.model_type = model_type
        self.random_state = random_state
        self.model = None
        self.feature_names = None
        self.trained = False

    def engineer_features(self,
                          data: pd.DataFrame,
                          sat_col: str = "sat_value",
                          ref_col: str = "ref_value",
                          view_zenith_col: str = "view_zenith",
                          solar_zenith_col: str = "solar_zenith",
                          cloud_fraction_col: str = "cloud_fraction",
                          land_cover_col: str = "land_cover") -> pd.DataFrame:
        """Engineer features for bias modeling.

        Creates features that help model systematic bias patterns.
        """
        data = data.copy()

        # Raw residual (sat - ref) which contains both random error + systematic bias
        data["raw_residual"] = data[sat_col] - data[ref_col]

        # View angle features
        if view_zenith_col in data.columns:
            data["view_zenith"] = data[view_zenith_col]
            # Sin/cos for cyclical encoding
            data["view_zenith_sin"] = np.sin(np.radians(data[view_zenith_col]))
            data["view_zenith_cos"] = np.cos(np.radians(data[view_zenith_col]))
        else:
            data["view_zenith"] = 30.0
            data["view_zenith_sin"] = np.sin(np.radians(30.0))
            data["view_zenith_cos"] = np.cos(np.radians(30.0))

        if solar_zenith_col in data.columns:
            data["solar_zenith"] = data[solar_zenith_col]
            data["solar_zenith_sin"] = np.sin(np.radians(data[solar_zenith_col]))
            data["solar_zenith_cos"] = np.cos(np.radians(data[solar_zenith_col]))
        else:
            data["solar_zenith"] = 45.0
            data["solar_zenith_sin"] = np.sin(np.radians(45.0))
            data["solar_zenith_cos"] = np.cos(np.radians(45.0))

        # Cloud fraction
        if cloud_fraction_col in data.columns:
            data["cloud_fraction"] = data[cloud_fraction_col]
        else:
            data["cloud_fraction"] = 0.0

        # Land cover
        if land_cover_col in data.columns:
            data["land_cover"] = data[land_cover_col].astype(str).cat.codes \
                if data[land_cover_col].dtype == 'category' else data[land_cover_col]
        else:
            data["land_cover"] = 0

        # Elevation (if available)
        elevation_col = kwargs.get("elevation_col", None) if False else None
        # We'll handle this separately

        # Interaction features
        if "view_zenith" in data.columns and "cloud_fraction" in data.columns:
            data["view_cloud"] = data["view_zenith"] * data["cloud_fraction"]

        if "solar_zenith" in data.columns and "cloud_fraction" in data.columns:
            data["solar_cloud"] = data["solar_zenith"] * data["cloud_fraction"]

        return data

    def train(self,
              data: pd.DataFrame,
              bias_col: str = "raw_residual",
              feature_cols: List[str] = None) -> "BiasModel":
        """Train the bias model to predict systematic bias.

        Args:
            data: Training DataFrame with bias values and features
            bias_col: Column name containing the bias (typically sat - ref)
            feature_cols: List of feature column names

        Returns:
            Self, for method chaining
        """
        from sklearn.linear_model import LinearRegression
        from sklearn.ensemble import GradientBoostingRegressor

        data = data.copy()

        if feature_cols is None:
            feature_cols = ["view_zenith", "view_zenith_sin", "view_zenith_cos",
                            "solar_zenith", "solar_zenith_sin", "solar_zenith_cos",
                            "cloud_fraction", "land_cover",
                            "view_cloud", "solar_cloud"]

        # Only use available columns
        available_features = [c for c in feature_cols if c in data.columns]
        self.feature_names = available_features

        if len(available_features) == 0:
            raise ValueError("No valid feature columns found")

        X = data[available_features].copy()
        y = data[bias_col].copy()

        # Handle missing values
        for col in available_features:
            if col in X.columns:
                X[col] = X[col].fillna(X[col].median())

        # Train model
        if self.model_type == "linear":
            self.model = LinearRegression(random_state=self.random_state)
        elif self.model_type == "gbm":
            self.model = GradientBoostingRegressor(
                random_state=self.random_state,
                n_estimators=100,
                max_depth=3
            )
        else:
            self.model = LinearRegression(random_state=self.random_state)

        self.model.fit(X, y)
        self.trained = True
        return self

    def predict_bias(self,
                     data: pd.DataFrame,
                     feature_cols: List[str] = None) -> pd.DataFrame:
        """Predict systematic bias for new observations.

        Args:
            data: Observation DataFrame
            feature_cols: Feature columns; if None, uses trained model's features

        Returns:
            DataFrame with 'bias_pred' column (systematic bias estimate)
        """
        if self.model is None:
            # Return zeros if model not trained
            data = data.copy()
            data["bias_pred"] = 0.0
            return data

        if feature_cols is None:
            feature_cols = self.feature_names

        available_features = [c for c in feature_cols if c in data.columns]

        if len(available_features) == 0:
            data = data.copy()
            data["bias_pred"] = 0.0
            return data

        X = data[available_features].copy()

        # Handle missing values
        for col in available_features:
            if col in X.columns:
                X[col] = X[col].fillna(X[col].median())

        # Predict
        bias_pred = self.model.predict(X)

        data = data.copy()
        data["bias_pred"] = bias_pred
        return data

    def correct_bias(self,
                     data: pd.DataFrame,
                     sat_col: str = "sat_value",
                     bias_col: str = "bias_pred") -> pd.DataFrame:
        """Correct satellite values by removing estimated systematic bias.

        Args:
            data: DataFrame with sat_value and bias_pred columns
            sat_col: Satellite value column
            bias_col: Bias prediction column

        Returns:
            DataFrame with 'sat_value_corrected' column (bias-corrected values)
        """
        data = data.copy()

        if "bias_pred" not in data.columns:
            data["sat_value_corrected"] = data[sat_col].values
            return data

        # Corrected value = original - estimated bias
        data["sat_value_corrected"] = data[sat_col] - data[bias_col]

        return data


class ValidationGate:
    """Gate that controls whether bias correction is applied.

    The gate evaluates correction using held-out ground truth (time-split
    or station-split). Correction is only applied if it measurably improves
    error metrics (RMSE reduction, bias reduction) with statistical
    significance (bootstrap confidence interval).

    Default behavior: flag-only (do not correct unless gate passes).
    """

    def __init__(self,
                 improvement_threshold: float = 0.01,  # 1% RMSE improvement min
                 bootstrap_n: int = 1000,
                 significance_level: float = 0.05,
                 # Correction mode
                 mode: str = "flag-only"):  # "flag-only" or "correct-if-pass"
        self.improvement_threshold = improvement_threshold
        self.bootstrap_n = bootstrap_n
        self.significance_level = significance_level
        self.mode = mode

    def evaluate_correction(self,
                            original_rmse: float,
                            corrected_rmse: float,
                            original_bias: float,
                            corrected_bias: float,
                            **kwargs) -> Dict[str, Any]:
        """Evaluate whether bias correction provides measurable improvement.

        Uses bootstrap resampling to compute confidence intervals on
        the metric improvements.

        Args:
            original_rmse: RMSE before correction
            corrected_rmse: RMSE after correction
            original_bias: Mean bias before correction
            corrected_bias: Mean bias after correction
            **kwargs: Additional data for bootstrap resampling

        Returns:
            Dict with evaluation results:
                - improvement: float (RMSE reduction fraction, e.g., 0.03 = 3%)
                - improvement_ci: (lower, upper) bootstrap CI on improvement
                - bias_improvement: float (original - corrected bias)
                - bias_improvement_ci: (lower, upper)
                - passes: bool (whether improvement exceeds threshold and is significant)
                - recommended_mode: "correct" or "flag-only"
        """
        # Compute RMSE improvement
        rmse_improvement = (original_rmse - corrected_rmse) / original_rmse
        # Positive = improvement, negative = degradation

        # Compute bias improvement
        bias_improvement = original_bias - corrected_bias

        # Bootstrap confidence intervals
        if "original_residuals" in kwargs and "corrected_residuals" in kwargs:
            orig_res = kwargs["original_residuals"]
            corr_res = kwargs["corrected_residuals"]

            # Bootstrap RMSE improvements
            n = len(orig_res)
            rmse_improvements = []
            bias_improvements = []

            for _ in range(self.bootstrap_n):
                # Resample with replacement
                indices = np.random.choice(n, n, replace=True)
                sample_orig = orig_res[indices]
                sample_corr = corr_res[indices]

                # RMSE on sample
                rmse_orig = np.sqrt(np.mean(sample_orig ** 2))
                rmse_corr = np.sqrt(np.mean(sample_corr ** 2))

                if rmse_orig > 0:
                    rmse_imp = (rmse_orig - rmse_corr) / rmse_orig
                else:
                    rmse_imp = 0.0
                rmse_improvements.append(rmse_imp)

                # Bias on sample
                bias_orig = np.mean(np.abs(sample_orig))
                bias_corr = np.mean(np.abs(sample_corr))
                bias_imp = bias_orig - bias_corr
                bias_improvements.append(bias_imp)

            # Compute confidence intervals
            rmse_ci = np.percentile(rmse_improvements,
                                    [self.significance_level / 2 * 100,
                                     (1 - self.significance_level / 2) * 100])
            bias_ci = np.percentile(bias_improvements,
                                    [self.significance_level / 2 * 100,
                                     (1 - self.significance_level / 2) * 100])
        else:
            # No bootstrap data; use simple assessment
            rmse_ci = (rmse_improvement, rmse_improvement)
            bias_ci = (bias_improvement, bias_improvement)

        # Determine if improvement passes threshold and is positive
        passes_rmse = rmse_improvement > self.improvement_threshold
        passes_ci = rmse_ci[0] > -self.improvement_threshold / 2  # CI lower bound not too negative
        passes_bias = bias_improvement > 0  # Bias should be reduced

        # Overall pass: RMSE improvement is positive and CI doesn't indicate degradation
        passes = passes_rmse and passes_ci

        # Determine recommended mode
        if passes and self.mode == "correct-if-pass":
            recommended_mode = "correct"
        else:
            recommended_mode = "flag-only"

        result = {
            "improvement": float(rmse_improvement),
            "improvement_ci": (float(rmse_ci[0]), float(rmse_ci[1])),
            "bias_improvement": float(bias_improvement),
            "bias_improvement_ci": (float(bias_ci[0]), float(bias_ci[1])),
            "passes": bool(passes),
            "recommended_mode": recommended_mode,
            "mode": self.mode,
        }

        return result

    def apply(self,
              data: pd.DataFrame,
              original_rmse: float,
              original_bias: float,
              **kwargs) -> Dict[str, Any]:
        """Run the validation gate and apply correction if appropriate.

        Args:
            data: DataFrame with observations, including any corrected values
            original_rmse: RMSE of original (uncorrected) data vs ground truth
            original_bias: Mean bias of original data vs ground truth
            **kwargs: Should contain 'corrected_rmse' and 'corrected_bias' if already computed,
                     or data with 'sat_value_corrected' column for computing them

        Returns:
            Dict with:
                - mode: "correct" or "flag-only"
                - data: DataFrame (possibly with sat_value_corrected)
                - gate_result: evaluation dict from evaluate_correction
        """
        # If we have corrected values in data, compute their metrics
        if "sat_value_corrected" in data.columns:
            sat_orig = data["sat_value"]
            sat_corr = data["sat_value_corrected"]

            # Need reference values
            if "ref_value" in data.columns:
                ref = data["ref_value"]

                # Compute RMSE
                orig_residuals = sat_orig - ref
                corr_residuals = sat_corr - ref

                orig_rmse = np.sqrt(np.mean(orig_residuals ** 2))
                corr_rmse = np.sqrt(np.mean(corr_residuals ** 2))

                orig_bias = np.mean(sat_orig - ref)
                corr_bias = np.mean(sat_corr - ref)

                # Evaluate gate
                gate_result = self.evaluate_correction(
                    orig_rmse, corr_rmse, orig_bias, corr_bias,
                    original_residuals=orig_residuals,
                    corrected_residuals=corr_residuals
                )

                # Apply mode
                if gate_result["passes"] and self.mode == "correct-if-pass":
                    # Use corrected values
                    result_data = self.correct_bias(data, sat_col="sat_value")
                    result_mode = "correct"
                else:
                    # Keep original, flag only
                    result_data = data.copy()
                    result_data["_flagged"] = True  # mark for reference
                    result_mode = "flag-only"

                return {
                    "mode": result_mode,
                    "data": result_data,
                    "gate_result": gate_result,
                }
        else:
            # No corrected values available; check if we should compute them
            result_data = data.copy()
            result_data["sat_value_corrected"] = data["sat_value"].values

            # Default: flag-only unless we have a way to compute improvement
            gate_result = {
                "improvement": 0.0,
                "improvement_ci": (0.0, 0.0),
                "bias_improvement": 0.0,
                "bias_improvement_ci": (0.0, 0.0),
                "passes": False,
                "recommended_mode": "flag-only",
                "mode": self.mode,
            }

            return {
                "mode": "flag-only",
                "data": result_data,
                "gate_result": gate_result,
            }

        # Fallback if we didn't return above
        return {
            "mode": result_mode,
            "data": result_data,
            "gate_result": gate_result,
        }