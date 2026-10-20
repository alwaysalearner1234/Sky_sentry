"""
Validation gate pipeline for bias correction.

Integrates the bias model with the validation gate to decide
whether correction should be applied based on held-out ground truth
improvement.
"""

from src.correct.bias import BiasModel, ValidationGate


def correction_pipeline(data: pd.DataFrame,
                        train_frac: float = 0.8,
                        bias_model_type: str = "linear",
                        gate_mode: str = "flag-only",
                        rmse_threshold: float = 0.01,
                        **kwargs) -> Dict[str, Any]:
    """Run the full bias correction pipeline with validation gate.

    The pipeline:
      1. Splits data into train/test (time-based or random)
      2. Trains bias model on training data
      3. Predicts bias on test data
      4. Evaluates gate: only corrects if test RMSE improves
      5. Applies correction or flags accordingly

    Args:
        data: Full DataFrame with sat_value, ref_value columns
        train_frac: Fraction of data for training (time-split if applicable)
        bias_model_type: "linear" or "gbm"
        gate_mode: "flag-only" or "correct-if-pass"
        rmse_threshold: Minimum RMSE improvement fraction to pass gate
        **kwargs: Passed to gate evaluation

    Returns:
        Dict with:
            - mode: "correct" or "flag-only"
            - data: DataFrame with sat_value_corrected (if corrected) or original
            - gate_result: Evaluation results
            - bias_model: Trained BiasModel (for inspection)
            - train_indices: Indices used for training
            - test_indices: Indices used for testing
    """
    data = data.copy()

    # Time-based split: take first train_frac for training, rest for testing
    # If time_utc column exists, sort by it first
    if "time_utc" in data.columns:
        data = data.sort_values("time_utc").reset_index(drop=True)

    n = len(data)
    train_size = int(n * train_frac)

    if train_size < 3 or n - train_size < 3:
        # Not enough data for split; use all as train, test is minimal
        train_indices = np.arange(n)
        test_indices = np.array([], dtype=int)
    else:
        train_indices = np.arange(train_size)
        test_indices = np.arange(train_size, n)

    train_data = data.iloc[train_indices].copy()
    test_data = data.iloc[test_indices].copy()

    # Train bias model on training data
    bias_model = BiasModel(model_type=bias_model_type)
    train_features = bias_model.engineer_features(train_data)
    bias_model.train(train_features, bias_col="raw_residual")

    # Predict bias on test data
    test_with_bias = bias_model.predict_bias(test_data)

    # Compute bias-corrected test values
    test_corrected = bias_model.correct_bias(test_with_bias, sat_col="sat_value")

    # Compute metrics
    if "ref_value" in test_data.columns:
        ref = test_data["ref_value"]

        # Original (uncorrected) residuals
        orig_residuals = test_data["sat_value"] - ref
        orig_rmse = np.sqrt(np.mean(orig_residuals ** 2))
        orig_bias = np.mean(orig_residuals)

        # Corrected residuals
        corr_residuals = test_corrected["sat_value_corrected"] - ref
        corr_rmse = np.sqrt(np.mean(corr_residuals ** 2))
        corr_bias = np.mean(corr_residuals)

        # Evaluate validation gate
        gate = ValidationGate(
            improvement_threshold=rmse_threshold,
            mode=gate_mode
        )

        gate_result = gate.evaluate_correction(
            orig_rmse, corr_rmse, orig_bias, corr_bias
        )

        # Apply the gate decision
        if gate_result["passes"] and gate_mode == "correct-if-pass":
            # Use corrected values for the full dataset
            # Build full corrected dataset
            full_corrected = bias_model.correct_bias(
                bias_model.engineer_features(data), sat_col="sat_value"
            )
            result_mode = "correct"
        else:
            # Flag-only: keep original values, mark for reference
            full_corrected = data.copy()
            full_corrected["_flagged"] = True
            result_mode = "flag-only"

        return {
            "mode": result_mode,
            "data": full_corrected,
            "gate_result": gate_result,
            "bias_model": bias_model,
            "train_indices": train_indices.tolist(),
            "test_indices": test_indices.tolist(),
            "original_rmse": float(orig_rmse),
            "corrected_rmse": float(corr_rmse),
            "rmse_improvement": gate_result["improvement"],
        }
    else:
        # No reference values; can't evaluate gate, flag-only
        return {
            "mode": "flag-only",
            "data": data.copy(),
            "gate_result": {
                "improvement": 0.0,
                "improvement_ci": (0.0, 0.0),
                "bias_improvement": 0.0,
                "bias_improvement_ci": (0.0, 0.0),
                "passes": False,
                "recommended_mode": "flag-only",
                "mode": gate_mode,
            },
            "bias_model": BiasModel(),
            "train_indices": [],
            "test_indices": [],
            "original_rmse": np.nan,
            "corrected_rmse": np.nan,
            "rmse_improvement": 0.0,
        }