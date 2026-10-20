"""WeatherTrust - AI-Powered Weather Satellite Data Reliability System.

P2 Module: ML and Validation Engine
"""

from src.inject.faults import create_faulty_dataset, inject_offset, inject_drift, inject_stripe, inject_dropout, inject_spike, inject_fault
from src.detect.rules import range_check, robust_z_score, isolation_forest_detect, rule_qc_pipeline, robust_zscore_pipeline, iforest_pipeline, combined_detector
from src.confidence.confidence import LightGBMErrorModel, ConformalIntervals, TrustScoreComputer
from src.explain.shap_reasons import shap_feature_importance, extract_top_reasons, dbscan_region_clustering
from src.correct.bias import BiasModel, ValidationGate
from src.evaluation.evaluation import (
    detection_metrics, trust_score_correlation, calibration_check,
    before_after_metrics, generate_evaluation_report
)
from src.pipeline import run_full_pipeline, quick_pipeline_demo

__all__ = [
    # Fault injection
    "create_faulty_dataset", "inject_offset", "inject_drift", "inject_stripe",
    "inject_dropout", "inject_spike", "inject_fault",
    # Detection
    "range_check", "robust_z_score", "isolation_forest_detect",
    "rule_qc_pipeline", "robust_zscore_pipeline", "iforest_pipeline",
    "combined_detector",
    # Confidence
    "LightGBMErrorModel", "ConformalIntervals", "TrustScoreComputer",
    # Explanation
    "shap_feature_importance", "extract_top_reasons", "dbscan_region_clustering",
    "quick_iforest",
    # Correction
    "BiasModel", "ValidationGate",
    # Evaluation
    "detection_metrics", "trust_score_correlation", "calibration_check",
    "before_after_metrics", "generate_evaluation_report",
    # Pipeline
    "run_full_pipeline", "quick_pipeline_demo",
]