"""
test_report_generator.py – Tests for app/utils/report_generator.py

Covers:
  - generate_patient_report() returns bytes
  - Output starts with the PDF magic header %PDF
  - Output is non-empty / of non-trivial length
  - Works for both Malignant and Benign predictions
  - Works with SHAP values provided
  - Works with SHAP values as None
  - Works with robustness dict provided
  - Works with robustness=None
  - Works with multiple clinical insights
  - Works with empty insights list
  - High confidence value is preserved (smoke test, not full PDF parse)
  - Various threshold values are accepted
"""

import pytest
import numpy as np
import pandas as pd
from utils.report_generator import generate_patient_report


# ---------------------------------------------------------------------------
# Helpers & fixtures
# ---------------------------------------------------------------------------

def _make_inputs(wdbc_features, df):
    """Return a single-row DataFrame with mean feature values."""
    means = df[wdbc_features].mean()
    return pd.DataFrame([means], columns=wdbc_features)


def _dummy_insights(n: int = 2):
    return [
        {
            "feature": f"Feature {i}",
            "insight": f"Insight text {i}.",
            "severity": ["high", "medium", "low"][i % 3]
        }
        for i in range(n)
    ]


def _dummy_shap(n_features: int):
    return np.random.uniform(-0.3, 0.3, n_features)


def _make_robustness():
    return {
        "stability_level": "Highly Stable",
        "stability_score": 92.0,
        "ci": (0.78, 0.96),
    }


# ---------------------------------------------------------------------------
# Core return type tests
# ---------------------------------------------------------------------------

class TestReturnType:

    def test_returns_bytes(self, df, wdbc_features):
        inp = _make_inputs(wdbc_features, df)
        result = generate_patient_report(
            "Benign", 0.88, [0.88, 0.12],
            wdbc_features, inp, None, [], 0.50
        )
        assert isinstance(result, bytes)

    def test_output_is_non_empty(self, df, wdbc_features):
        inp = _make_inputs(wdbc_features, df)
        result = generate_patient_report(
            "Benign", 0.88, [0.88, 0.12],
            wdbc_features, inp, None, [], 0.50
        )
        assert len(result) > 1000  # a real PDF is never trivially small

    def test_output_starts_with_pdf_magic_bytes(self, df, wdbc_features):
        inp = _make_inputs(wdbc_features, df)
        result = generate_patient_report(
            "Malignant", 0.91, [0.09, 0.91],
            wdbc_features, inp, None, [], 0.50
        )
        # PDF files always start with %PDF
        assert result[:4] == b"%PDF"


# ---------------------------------------------------------------------------
# Prediction class variants
# ---------------------------------------------------------------------------

class TestPredictionVariants:

    def test_benign_prediction_produces_output(self, df, wdbc_features):
        inp = _make_inputs(wdbc_features, df)
        result = generate_patient_report(
            "Benign", 0.92, [0.92, 0.08],
            wdbc_features, inp, None, _dummy_insights(1), 0.50
        )
        assert isinstance(result, bytes) and len(result) > 0

    def test_malignant_prediction_produces_output(self, df, wdbc_features):
        inp = _make_inputs(wdbc_features, df)
        result = generate_patient_report(
            "Malignant", 0.87, [0.13, 0.87],
            wdbc_features, inp, None, _dummy_insights(2), 0.50
        )
        assert isinstance(result, bytes) and len(result) > 0


# ---------------------------------------------------------------------------
# SHAP values
# ---------------------------------------------------------------------------

class TestSHAPValueHandling:

    def test_with_shap_values_provided(self, df, wdbc_features):
        inp = _make_inputs(wdbc_features, df)
        shap_vals = _dummy_shap(len(wdbc_features))
        result = generate_patient_report(
            "Malignant", 0.85, [0.15, 0.85],
            wdbc_features, inp, shap_vals, _dummy_insights(), 0.50
        )
        assert isinstance(result, bytes) and len(result) > 0

    def test_with_shap_none(self, df, wdbc_features):
        inp = _make_inputs(wdbc_features, df)
        result = generate_patient_report(
            "Benign", 0.82, [0.82, 0.18],
            wdbc_features, inp, None, _dummy_insights(), 0.50
        )
        assert isinstance(result, bytes) and len(result) > 0

    def test_with_shap_all_zeros(self, df, wdbc_features):
        inp = _make_inputs(wdbc_features, df)
        shap_vals = np.zeros(len(wdbc_features))
        result = generate_patient_report(
            "Benign", 0.90, [0.90, 0.10],
            wdbc_features, inp, shap_vals, [], 0.50
        )
        assert isinstance(result, bytes)

    def test_with_shap_extreme_values(self, df, wdbc_features):
        inp = _make_inputs(wdbc_features, df)
        shap_vals = np.full(len(wdbc_features), 999.9)
        result = generate_patient_report(
            "Malignant", 0.99, [0.01, 0.99],
            wdbc_features, inp, shap_vals, [], 0.50
        )
        assert isinstance(result, bytes)


# ---------------------------------------------------------------------------
# Robustness section
# ---------------------------------------------------------------------------

class TestRobustnessSection:

    def test_with_robustness_provided(self, df, wdbc_features):
        inp = _make_inputs(wdbc_features, df)
        result = generate_patient_report(
            "Benign", 0.89, [0.89, 0.11],
            wdbc_features, inp, None, [], 0.50,
            robustness=_make_robustness()
        )
        assert isinstance(result, bytes) and len(result) > 0

    def test_with_robustness_none(self, df, wdbc_features):
        inp = _make_inputs(wdbc_features, df)
        result = generate_patient_report(
            "Benign", 0.89, [0.89, 0.11],
            wdbc_features, inp, None, [], 0.50,
            robustness=None
        )
        assert isinstance(result, bytes) and len(result) > 0


# ---------------------------------------------------------------------------
# Clinical insights
# ---------------------------------------------------------------------------

class TestClinicalInsightsSection:

    def test_empty_insights_list(self, df, wdbc_features):
        inp = _make_inputs(wdbc_features, df)
        result = generate_patient_report(
            "Benign", 0.88, [0.88, 0.12],
            wdbc_features, inp, None, [], 0.50
        )
        assert isinstance(result, bytes)

    def test_many_insights(self, df, wdbc_features):
        inp = _make_inputs(wdbc_features, df)
        insights = _dummy_insights(6)
        result = generate_patient_report(
            "Malignant", 0.90, [0.10, 0.90],
            wdbc_features, inp, None, insights, 0.50
        )
        assert isinstance(result, bytes)

    def test_all_severity_levels(self, df, wdbc_features):
        inp = _make_inputs(wdbc_features, df)
        insights = [
            {"feature": "Radius", "insight": "High.", "severity": "high"},
            {"feature": "Area", "insight": "Medium.", "severity": "medium"},
            {"feature": "Compactness", "insight": "Low.", "severity": "low"},
        ]
        result = generate_patient_report(
            "Malignant", 0.88, [0.12, 0.88],
            wdbc_features, inp, None, insights, 0.50
        )
        assert isinstance(result, bytes)


# ---------------------------------------------------------------------------
# Threshold variants
# ---------------------------------------------------------------------------

class TestThresholdVariants:

    @pytest.mark.parametrize("threshold", [0.20, 0.35, 0.50, 0.65, 0.80])
    def test_various_thresholds(self, df, wdbc_features, threshold):
        inp = _make_inputs(wdbc_features, df)
        result = generate_patient_report(
            "Benign", 0.85, [0.85, 0.15],
            wdbc_features, inp, None, [], threshold
        )
        assert isinstance(result, bytes) and len(result) > 0
