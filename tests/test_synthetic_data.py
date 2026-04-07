"""
test_synthetic_data.py – Tests for app/utils/synthetic_data.py

Covers:
  - generate_synthetic_samples() returns a DataFrame
  - Returned DataFrame has the correct number of rows
  - Returned DataFrame has exactly the requested feature columns
  - Values are clipped within the observed data range (min/max)
  - Targeting class 0 (Benign) samples from the correct distribution
  - Targeting class 1 (Malignant) samples from the correct distribution
  - class_target=None uses the full dataset distribution
  - evaluate_generalization() returns a DataFrame
  - evaluate_generalization() has Predicted_Class and Confidence columns
  - Predicted_Class values are 'Benign' or 'Malignant' strings
  - Confidence values are in [0, 1]
  - Synthetic samples fed through model produce valid predictions
"""

import numpy as np
import pandas as pd
import pytest
from utils.synthetic_data import generate_synthetic_samples, evaluate_generalization


# ---------------------------------------------------------------------------
# generate_synthetic_samples
# ---------------------------------------------------------------------------

class TestGenerateSyntheticSamples:

    def test_returns_dataframe(self, df, wdbc_features):
        result = generate_synthetic_samples(df, wdbc_features, n_samples=5)
        assert isinstance(result, pd.DataFrame)

    def test_correct_row_count_default(self, df, wdbc_features):
        n = 7
        result = generate_synthetic_samples(df, wdbc_features, n_samples=n)
        assert len(result) == n

    def test_correct_row_count_single(self, df, wdbc_features):
        result = generate_synthetic_samples(df, wdbc_features, n_samples=1)
        assert len(result) == 1

    def test_correct_columns(self, df, wdbc_features):
        result = generate_synthetic_samples(df, wdbc_features, n_samples=5)
        assert list(result.columns) == wdbc_features

    def test_values_within_min_max_bounds(self, df, wdbc_features):
        """Clipping should ensure no generated value exceeds the observed range."""
        np.random.seed(42)
        result = generate_synthetic_samples(df, wdbc_features, n_samples=50)
        for feat in wdbc_features:
            assert result[feat].min() >= df[feat].min() - 1e-9, \
                f"{feat}: generated min below data min"
            assert result[feat].max() <= df[feat].max() + 1e-9, \
                f"{feat}: generated max above data max"

    def test_class_target_0_uses_benign_distribution(self, df, wdbc_features):
        """Sampling from class 0 should center near the benign mean, not the malignant."""
        np.random.seed(0)
        benign_mean = df[df["diagnosis"] == 0][wdbc_features[0]].mean()
        malignant_mean = df[df["diagnosis"] == 1][wdbc_features[0]].mean()
        result = generate_synthetic_samples(df, wdbc_features, n_samples=200, class_target=0)
        generated_mean = result[wdbc_features[0]].mean()
        # Should be closer to benign mean than malignant mean
        assert abs(generated_mean - benign_mean) < abs(generated_mean - malignant_mean)

    def test_class_target_1_uses_malignant_distribution(self, df, wdbc_features):
        np.random.seed(0)
        benign_mean = df[df["diagnosis"] == 0][wdbc_features[0]].mean()
        malignant_mean = df[df["diagnosis"] == 1][wdbc_features[0]].mean()
        result = generate_synthetic_samples(df, wdbc_features, n_samples=200, class_target=1)
        generated_mean = result[wdbc_features[0]].mean()
        assert abs(generated_mean - malignant_mean) < abs(generated_mean - benign_mean)

    def test_class_target_none_uses_all_data(self, df, wdbc_features):
        """None target should not raise and should use the whole dataset."""
        np.random.seed(0)
        result = generate_synthetic_samples(df, wdbc_features, n_samples=10, class_target=None)
        assert len(result) == 10

    def test_no_nan_in_output(self, df, wdbc_features):
        np.random.seed(1)
        result = generate_synthetic_samples(df, wdbc_features, n_samples=20)
        assert not result.isnull().any().any(), "Generated samples contain NaN values"

    @pytest.mark.parametrize("n", [5, 10, 25, 50])
    def test_various_sample_sizes(self, df, wdbc_features, n):
        result = generate_synthetic_samples(df, wdbc_features, n_samples=n)
        assert len(result) == n


# ---------------------------------------------------------------------------
# evaluate_generalization
# ---------------------------------------------------------------------------

class TestEvaluateGeneralization:

    def test_returns_dataframe(self, df, wdbc_features, scaler, trained_model):
        synth = generate_synthetic_samples(df, wdbc_features, n_samples=5)
        result = evaluate_generalization(trained_model, scaler, synth, wdbc_features)
        assert isinstance(result, pd.DataFrame)

    def test_has_predicted_class_column(self, df, wdbc_features, scaler, trained_model):
        synth = generate_synthetic_samples(df, wdbc_features, n_samples=5)
        result = evaluate_generalization(trained_model, scaler, synth, wdbc_features)
        assert "Predicted_Class" in result.columns

    def test_has_confidence_column(self, df, wdbc_features, scaler, trained_model):
        synth = generate_synthetic_samples(df, wdbc_features, n_samples=5)
        result = evaluate_generalization(trained_model, scaler, synth, wdbc_features)
        assert "Confidence" in result.columns

    def test_predicted_class_values_are_valid_strings(self, df, wdbc_features, scaler, trained_model):
        synth = generate_synthetic_samples(df, wdbc_features, n_samples=10)
        result = evaluate_generalization(trained_model, scaler, synth, wdbc_features)
        valid = {"Malignant", "Benign"}
        for val in result["Predicted_Class"]:
            assert val in valid, f"Invalid Predicted_Class: {val}"

    def test_confidence_values_are_in_0_1(self, df, wdbc_features, scaler, trained_model):
        synth = generate_synthetic_samples(df, wdbc_features, n_samples=10)
        result = evaluate_generalization(trained_model, scaler, synth, wdbc_features)
        for conf in result["Confidence"]:
            assert 0.0 <= conf <= 1.0, f"Confidence out of range: {conf}"

    def test_output_row_count_matches_input(self, df, wdbc_features, scaler, trained_model):
        n = 12
        synth = generate_synthetic_samples(df, wdbc_features, n_samples=n)
        result = evaluate_generalization(trained_model, scaler, synth, wdbc_features)
        assert len(result) == n

    def test_original_feature_columns_preserved(self, df, wdbc_features, scaler, trained_model):
        synth = generate_synthetic_samples(df, wdbc_features, n_samples=5)
        result = evaluate_generalization(trained_model, scaler, synth, wdbc_features)
        for feat in wdbc_features:
            assert feat in result.columns
