"""
test_sensitivity.py – Tests for app/utils/sensitivity.py

Covers:
  - calculate_sensitivity_curve() returns a dict
  - Returned dict has keys: x, y, current_val, current_prob
  - x and y are lists of the correct length (steps)
  - x values span [f_min, f_max] of the dataset
  - y values are all in [0, 1] (valid probabilities)
  - current_val matches the input value for the chosen feature
  - current_prob is in [0, 1]
  - Varying steps parameter changes x/y length accordingly
  - Works correctly for every feature in the priority list
"""

import pytest
from utils.sensitivity import calculate_sensitivity_curve


class TestCalculateSensitivityCurve:

    def test_returns_dict(self, trained_model, scaler, df, sample_input, wdbc_features):
        result = calculate_sensitivity_curve(
            trained_model, scaler, sample_input,
            wdbc_features[0], wdbc_features, df
        )
        assert isinstance(result, dict)

    def test_dict_has_required_keys(self, trained_model, scaler, df, sample_input, wdbc_features):
        result = calculate_sensitivity_curve(
            trained_model, scaler, sample_input,
            wdbc_features[0], wdbc_features, df
        )
        for key in ("x", "y", "current_val", "current_prob"):
            assert key in result, f"Missing key: {key}"

    def test_x_is_list_of_correct_length(self, trained_model, scaler, df, sample_input, wdbc_features):
        steps = 30
        result = calculate_sensitivity_curve(
            trained_model, scaler, sample_input,
            wdbc_features[0], wdbc_features, df, steps=steps
        )
        assert isinstance(result["x"], list)
        assert len(result["x"]) == steps

    def test_y_is_list_of_correct_length(self, trained_model, scaler, df, sample_input, wdbc_features):
        steps = 30
        result = calculate_sensitivity_curve(
            trained_model, scaler, sample_input,
            wdbc_features[0], wdbc_features, df, steps=steps
        )
        assert isinstance(result["y"], list)
        assert len(result["y"]) == steps

    def test_x_starts_at_feature_minimum(self, trained_model, scaler, df, sample_input, wdbc_features):
        feat = wdbc_features[0]
        result = calculate_sensitivity_curve(
            trained_model, scaler, sample_input, feat, wdbc_features, df
        )
        assert abs(result["x"][0] - df[feat].min()) < 1e-9

    def test_x_ends_at_feature_maximum(self, trained_model, scaler, df, sample_input, wdbc_features):
        feat = wdbc_features[0]
        result = calculate_sensitivity_curve(
            trained_model, scaler, sample_input, feat, wdbc_features, df
        )
        assert abs(result["x"][-1] - df[feat].max()) < 1e-9

    def test_y_values_are_valid_probabilities(self, trained_model, scaler, df, sample_input, wdbc_features):
        result = calculate_sensitivity_curve(
            trained_model, scaler, sample_input,
            wdbc_features[0], wdbc_features, df
        )
        for prob in result["y"]:
            assert 0.0 <= prob <= 1.0, f"Probability out of range: {prob}"

    def test_current_prob_is_valid(self, trained_model, scaler, df, sample_input, wdbc_features):
        result = calculate_sensitivity_curve(
            trained_model, scaler, sample_input,
            wdbc_features[0], wdbc_features, df
        )
        assert 0.0 <= result["current_prob"] <= 1.0

    def test_current_val_matches_feature_value(self, trained_model, scaler, df, sample_input, wdbc_features):
        feat = wdbc_features[0]
        result = calculate_sensitivity_curve(
            trained_model, scaler, sample_input, feat, wdbc_features, df
        )
        assert abs(result["current_val"] - sample_input[feat].iloc[0]) < 1e-9

    @pytest.mark.parametrize("steps", [10, 25, 50, 100])
    def test_various_step_counts(self, trained_model, scaler, df, sample_input, wdbc_features, steps):
        result = calculate_sensitivity_curve(
            trained_model, scaler, sample_input,
            wdbc_features[0], wdbc_features, df, steps=steps
        )
        assert len(result["x"]) == steps
        assert len(result["y"]) == steps

    def test_works_for_all_priority_features(self, trained_model, scaler, df, sample_input, wdbc_features):
        """No feature should raise an exception when used as the sensitivity variable."""
        for feat in wdbc_features:
            result = calculate_sensitivity_curve(
                trained_model, scaler, sample_input, feat, wdbc_features, df
            )
            assert len(result["y"]) > 0, f"Empty y for feature: {feat}"

    def test_x_is_monotonically_increasing(self, trained_model, scaler, df, sample_input, wdbc_features):
        result = calculate_sensitivity_curve(
            trained_model, scaler, sample_input,
            wdbc_features[0], wdbc_features, df
        )
        x = result["x"]
        for i in range(len(x) - 1):
            assert x[i] <= x[i + 1], "x values are not monotonically increasing"
