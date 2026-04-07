"""
test_robustness.py – Tests for app/utils/robustness.py

Covers:
  - Non-forest model returns the deterministic fallback dict
  - Random Forest model returns the full dict with all required keys
  - stability_score is in [0, 100]
  - ci[0] >= 0 and ci[1] <= 1 (valid probability interval)
  - ci[0] <= ci[1]
  - stability_level is one of the three valid string labels
  - stability_color matches the level correctly
  - all_tree_probs length matches estimators_ count
  - variance is >= 0
  - std_dev >= 0
  - Highly Stable is reported for a very consistent input
  - Borderline level is reported when predictions are highly variable
"""

import numpy as np
import pytest
from utils.robustness import calculate_model_robustness


# ---------------------------------------------------------------------------
# Helpers / Stubs
# ---------------------------------------------------------------------------

class _FakeTree:
    """Minimal tree stub: always predicts a fixed malignant probability."""
    def __init__(self, prob_malignant: float):
        self._prob = prob_malignant

    def predict_proba(self, X):
        return np.array([[1 - self._prob, self._prob]])


class _FakeForest:
    """Stub ensemble with uniform tree predictions."""
    def __init__(self, n_trees: int, prob_malignant: float):
        self.estimators_ = [_FakeTree(prob_malignant) for _ in range(n_trees)]


class _FakeVariableForest:
    """Forest whose trees disagree wildly (high variance)."""
    def __init__(self, n_trees: int = 20):
        probs = np.linspace(0.0, 1.0, n_trees)
        self.estimators_ = [_FakeTree(p) for p in probs]


class _NonForestModel:
    """Model with no estimators_ attribute."""
    def predict_proba(self, X):
        return np.array([[0.3, 0.7]])


DUMMY_INPUT = np.zeros((1, 10))


# ---------------------------------------------------------------------------
# Non-Forest Models
# ---------------------------------------------------------------------------

class TestNonForestFallback:

    def test_returns_dict(self):
        model = _NonForestModel()
        result = calculate_model_robustness(model, DUMMY_INPUT)
        assert isinstance(result, dict)

    def test_fallback_has_stability_level(self):
        model = _NonForestModel()
        result = calculate_model_robustness(model, DUMMY_INPUT)
        assert "stability_level" in result
        assert "Deterministic" in result["stability_level"]

    def test_fallback_has_confidence_interval(self):
        model = _NonForestModel()
        result = calculate_model_robustness(model, DUMMY_INPUT)
        assert "confidence_interval" in result or "ci" in result


# ---------------------------------------------------------------------------
# Random Forest Models
# ---------------------------------------------------------------------------

class TestRandomForestRobustness:

    def test_returns_all_required_keys(self):
        model = _FakeForest(10, 0.8)
        result = calculate_model_robustness(model, DUMMY_INPUT)
        for key in ("all_tree_probs", "variance", "std_dev", "ci",
                    "stability_score", "stability_level", "stability_color"):
            assert key in result, f"Missing key: {key}"

    def test_all_tree_probs_length(self):
        n = 15
        model = _FakeForest(n, 0.7)
        result = calculate_model_robustness(model, DUMMY_INPUT)
        assert len(result["all_tree_probs"]) == n

    def test_stability_score_in_valid_range(self):
        model = _FakeForest(10, 0.9)
        result = calculate_model_robustness(model, DUMMY_INPUT)
        assert 0.0 <= result["stability_score"] <= 100.0

    def test_variance_is_non_negative(self):
        model = _FakeForest(10, 0.6)
        result = calculate_model_robustness(model, DUMMY_INPUT)
        assert result["variance"] >= 0.0

    def test_std_dev_is_non_negative(self):
        model = _FakeForest(10, 0.6)
        result = calculate_model_robustness(model, DUMMY_INPUT)
        assert result["std_dev"] >= 0.0

    def test_ci_lower_is_non_negative(self):
        model = _FakeForest(10, 0.5)
        result = calculate_model_robustness(model, DUMMY_INPUT)
        assert result["ci"][0] >= 0.0

    def test_ci_upper_is_at_most_one(self):
        model = _FakeForest(10, 0.5)
        result = calculate_model_robustness(model, DUMMY_INPUT)
        assert result["ci"][1] <= 1.0

    def test_ci_lower_lte_upper(self):
        model = _FakeForest(10, 0.5)
        result = calculate_model_robustness(model, DUMMY_INPUT)
        assert result["ci"][0] <= result["ci"][1]

    def test_stability_level_is_valid_string(self):
        model = _FakeForest(10, 0.9)
        result = calculate_model_robustness(model, DUMMY_INPUT)
        valid_levels = {"Highly Stable", "Stable", "Borderline / High Variance"}
        assert result["stability_level"] in valid_levels

    def test_uniform_predictions_are_highly_stable(self):
        """All trees predicting the same probability → zero variance → Highly Stable."""
        model = _FakeForest(20, 0.9)
        result = calculate_model_robustness(model, DUMMY_INPUT)
        assert result["stability_level"] == "Highly Stable"
        assert result["variance"] == pytest.approx(0.0, abs=1e-10)

    def test_high_variance_gives_borderline_level(self):
        """Trees spread uniformly from 0→1 → high std → Borderline."""
        model = _FakeVariableForest(n_trees=100)
        result = calculate_model_robustness(model, DUMMY_INPUT)
        assert result["stability_level"] == "Borderline / High Variance"

    def test_stability_color_matches_level(self):
        model = _FakeForest(20, 0.9)
        result = calculate_model_robustness(model, DUMMY_INPUT)
        if result["stability_level"] == "Highly Stable":
            assert result["stability_color"] == "#51cf66"
        elif result["stability_level"] == "Stable":
            assert result["stability_color"] == "#fcc419"
        else:
            assert result["stability_color"] == "#ff6b6b"

    def test_variance_near_zero_for_uniform_forest(self):
        model = _FakeForest(50, 0.75)
        result = calculate_model_robustness(model, DUMMY_INPUT)
        assert result["variance"] < 1e-10


# ---------------------------------------------------------------------------
# Real sklearn RF (integration-level)
# ---------------------------------------------------------------------------

class TestWithRealRandomForest:

    def test_real_rf_returns_valid_structure(self, trained_model, scaler, sample_input, wdbc_features):
        scaled = scaler.transform(sample_input[wdbc_features])
        result = calculate_model_robustness(trained_model, scaled)
        assert "stability_score" in result
        assert 0.0 <= result["stability_score"] <= 100.0
        assert result["ci"][0] <= result["ci"][1]
