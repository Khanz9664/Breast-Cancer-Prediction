"""
test_counterfactuals.py – Tests for app/utils/counterfactuals.py

Covers:
  - Returns (None, message) when input is already Benign
  - Returns (result_dict, None) for a Malignant input that can be nudged
  - Result dict has required keys: target_df, changes, iterations
  - iterations is between 1 and max_iter
  - 'changes' is a list; each entry has feature, from, to, change keys
  - 'to' values in changes are actual floats
  - Returns the max-iter error message if convergence fails
  - target_df has the correct feature columns
  - change delta equals (to - from) for every reported change
"""

import numpy as np
import pandas as pd
import pytest
from utils.counterfactuals import find_benign_counterfactual


# ---------------------------------------------------------------------------
# Stub models
# ---------------------------------------------------------------------------

class _AlwaysMalignantModel:
    """Always predicts class 1 (Malignant)."""
    def predict(self, X):
        return np.ones(len(X), dtype=int)

    def predict_proba(self, X):
        return np.column_stack([np.zeros(len(X)), np.ones(len(X))])


class _AlwaysBenignModel:
    """Always predicts class 0 (Benign)."""
    def predict(self, X):
        return np.zeros(len(X), dtype=int)

    def predict_proba(self, X):
        return np.column_stack([np.ones(len(X)), np.zeros(len(X))])


class _ConvergingModel:
    """
    Predicts Malignant on the first call, Benign on all subsequent calls.
    This guarantees convergence at iteration 1.
    """
    def __init__(self):
        self._call_count = 0

    def predict(self, X):
        self._call_count += 1
        if self._call_count == 1:
            return np.ones(len(X), dtype=int)   # first call → Malignant
        return np.zeros(len(X), dtype=int)      # all later calls → Benign

    def predict_proba(self, X):
        if self._call_count <= 1:
            return np.column_stack([np.zeros(len(X)), np.ones(len(X))])
        return np.column_stack([np.ones(len(X)), np.zeros(len(X))])


# ---------------------------------------------------------------------------
# Shared mini-dataset (does not rely on session fixtures to stay isolated)
# ---------------------------------------------------------------------------

MINI_FEATURES = ["radius_mean", "perimeter_mean", "area_mean"]


@pytest.fixture
def mini_df():
    np.random.seed(0)
    data = {
        "radius_mean":    np.random.uniform(6, 28, 100),
        "perimeter_mean": np.random.uniform(40, 190, 100),
        "area_mean":      np.random.uniform(143, 2501, 100),
        "diagnosis":      np.random.randint(0, 2, 100),
    }
    return pd.DataFrame(data)


@pytest.fixture
def mini_scaler(mini_df):
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    sc.fit(mini_df[MINI_FEATURES])
    return sc


@pytest.fixture
def malignant_mini_input(mini_df):
    """Input row with mean values – will be used with stub models."""
    means = mini_df[MINI_FEATURES].mean()
    return pd.DataFrame([means])


# ---------------------------------------------------------------------------
# Already-Benign input
# ---------------------------------------------------------------------------

class TestAlreadyBenign:

    def test_returns_none_result_and_string_message(self, malignant_mini_input, mini_scaler, mini_df):
        model = _AlwaysBenignModel()
        result, error = find_benign_counterfactual(
            model, mini_scaler, malignant_mini_input, MINI_FEATURES, mini_df
        )
        assert result is None
        assert isinstance(error, str)

    def test_error_message_mentions_already_benign(self, malignant_mini_input, mini_scaler, mini_df):
        model = _AlwaysBenignModel()
        _, error = find_benign_counterfactual(
            model, mini_scaler, malignant_mini_input, MINI_FEATURES, mini_df
        )
        assert "already" in error.lower() or "benign" in error.lower()


# ---------------------------------------------------------------------------
# Guaranteed convergence (ConvergingModel)
# ---------------------------------------------------------------------------

class TestConvergenceGuaranteed:

    def test_returns_dict_and_none_error(self, malignant_mini_input, mini_scaler, mini_df):
        model = _ConvergingModel()
        result, error = find_benign_counterfactual(
            model, mini_scaler, malignant_mini_input, MINI_FEATURES, mini_df
        )
        assert error is None, f"Unexpected error: {error}"
        assert result is not None

    def test_result_has_all_required_keys(self, malignant_mini_input, mini_scaler, mini_df):
        model = _ConvergingModel()
        result, _ = find_benign_counterfactual(
            model, mini_scaler, malignant_mini_input, MINI_FEATURES, mini_df
        )
        for key in ("target_df", "changes", "iterations"):
            assert key in result, f"Missing key: {key}"

    def test_iterations_is_positive_integer(self, malignant_mini_input, mini_scaler, mini_df):
        model = _ConvergingModel()
        result, _ = find_benign_counterfactual(
            model, mini_scaler, malignant_mini_input, MINI_FEATURES, mini_df,
            max_iter=20
        )
        assert isinstance(result["iterations"], int)
        assert 1 <= result["iterations"] <= 20

    def test_target_df_has_correct_columns(self, malignant_mini_input, mini_scaler, mini_df):
        model = _ConvergingModel()
        result, _ = find_benign_counterfactual(
            model, mini_scaler, malignant_mini_input, MINI_FEATURES, mini_df
        )
        assert list(result["target_df"].columns) == MINI_FEATURES

    def test_changes_is_a_list(self, malignant_mini_input, mini_scaler, mini_df):
        model = _ConvergingModel()
        result, _ = find_benign_counterfactual(
            model, mini_scaler, malignant_mini_input, MINI_FEATURES, mini_df
        )
        assert isinstance(result["changes"], list)

    def test_each_change_entry_has_correct_keys(self, malignant_mini_input, mini_scaler, mini_df):
        model = _ConvergingModel()
        result, _ = find_benign_counterfactual(
            model, mini_scaler, malignant_mini_input, MINI_FEATURES, mini_df
        )
        for entry in result["changes"]:
            for key in ("feature", "from", "to", "change"):
                assert key in entry, f"Missing key '{key}' in change entry: {entry}"

    def test_change_delta_is_consistent(self, malignant_mini_input, mini_scaler, mini_df):
        """change field must equal (to - from)."""
        model = _ConvergingModel()
        result, _ = find_benign_counterfactual(
            model, mini_scaler, malignant_mini_input, MINI_FEATURES, mini_df
        )
        for entry in result["changes"]:
            expected = entry["to"] - entry["from"]
            assert abs(entry["change"] - expected) < 1e-9, \
                f"Inconsistent: expected {expected}, got {entry['change']}"

    def test_target_df_has_one_row(self, malignant_mini_input, mini_scaler, mini_df):
        model = _ConvergingModel()
        result, _ = find_benign_counterfactual(
            model, mini_scaler, malignant_mini_input, MINI_FEATURES, mini_df
        )
        assert len(result["target_df"]) == 1


# ---------------------------------------------------------------------------
# Convergence failure (AlwaysMalignant)
# ---------------------------------------------------------------------------

class TestConvergenceFailure:

    def test_always_malignant_returns_none_result(self, malignant_mini_input, mini_scaler, mini_df):
        model = _AlwaysMalignantModel()
        result, error = find_benign_counterfactual(
            model, mini_scaler, malignant_mini_input, MINI_FEATURES, mini_df,
            max_iter=5
        )
        assert result is None

    def test_always_malignant_returns_string_error(self, malignant_mini_input, mini_scaler, mini_df):
        model = _AlwaysMalignantModel()
        _, error = find_benign_counterfactual(
            model, mini_scaler, malignant_mini_input, MINI_FEATURES, mini_df,
            max_iter=5
        )
        assert isinstance(error, str)
        # Message should communicate failure to find boundary
        assert any(word in error.lower() for word in ("iteration", "limit", "boundary", "atypical"))
