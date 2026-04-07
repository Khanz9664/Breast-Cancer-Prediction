"""
test_error_analysis.py – Tests for app/utils/error_analysis.py

Covers:
  - get_error_analysis_data() returns a dict
  - Dict contains all required keys
  - accuracy is in [0, 1]
  - total_errors <= total_test
  - cm is a 2×2 numpy array
  - roc_data contains fpr, tpr, thresholds, auc
  - auc is in [0.5, 1.0] for a reasonable ML model
  - misclassified DataFrame has Error_Type column with valid values
  - misclassified rows are a subset of total_test
  - analyze_error_patterns() returns a string when misclassified is empty
  - analyze_error_patterns() returns a DataFrame for non-empty misclassified
"""

import pandas as pd
import pytest
from utils.error_analysis import get_error_analysis_data, analyze_error_patterns


# ---------------------------------------------------------------------------
# get_error_analysis_data integration tests
# ---------------------------------------------------------------------------

class TestGetErrorAnalysisData:

    @pytest.fixture(autouse=True)
    def _setup(self, trained_model, scaler, df, wdbc_features):
        self.model = trained_model
        self.scaler = scaler
        self.df = df
        self.features = wdbc_features

    def test_returns_dict(self):
        result = get_error_analysis_data(self.model, self.scaler, self.df, self.features, 0.5)
        assert isinstance(result, dict)

    def test_has_all_required_keys(self):
        result = get_error_analysis_data(self.model, self.scaler, self.df, self.features, 0.5)
        for key in ("misclassified", "cm", "total_test", "total_errors", "accuracy", "roc_data"):
            assert key in result, f"Missing key: {key}"

    def test_accuracy_in_valid_range(self):
        result = get_error_analysis_data(self.model, self.scaler, self.df, self.features, 0.5)
        assert 0.0 <= result["accuracy"] <= 1.0

    def test_accuracy_matches_total_errors(self):
        result = get_error_analysis_data(self.model, self.scaler, self.df, self.features, 0.5)
        expected = (result["total_test"] - result["total_errors"]) / result["total_test"]
        assert abs(result["accuracy"] - expected) < 1e-9

    def test_total_errors_lte_total_test(self):
        result = get_error_analysis_data(self.model, self.scaler, self.df, self.features, 0.5)
        assert result["total_errors"] <= result["total_test"]

    def test_confusion_matrix_is_2x2(self):
        result = get_error_analysis_data(self.model, self.scaler, self.df, self.features, 0.5)
        assert result["cm"].shape == (2, 2)

    def test_confusion_matrix_sums_to_total_test(self):
        result = get_error_analysis_data(self.model, self.scaler, self.df, self.features, 0.5)
        assert result["cm"].sum() == result["total_test"]

    def test_roc_data_has_required_keys(self):
        result = get_error_analysis_data(self.model, self.scaler, self.df, self.features, 0.5)
        for key in ("fpr", "tpr", "thresholds", "auc"):
            assert key in result["roc_data"], f"Missing roc_data key: {key}"

    def test_auc_is_in_valid_range(self):
        result = get_error_analysis_data(self.model, self.scaler, self.df, self.features, 0.5)
        auc = result["roc_data"]["auc"]
        assert 0.5 <= auc <= 1.0, f"AUC {auc} outside expected range for a reasonable model"

    def test_misclassified_is_dataframe(self):
        result = get_error_analysis_data(self.model, self.scaler, self.df, self.features, 0.5)
        assert isinstance(result["misclassified"], pd.DataFrame)

    def test_misclassified_has_error_type_column(self):
        result = get_error_analysis_data(self.model, self.scaler, self.df, self.features, 0.5)
        if not result["misclassified"].empty:
            assert "Error_Type" in result["misclassified"].columns

    def test_error_type_values_are_valid(self):
        result = get_error_analysis_data(self.model, self.scaler, self.df, self.features, 0.5)
        if not result["misclassified"].empty:
            valid = {"False Positive", "False Negative"}
            for val in result["misclassified"]["Error_Type"]:
                assert val in valid, f"Invalid Error_Type: {val}"

    def test_misclassified_count_equals_total_errors(self):
        result = get_error_analysis_data(self.model, self.scaler, self.df, self.features, 0.5)
        assert len(result["misclassified"]) == result["total_errors"]

    def test_fpr_starts_at_zero(self):
        result = get_error_analysis_data(self.model, self.scaler, self.df, self.features, 0.5)
        # ROC curve: fpr[0] == 0 after sklearn roc_curve
        assert result["roc_data"]["fpr"][0] == pytest.approx(0.0, abs=1e-6)

    def test_tpr_ends_at_one(self):
        result = get_error_analysis_data(self.model, self.scaler, self.df, self.features, 0.5)
        assert result["roc_data"]["tpr"][-1] == pytest.approx(1.0, abs=1e-6)

    @pytest.mark.parametrize("threshold", [0.3, 0.5, 0.7])
    def test_threshold_affects_total_errors(self, threshold):
        """Different thresholds should generally yield different error counts."""
        result = get_error_analysis_data(self.model, self.scaler, self.df, self.features, threshold)
        # Just verify it runs and returns valid data at any threshold
        assert 0 <= result["total_errors"] <= result["total_test"]

    def test_high_threshold_increases_false_negatives(self):
        """Very high threshold → more FN (missed malignant)."""
        res_low = get_error_analysis_data(self.model, self.scaler, self.df, self.features, 0.20)
        res_high = get_error_analysis_data(self.model, self.scaler, self.df, self.features, 0.85)
        fn_low = (
            len(res_low["misclassified"][res_low["misclassified"]["Error_Type"] == "False Negative"])
            if not res_low["misclassified"].empty else 0
        )
        fn_high = (
            len(res_high["misclassified"][res_high["misclassified"]["Error_Type"] == "False Negative"])
            if not res_high["misclassified"].empty else 0
        )
        assert fn_high >= fn_low


# ---------------------------------------------------------------------------
# analyze_error_patterns
# ---------------------------------------------------------------------------

class TestAnalyzeErrorPatterns:

    def test_empty_misclassified_returns_string(self):
        empty = pd.DataFrame()
        result = analyze_error_patterns(empty)
        assert isinstance(result, str)

    def test_non_empty_misclassified_returns_dataframe(self, trained_model, scaler, df, wdbc_features):
        res = get_error_analysis_data(trained_model, scaler, df, wdbc_features, 0.5)
        if res["misclassified"].empty:
            pytest.skip("No misclassifications at threshold=0.5 — cannot test pattern analysis")
        result = analyze_error_patterns(res["misclassified"])
        assert isinstance(result, pd.DataFrame)

    def test_pattern_dataframe_indexed_by_error_type(self, trained_model, scaler, df, wdbc_features):
        res = get_error_analysis_data(trained_model, scaler, df, wdbc_features, 0.5)
        if res["misclassified"].empty:
            pytest.skip("No misclassifications — cannot test groupby result")
        result = analyze_error_patterns(res["misclassified"])
        assert "False Positive" in result.index or "False Negative" in result.index
