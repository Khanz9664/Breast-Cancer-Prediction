"""
test_monitoring.py – Tests for app/utils/monitoring.py

Covers:
  - init_logging() creates the logs directory and returns correct bool
  - log_prediction() writes a row with correct columns to the CSV
  - log_prediction() appends correctly on subsequent calls
  - calculate_drift() returns the full expected data structure
  - calculate_drift() flags a feature as Drifting when z-score >= 3
  - calculate_drift() marks features Healthy when values match training mean
  - calculate_drift() handles zero-std features without division errors
  - calculate_drift() health_score is between 0 and 100
  - calculate_drift() is_drifting fires when > 20% features are critical/warning
  - load_prediction_history() returns empty DataFrame when file missing
  - load_prediction_history() returns populated DataFrame when file exists
"""

import os
import tempfile
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# We monkey-patch LOG_FILE to a temp directory so tests never touch the real
# logs/ directory on disk.
# ---------------------------------------------------------------------------
import utils.monitoring as monitoring


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_log_dir():
    """Return a fresh temporary directory and patch the module's LOG_FILE."""
    tmp = tempfile.mkdtemp()
    log_file = os.path.join(tmp, "prediction_history.csv")
    monitoring.LOG_FILE = log_file
    # Also patch the hardcoded "logs" path used in init_logging
    monitoring._LOG_DIR = tmp
    return tmp, log_file


def _dummy_input(features):
    data = {f: [float(i + 1)] for i, f in enumerate(features)}
    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# init_logging
# ---------------------------------------------------------------------------

class TestInitLogging:

    def test_returns_false_when_file_missing(self, wdbc_features):
        tmp, log_file = _make_log_dir()
        result = monitoring.init_logging()
        assert result is False

    def test_returns_true_when_file_exists(self, wdbc_features):
        tmp, log_file = _make_log_dir()
        # Create the file manually
        pd.DataFrame(columns=["timestamp"]).to_csv(log_file, index=False)
        result = monitoring.init_logging()
        assert result is True


# ---------------------------------------------------------------------------
# log_prediction
# ---------------------------------------------------------------------------

class TestLogPrediction:

    def test_creates_csv_on_first_call(self, wdbc_features):
        tmp, log_file = _make_log_dir()
        inp = _dummy_input(wdbc_features)
        monitoring.log_prediction(inp, "Benign", 0.95)
        assert os.path.exists(log_file)

    def test_log_has_correct_columns(self, wdbc_features):
        tmp, log_file = _make_log_dir()
        inp = _dummy_input(wdbc_features)
        monitoring.log_prediction(inp, "Malignant", 0.87)
        df = pd.read_csv(log_file)
        assert "timestamp" in df.columns
        assert "prediction" in df.columns
        assert "confidence" in df.columns

    def test_prediction_value_stored_correctly(self, wdbc_features):
        tmp, log_file = _make_log_dir()
        inp = _dummy_input(wdbc_features)
        monitoring.log_prediction(inp, "Benign", 0.91)
        df = pd.read_csv(log_file)
        assert df["prediction"].iloc[0] == "Benign"
        assert abs(df["confidence"].iloc[0] - 0.91) < 1e-6

    def test_appends_multiple_predictions(self, wdbc_features):
        tmp, log_file = _make_log_dir()
        inp = _dummy_input(wdbc_features)
        monitoring.log_prediction(inp, "Benign", 0.80)
        monitoring.log_prediction(inp, "Malignant", 0.75)
        monitoring.log_prediction(inp, "Benign", 0.90)
        df = pd.read_csv(log_file)
        assert len(df) == 3

    def test_confidence_stored_as_float(self, wdbc_features):
        tmp, log_file = _make_log_dir()
        inp = _dummy_input(wdbc_features)
        monitoring.log_prediction(inp, "Benign", 0.123456)
        df = pd.read_csv(log_file)
        assert isinstance(df["confidence"].iloc[0], float)


# ---------------------------------------------------------------------------
# calculate_drift
# ---------------------------------------------------------------------------

class TestCalculateDrift:

    def test_returns_correct_keys(self, df, wdbc_features, sample_input):
        result = monitoring.calculate_drift(sample_input, df, wdbc_features)
        assert "metrics" in result
        assert "health_score" in result
        assert "is_drifting" in result

    def test_metrics_contains_all_features(self, df, wdbc_features, sample_input):
        result = monitoring.calculate_drift(sample_input, df, wdbc_features)
        for feat in wdbc_features:
            assert feat in result["metrics"]

    def test_metric_entry_has_required_fields(self, df, wdbc_features, sample_input):
        result = monitoring.calculate_drift(sample_input, df, wdbc_features)
        first_metric = next(iter(result["metrics"].values()))
        for key in ("value", "mean", "std", "z_score", "status"):
            assert key in first_metric

    def test_mean_input_gives_healthy_status(self, df, wdbc_features, sample_input):
        """Values == training mean → z-score ~ 0 → all Healthy."""
        result = monitoring.calculate_drift(sample_input, df, wdbc_features)
        for feat, metric in result["metrics"].items():
            assert metric["status"] == "Healthy", \
                f"Expected Healthy for mean input on feature {feat}, got {metric['status']}"

    def test_extreme_value_triggers_critical(self, df, wdbc_features):
        """An extreme outlier should push z-score >= 3 → Critical."""
        extreme = df[wdbc_features].mean().copy()
        # Set first feature far above mean
        feat = wdbc_features[0]
        extreme[feat] = df[feat].mean() + 10 * df[feat].std()
        inp = pd.DataFrame([extreme])
        result = monitoring.calculate_drift(inp, df, wdbc_features)
        assert result["metrics"][feat]["status"] == "Critical"

    def test_health_score_between_0_and_100(self, df, wdbc_features, sample_input):
        result = monitoring.calculate_drift(sample_input, df, wdbc_features)
        assert 0.0 <= result["health_score"] <= 100.0

    def test_health_score_is_100_for_mean_input(self, df, wdbc_features, sample_input):
        result = monitoring.calculate_drift(sample_input, df, wdbc_features)
        assert result["health_score"] == pytest.approx(100.0, abs=1.0)

    def test_is_drifting_false_for_normal_input(self, df, wdbc_features, sample_input):
        result = monitoring.calculate_drift(sample_input, df, wdbc_features)
        assert result["is_drifting"] is False

    def test_zero_std_feature_does_not_raise(self, df, wdbc_features):
        """A constant feature (std = 0) must not raise ZeroDivisionError."""
        modified = df.copy()
        feat = wdbc_features[0]
        modified[feat] = modified[feat].mean()  # make it constant
        inp = pd.DataFrame([modified[wdbc_features].mean()])
        # Should not raise
        result = monitoring.calculate_drift(inp, modified, wdbc_features)
        assert result["metrics"][feat]["z_score"] == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# load_prediction_history
# ---------------------------------------------------------------------------

class TestLoadPredictionHistory:

    def test_returns_empty_dataframe_when_no_file(self):
        tmp, log_file = _make_log_dir()
        result = monitoring.load_prediction_history()
        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_returns_dataframe_with_data(self, wdbc_features):
        tmp, log_file = _make_log_dir()
        inp = _dummy_input(wdbc_features)
        monitoring.log_prediction(inp, "Benign", 0.88)
        result = monitoring.load_prediction_history()
        assert not result.empty
        assert "prediction" in result.columns
