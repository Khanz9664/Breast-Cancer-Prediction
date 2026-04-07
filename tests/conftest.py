"""
conftest.py – Shared pytest fixtures for the Breast-Cancer-Prediction test suite.

All fixtures are designed to be lightweight and completely self-contained so that
the test suite can run in CI without the real model artifact or any Streamlit
runtime.
"""

import sys
import os
import types

import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Path setup – make `app/` importable without installing the package
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
APP_DIR = os.path.join(PROJECT_ROOT, "app")
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)


# ---------------------------------------------------------------------------
# Stub out `streamlit` so utility modules that import it can be loaded in
# plain Python (no Streamlit server context required in CI).
# ---------------------------------------------------------------------------
def _make_streamlit_stub():
    st = types.ModuleType("streamlit")

    def _passthrough(fn=None, **kwargs):
        if fn is not None:
            return fn
        return lambda f: f

    st.cache_resource = _passthrough
    st.cache_data = _passthrough

    for _attr in ("error", "warning", "stop", "success", "info", "write",
                  "markdown", "subheader", "header", "title",
                  "dataframe", "plotly_chart", "metric", "caption",
                  "expander", "container", "columns", "tabs",
                  "button", "download_button", "selectbox", "slider",
                  "multiselect", "spinner", "rerun", "set_page_config"):
        setattr(st, _attr, lambda *a, **kw: None)

    st.session_state = {}
    sys.modules["streamlit"] = st
    return st


_make_streamlit_stub()


# ---------------------------------------------------------------------------
# Priority features (matching the app's naming convention)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def priority_features():
    return [
        "radius_mean", "perimeter_mean", "area_mean", "compactness_mean",
        "concavity_mean", "concave_points_mean", "radius_se", "perimeter_se",
        "area_se", "radius_worst", "perimeter_worst", "area_worst",
        "compactness_worst", "concavity_worst", "concave_points_worst",
    ]


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def _build_fallback_df(priority_features):
    """
    Build a WDBC-compatible DataFrame using sklearn's breast_cancer dataset.
    sklearn uses 'mean radius' style names; we remap to 'radius_mean' style.
    """
    from sklearn.datasets import load_breast_cancer

    bc = load_breast_cancer()
    sk_names = list(bc.feature_names)

    stat_map = {
        "mean radius":          "radius_mean",
        "mean perimeter":       "perimeter_mean",
        "mean area":            "area_mean",
        "mean compactness":     "compactness_mean",
        "mean concavity":       "concavity_mean",
        "mean concave points":  "concave_points_mean",
        "radius error":         "radius_se",
        "perimeter error":      "perimeter_se",
        "area error":           "area_se",
        "worst radius":         "radius_worst",
        "worst perimeter":      "perimeter_worst",
        "worst area":           "area_worst",
        "worst compactness":    "compactness_worst",
        "worst concavity":      "concavity_worst",
        "worst concave points": "concave_points_worst",
    }

    raw = pd.DataFrame(bc.data, columns=sk_names)
    raw = raw.rename(columns=stat_map)
    # sklearn: 0 = malignant → app: 1 = malignant
    raw["diagnosis"] = (bc.target == 0).astype(int)

    available = [f for f in priority_features if f in raw.columns]
    return raw[available + ["diagnosis"]].copy()


@pytest.fixture(scope="session")
def df(priority_features):
    """
    Load the WDBC dataset.  Uses the project's data/data.csv when present
    (local dev & CI runs that include the data file), otherwise falls back to
    building a compatible DataFrame from sklearn's breast_cancer dataset.
    """
    data_path = os.path.join(PROJECT_ROOT, "data", "data.csv")

    if os.path.exists(data_path):
        raw = pd.read_csv(data_path)
        raw = raw.drop(columns=["id"], errors="ignore")
        if "diagnosis" in raw.columns:
            raw["diagnosis"] = raw["diagnosis"].apply(lambda x: 1 if x == "M" else 0)
        available = [f for f in priority_features if f in raw.columns]
        return raw[available + ["diagnosis"]].copy()

    return _build_fallback_df(priority_features)


@pytest.fixture(scope="session")
def wdbc_features(df, priority_features):
    """Priority features that actually exist in the loaded DataFrame."""
    return [f for f in priority_features if f in df.columns]


# ---------------------------------------------------------------------------
# ML objects
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def scaler(df, wdbc_features):
    X = df[wdbc_features]
    sc = StandardScaler()
    sc.fit(X)
    return sc


@pytest.fixture(scope="session")
def trained_model(df, wdbc_features, scaler):
    """Lightweight Random-Forest trained on the WDBC priority features."""
    X = scaler.transform(df[wdbc_features])
    y = df["diagnosis"]
    model = RandomForestClassifier(n_estimators=20, random_state=42, n_jobs=1)
    model.fit(X, y)
    return model


@pytest.fixture(scope="session")
def voting_model(df, wdbc_features, scaler):
    """Soft-voting ensemble mirroring the production model structure."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC

    X = scaler.transform(df[wdbc_features])
    y = df["diagnosis"]

    rf = RandomForestClassifier(n_estimators=10, random_state=0)
    lr = LogisticRegression(max_iter=1000, random_state=0)
    svc = SVC(probability=True, random_state=0)

    vc = VotingClassifier(estimators=[("rf", rf), ("lr", lr), ("svc", svc)], voting="soft")
    vc.fit(X, y)
    return vc


@pytest.fixture(scope="session")
def sample_input(df, wdbc_features):
    """Single-row DataFrame with mean values (the sidebar's default state)."""
    means = df[wdbc_features].mean()
    return pd.DataFrame([means], columns=wdbc_features)


@pytest.fixture(scope="session")
def malignant_sample(df, wdbc_features):
    """A confirmed malignant sample from the dataset."""
    row = df[df["diagnosis"] == 1].iloc[0]
    return pd.DataFrame([row[wdbc_features].values], columns=wdbc_features)


@pytest.fixture(scope="session")
def benign_sample(df, wdbc_features):
    """A confirmed benign sample from the dataset."""
    row = df[df["diagnosis"] == 0].iloc[0]
    return pd.DataFrame([row[wdbc_features].values], columns=wdbc_features)
