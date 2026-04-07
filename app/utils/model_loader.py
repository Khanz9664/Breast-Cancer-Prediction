import streamlit as st
import joblib
import os


@st.cache_resource
def load_model_and_scaler():
    """Load the trained model and scaler from the models directory."""
    model_path = os.path.join("models", "breast_cancer_model_v2.pkl")
    try:
        bundle = joblib.load(model_path)
        if isinstance(bundle, dict):
            return bundle["model"], bundle.get("scaler", None), bundle.get("feature_names", None)
        return bundle, None, None
    except Exception:
        st.error(f"Error loading model from {model_path}. Please ensure the file exists.")
        st.stop()


@st.cache_resource
def get_explainer(_model, _background_data):
    """
    Initialize a SHAP explainer for the model.
    Using KernelExplainer for VotingClassifier compatibility.
    """
    import shap
    try:
        # Use a small representative sample of background data to speed up KernelExplainer
        background_summary = (
            shap.kmeans(_background_data, 5)
            if len(_background_data) > 5
            else _background_data
        )
        explainer = shap.KernelExplainer(_model.predict_proba, background_summary)
        return explainer
    except Exception:
        st.warning("Could not initialize SHAP explainer. Check model compatibility.")
        return None
