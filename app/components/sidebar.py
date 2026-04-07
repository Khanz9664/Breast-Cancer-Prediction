import streamlit as st
import pandas as pd

# ── Clinical tooltips per feature ──────────────────────────────────────────────
FEATURE_TOOLTIPS = {
    "radius_mean":           "Mean of distances from center to points on the nucleus perimeter (micrometers). Larger values correlate with malignancy.",  # noqa: E501
    "perimeter_mean":        "Mean perimeter of the cell nucleus. Closely correlated with radius and area.",
    "area_mean":             "Mean cross-sectional area of cell nuclei (sq. micrometers).",
    "compactness_mean":      "Mean compactness = (perimeter² / area − 1.0). Values above dataset mean suggest irregular morphology.",  # noqa: E501
    "concavity_mean":        "Mean severity of concave contour sections on the cell nucleus.",
    "concave_points_mean":   "Mean count of concave contour points. One of the strongest malignancy predictors in this dataset.",  # noqa: E501
    "radius_se":             "Standard error of nuclear radius across sampled cells — indicates measurement variability.",  # noqa: E501
    "perimeter_se":          "Standard error of the nuclear perimeter across cells.",
    "area_se":               "Standard error of nuclear area — high values suggest heterogeneous cell population.",  # noqa: E501
    "radius_worst":          "Mean radius of the three largest nuclei in the sample.",
    "perimeter_worst":       "Mean perimeter of the three largest nuclei.",
    "area_worst":            "Mean area of the three largest nuclei — captures the most abnormal cells.",
    "compactness_worst":     "Worst-case compactness across the three largest nuclei.",
    "concavity_worst":       "Worst-case concavity across the three largest nuclei.",
    "concave_points_worst":  "Worst-case concave contour count across the three largest nuclei.",
}

PRIORITY_FEATURES = [
    "radius_mean", "perimeter_mean", "area_mean", "compactness_mean",
    "concavity_mean", "concave_points_mean", "radius_se", "perimeter_se",
    "area_se", "radius_worst", "perimeter_worst", "area_worst",
    "compactness_worst", "concavity_worst", "concave_points_worst",
]

CATEGORIES = [
    ("Mean Measurements", lambda f: f.endswith('_mean')),
    ("Standard Error", lambda f: f.endswith('_se')),
    ("Worst-Case Measurements", lambda f: f.endswith('_worst')),
]


def render_sidebar(df, expected_features):
    """Render the Clinical Configuration sidebar. Returns (user_input_df, threshold)."""

    # ── Panel header ──────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("""
            <div class="sidebar-inner">
                <div class="sidebar-panel-label">Breast Cancer DSS</div>
                <div class="sidebar-panel-title">Clinical Configuration</div>
            </div>
        """, unsafe_allow_html=True)

        # ── Classification Policy ─────────────────────────────────────────────
        st.markdown(
            '<div class="sidebar-section-label" style="margin:6px 16px 0 16px;">Classification Policy</div>',
            unsafe_allow_html=True,
        )
        with st.container():
            threshold = st.slider(
                label="Malignancy Threshold",
                min_value=0.20,
                max_value=0.80,
                value=0.50,
                step=0.05,
                help=(
                    "The model classifies a sample as Malignant when predicted "
                    "probability ≥ this value.\n\n"
                    "• **< 0.45** — High sensitivity (minimize false negatives)\n"
                    "• **0.45 – 0.55** — Balanced performance\n"
                    "• **> 0.55** — High specificity (minimize false positives)"
                ),
            )

            # Contextual mode badge
            if threshold < 0.45:
                mode_badge = '<span class="badge badge-error">High Sensitivity</span>'
                mode_note = "Fewer false negatives — favors detecting malignancy."
            elif threshold > 0.55:
                mode_badge = '<span class="badge badge-info">High Specificity</span>'
                mode_note = "Fewer false positives — reduces unnecessary alerts."
            else:
                mode_badge = '<span class="badge badge-neutral">Balanced</span>'
                mode_note = "Equal weight to sensitivity and specificity."

            st.markdown(
                f'<div style="margin:2px 0 12px 0;">{mode_badge}</div>'
                f'<div style="font-size:12px;color:#64748b;margin-bottom:4px;">{mode_note}</div>',
                unsafe_allow_html=True,
            )

        st.markdown('<hr style="border:none;border-top:1px solid #e2e8f0;margin:4px 0 12px 0;">', unsafe_allow_html=True)  # noqa: E501

        # ── Tumor Characteristics ─────────────────────────────────────────────
        st.markdown(
            '<div class="sidebar-section-label" style="margin:0 16px 8px 16px;">Tumor Characteristics</div>',
            unsafe_allow_html=True,
        )

        valid_features = [f for f in expected_features if f in PRIORITY_FEATURES]
        if not valid_features:
            valid_features = expected_features

        X = df[valid_features]
        input_dict = {}

        for cat_name, predicate in CATEGORIES:
            features_in_cat = [f for f in valid_features if predicate(f)]
            if not features_in_cat:
                continue

            with st.expander(cat_name, expanded=(cat_name == "Mean Measurements")):
                for feature in features_in_cat:
                    col_label = (
                        feature
                        .replace('_mean', '')
                        .replace('_se', '')
                        .replace('_worst', '')
                        .replace('_', ' ')
                        .title()
                    )
                    step = float((X[feature].max() - X[feature].min()) / 200)
                    input_dict[feature] = st.slider(
                        label=col_label,
                        min_value=float(X[feature].min()),
                        max_value=float(X[feature].max()),
                        value=float(X[feature].mean()),
                        step=max(step, 1e-6),
                        help=FEATURE_TOOLTIPS.get(feature, f"Measurement value for {col_label}."),
                        key=f"sidebar_{feature}",
                    )

        # Reset note
        st.markdown(
            '<div style="font-size:12px;color:#94a3b8;text-align:center;'
            'border-top:1px solid #e2e8f0;padding-top:12px;margin-top:8px;">'
            'Default values = population mean</div>',
            unsafe_allow_html=True,
        )

    # Fill any missing expected features with dataset mean
    for f in expected_features:
        if f not in input_dict:
            input_dict[f] = float(df[f].mean())

    return pd.DataFrame(input_dict, index=[0]), threshold
