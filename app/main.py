import streamlit as st
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import StandardScaler

from utils.model_loader import load_model_and_scaler, get_explainer
from utils.data_loader import load_data
from utils.monitoring import log_prediction, calculate_drift, load_prediction_history
from utils.report_generator import generate_patient_report
from utils.clinical_insights import get_clinical_insights
from utils.counterfactuals import find_benign_counterfactual
from utils.robustness import calculate_model_robustness
from utils.error_analysis import get_error_analysis_data
from utils.sensitivity import calculate_sensitivity_curve
from utils.synthetic_data import generate_synthetic_samples, evaluate_generalization
from components.ui import inject_custom_css, render_hero, render_instructions, render_stats, render_footer
from components.sidebar import render_sidebar
from components.visualizations import (
    render_probability_chart, render_correlation_heatmap,
    render_shap_plot, render_drift_monitoring, render_history_charts,
    render_radar_chart, render_pca_plot, render_robustness_chart,
    render_confusion_matrix, render_error_distribution,
    render_sensitivity_plot, render_synthetic_pca_plot, render_roc_curve_explorer,
)
from components.ethics import render_ethics_disclaimer, render_ethics_content
from components.explorer import render_dataset_explorer

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Breast Cancer Diagnostic Support",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)


def _insight_severity_badge(severity: str) -> str:
    mapping = {
        'high':   '<span class="badge badge-error">High</span>',
        'medium': '<span class="badge badge-warning">Moderate</span>',
        'low':    '<span class="badge badge-success">Low</span>',
    }
    return mapping.get(severity, '<span class="badge badge-neutral">Info</span>')


def _insight_block(insight: dict) -> str:
    css = {'high': 'high', 'medium': 'medium', 'low': 'low'}.get(insight.get('severity', 'low'), 'low')
    badge = _insight_severity_badge(insight.get('severity', 'low'))
    return (
        f'<div class="insight-row {css}">'
        f'<strong>{insight["feature"]} {badge}</strong>'
        f'{insight["insight"]}'
        f'</div>'
    )


def main():
    inject_custom_css()
    render_ethics_disclaimer()

    # ── Load assets ────────────────────────────────────────────────────────────
    model, loaded_scaler, saved_feature_names = load_model_and_scaler()
    df = load_data()

    PRIORITY_FEATURES = [
        "radius_mean", "perimeter_mean", "area_mean", "compactness_mean",
        "concavity_mean", "concave_points_mean", "radius_se", "perimeter_se",
        "area_se", "radius_worst", "perimeter_worst", "area_worst",
        "compactness_worst", "concavity_worst", "concave_points_worst",
    ]

    if saved_feature_names:
        expected_features = saved_feature_names
    elif loaded_scaler and hasattr(loaded_scaler, "feature_names_in_"):
        expected_features = loaded_scaler.feature_names_in_.tolist()
    else:
        cols = df.columns.tolist()
        expected_features = (
            PRIORITY_FEATURES if all(f in cols for f in PRIORITY_FEATURES)
            else [c for c in cols if c != "diagnosis"]
        )

    X = df[expected_features]
    scaler = loaded_scaler if loaded_scaler else StandardScaler().fit(X)

    # ── Session state ──────────────────────────────────────────────────────────
    for key in ('analysis_results', 'pdf_report', 'last_input_hash'):
        if key not in st.session_state:
            st.session_state[key] = None

    # ── Sidebar ────────────────────────────────────────────────────────────────
    user_input, decision_threshold = render_sidebar(df, expected_features)

    input_hash = hash(str(user_input.to_dict()))
    if st.session_state.last_input_hash != input_hash:
        st.session_state.analysis_results = None
        st.session_state.pdf_report = None
        st.session_state.last_input_hash = input_hash

    # ── Tab navigation ─────────────────────────────────────────────────────────
    tabs = st.tabs([
        "AI Analysis",
        "Model Monitoring",
        "Model Card & Ethics",
        "Error Analysis",
        "Research Lab",
        "Dataset Explorer",
    ])
    tab_diag, tab_monitor, tab_ethics, tab_error, tab_lab, tab_explore = tabs

    # ══════════════════════════════════════════════════════════════════════════
    #  TAB 1 — AI Analysis
    # ══════════════════════════════════════════════════════════════════════════
    with tab_diag:
        render_hero()
        render_instructions()

        # ── Input Summary ──────────────────────────────────────────────────────
        st.markdown('<div class="section-label">Patient Input Summary</div>', unsafe_allow_html=True)
        with st.container(border=True):
            st.markdown(
                '<div class="card-header">Tumor Measurements (Current Configuration)</div>',
                unsafe_allow_html=True,
            )
            st.dataframe(user_input.style.format(precision=3), use_container_width=True)

        # ── Run Analysis ───────────────────────────────────────────────────────
        st.markdown('<div class="section-label">AI Analysis</div>', unsafe_allow_html=True)
        col_b1, col_b2, col_b3 = st.columns([1, 2, 1])
        with col_b2:
            run_clicked = st.button("Run Analysis", key="predict_btn", use_container_width=True)

        if run_clicked:
            with st.spinner("Processing measurements through ensemble model..."):
                aligned = user_input[expected_features]
                scaled = scaler.transform(aligned)

                proba = model.predict_proba(scaled)[0]
                result = "Malignant" if proba[1] >= decision_threshold else "Benign"
                confidence = max(proba)

                X_scaled = scaler.transform(X)
                explainer = get_explainer(model, X_scaled)
                shap_vals = None
                if explainer:
                    sv = explainer.shap_values(scaled)
                    shap_vals = (
                        np.array(sv[1]).flatten()
                        if isinstance(sv, list)
                        else np.array(sv).flatten()
                    )

                insights = get_clinical_insights(user_input)
                robustness = calculate_model_robustness(model, scaled)
                log_prediction(user_input[expected_features], result, confidence)

                st.session_state.analysis_results = {
                    'result': result, 'confidence': confidence,
                    'proba': proba, 'shap': shap_vals,
                    'insights': insights, 'robustness': robustness,
                    'threshold': decision_threshold,
                }
                st.session_state.pdf_report = None

        # ── Results ────────────────────────────────────────────────────────────
        if st.session_state.analysis_results:
            res = st.session_state.analysis_results
            result = res['result']
            css_cls = 'malignant' if result == 'Malignant' else 'benign'
            card_cls = 'malignant-card' if result == 'Malignant' else 'benign-card'

            col_r1, col_r2, col_r3 = st.columns([1, 2, 1])
            with col_r2:

                # Primary classification result
                st.markdown(f"""
                    <div class="result-card {card_cls}">
                        <div class="result-card-label">Classification Result</div>
                        <div class="result-value {css_cls}">{result}</div>
                        <div class="result-confidence">
                            Model confidence: <strong>{res['confidence']:.1%}</strong>
                            &nbsp;&middot;&nbsp;
                            Threshold: <strong>T = {res['threshold']:.2f}</strong>
                        </div>
                    </div>
                """, unsafe_allow_html=True)

                # Probability cells
                st.markdown(f"""
                    <div class="prob-row">
                        <div class="prob-cell malignant">
                            <div class="prob-cell-label">Malignant Risk</div>
                            <div class="prob-cell-value">{res['proba'][1]:.1%}</div>
                        </div>
                        <div class="prob-cell benign">
                            <div class="prob-cell-label">Benign Probability</div>
                            <div class="prob-cell-value">{res['proba'][0]:.1%}</div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)

                # Robustness summary
                rob = res['robustness']
                st.markdown(f"""
                    <div class="robustness-card">
                        <div class="robustness-title">Diagnostic Robustness</div>
                        <div class="robustness-row">
                            <span class="robustness-level">{rob['stability_level']}</span>
                            <span class="robustness-score">Ensemble Agreement: {rob['stability_score']:.0f}%</span>  # noqa: E501
                        </div>
                        <div class="robustness-ci">95% CI&nbsp; [{rob['ci'][0]:.1%} &mdash; {rob['ci'][1]:.1%}]</div>  # noqa: E501
                    </div>
                """, unsafe_allow_html=True)

                st.markdown(
                    '<div class="section-label">Probability Distribution</div>',
                    unsafe_allow_html=True,
                )
                with st.container(border=True):
                    render_probability_chart(res['proba'])

                st.markdown(
                    '<div class="section-label">Ensemble Variance</div>',
                    unsafe_allow_html=True,
                )
                with st.container(border=True):
                    render_robustness_chart(rob)

                # SHAP
                if res['shap'] is not None:
                    st.markdown(
                        '<div class="section-label">Explainability (SHAP)</div>',
                        unsafe_allow_html=True,
                    )
                    with st.container(border=True):
                        render_shap_plot(res['shap'], expected_features)

                # Radar + PCA
                st.markdown('<div class="section-label">Spatial Analysis</div>', unsafe_allow_html=True)
                col_v1, col_v2 = st.columns(2)
                with col_v1:
                    with st.container(border=True):
                        render_radar_chart(user_input, df, expected_features)
                with col_v2:
                    with st.container(border=True):
                        render_pca_plot(user_input, df, expected_features)

                # Clinical Insights
                st.markdown(
                    '<div class="section-label">Clinical Insights (Rule-Based)</div>',
                    unsafe_allow_html=True,
                )
                with st.container(border=True):
                    if res['insights']:
                        st.markdown(
                            "".join(_insight_block(i) for i in res['insights']),
                            unsafe_allow_html=True,
                        )
                    else:
                        st.info("No notable clinical flags for the current configuration.")

                # What-If Analysis
                if result == "Malignant":
                    st.markdown(
                        '<div class="section-label">What-If Analysis</div>',
                        unsafe_allow_html=True,
                    )
                    with st.expander("Explore the decision boundary — required changes for Benign classification"):  # noqa: E501
                        st.info(
                            "This simulation finds the minimum adjustments needed to cross "
                            "the statistical Benign classification boundary."
                        )
                        if st.button("Calculate Benign Boundary", key="cf_btn", use_container_width=True):
                            with st.spinner("Searching decision boundary..."):
                                cf_data, error = find_benign_counterfactual(
                                    model, scaler, user_input, expected_features, df
                                )
                                if error:
                                    st.warning(error)
                                else:
                                    st.success(f"Boundary found in {cf_data['iterations']} optimisation iterations.")  # noqa: E501
                                    chg_df = pd.DataFrame(cf_data['changes'])
                                    st.dataframe(
                                        chg_df[['feature', 'from', 'to', 'change']].style.format({
                                            'from': '{:.3f}', 'to': '{:.3f}', 'change': '{:+.3f}'
                                        }),
                                        use_container_width=True, hide_index=True,
                                    )
                                    st.caption("Features not listed remained within 1% of their original value.")  # noqa: E501

                # Sensitivity Analysis
                st.markdown(
                    '<div class="section-label">Feature Sensitivity Analysis</div>',
                    unsafe_allow_html=True,
                )
                with st.container(border=True):
                    st.markdown(
                        '<div class="card-header">Sensitivity Curve</div>',
                        unsafe_allow_html=True,
                    )
                    def_idx = (
                        expected_features.index("concave_points_mean")
                        if "concave_points_mean" in expected_features else 0
                    )
                    sel_feat = st.selectbox(
                        "Select feature",
                        options=expected_features, index=def_idx,
                        help="Shows malignancy probability as this feature varies across its clinical range.",
                    )
                    curve = calculate_sensitivity_curve(
                        model, scaler, user_input, sel_feat, expected_features, df
                    )
                    render_sensitivity_plot(curve, sel_feat)

                # Report
                st.markdown('<div class="section-label">Clinical Report</div>', unsafe_allow_html=True)
                with st.container(border=True):
                    st.markdown('<div class="card-header">PDF Export</div>', unsafe_allow_html=True)
                    if st.session_state.pdf_report is None:
                        if st.button("Prepare Clinical Report", key="prep_report", use_container_width=True):
                            with st.spinner("Compiling report..."):
                                try:
                                    pdf = generate_patient_report(
                                        result, res['confidence'], res['proba'],
                                        expected_features, user_input, res['shap'],
                                        res['insights'], res['threshold'],
                                    )
                                    st.session_state.pdf_report = pdf
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"Report generation failed: {e}")
                    else:
                        col_dl, col_disc = st.columns([3, 1])
                        with col_dl:
                            st.download_button(
                                label="Download Clinical Risk Report (PDF)",
                                data=st.session_state.pdf_report,
                                file_name=f"CancerDiagnostics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                                mime="application/pdf",
                                key="dl_pdf",
                                use_container_width=True,
                            )
                        with col_disc:
                            if st.button("Discard", key="discard_report", use_container_width=True):
                                st.session_state.pdf_report = None
                                st.rerun()

    # ══════════════════════════════════════════════════════════════════════════
    #  TAB 2 — Model Monitoring
    # ══════════════════════════════════════════════════════════════════════════
    with tab_monitor:
        st.markdown('<div class="section-heading">Operational Monitoring</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-subtext">'  # noqa: E501
            'Real-time data drift analysis and prediction history tracking.</div>',
            unsafe_allow_html=True,
        )

        # ── Drift ──────────────────────────────────────────────────────────────
        st.markdown('<div class="section-label">Data Drift Analysis</div>', unsafe_allow_html=True)

        if st.session_state.analysis_results:
            drift_data = calculate_drift(user_input[expected_features], df, expected_features)

            # Key metric cards
            d_col1, d_col2, d_col3 = st.columns(3)
            drift_count = sum(1 for m in drift_data['metrics'].values() if m['status'] == 'Drifting')
            d_col1.metric('Health Score', f"{drift_data['health_score']:.0f}/100",
                          help='Composite score: 100 = fully within training distribution.')
            d_col2.metric('Features Drifting', drift_count,
                          delta=f"{'Above threshold' if drift_count > 0 else 'None'}",
                          delta_color='inverse')
            d_col3.metric('Drift Status',
                          'Detected' if drift_data['is_drifting'] else 'Stable',
                          help='Drift is flagged when z-score > 3 on any feature.')

            with st.container(border=True):
                render_drift_monitoring(drift_data)
        else:
            st.info("Run an analysis in the AI Analysis tab to enable real-time drift monitoring.")

        # ── History ────────────────────────────────────────────────────────────
        st.markdown('<div class="section-label">Prediction History</div>', unsafe_allow_html=True)
        history_df = load_prediction_history()

        if not history_df.empty:
            h_col1, h_col2, h_col3 = st.columns(3)
            total_preds = len(history_df)
            mal_preds = (history_df['prediction'] == 'Malignant').sum()
            ben_preds = total_preds - mal_preds

            h_col1.metric('Total Predictions', total_preds)
            h_col2.metric('Malignant', mal_preds)
            h_col3.metric('Benign', ben_preds)

            with st.container(border=True):
                render_history_charts(history_df)

            with st.expander("Raw Prediction Log"):
                st.dataframe(
                    history_df.sort_values('timestamp', ascending=False),
                    use_container_width=True, hide_index=True,
                )
        else:
            with st.container(border=True):
                render_history_charts(history_df)

    # ══════════════════════════════════════════════════════════════════════════
    #  TAB 3 — Model Card & Ethics
    # ══════════════════════════════════════════════════════════════════════════
    with tab_ethics:
        render_ethics_content(df)

    # ══════════════════════════════════════════════════════════════════════════
    #  TAB 4 — Error Analysis
    # ══════════════════════════════════════════════════════════════════════════
    with tab_error:
        st.markdown('<div class="section-heading">Model Error Analysis</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-subtext">Evaluate misclassification behaviour across the full '
            'WDBC dataset at the current decision threshold.</div>',
            unsafe_allow_html=True,
        )

        error_data = get_error_analysis_data(model, scaler, df, expected_features, decision_threshold)

        # Key metric cards
        e_col1, e_col2, e_col3, e_col4 = st.columns(4)
        e_col1.metric('Accuracy', f"{error_data['accuracy']:.1%}")
        e_col2.metric('Threshold', f"T = {decision_threshold:.2f}")
        e_col3.metric('Total Errors', error_data['total_errors'])
        misclassified = error_data['misclassified']
        fp = (
            len(misclassified[misclassified['Error_Type'] == 'False Positive'])
            if not misclassified.empty else 0
        )
        fn = error_data['total_errors'] - fp
        e_col4.metric('FN / FP', f"{fn} / {fp}",
                      help='False Negatives (missed malignant) / False Positives (benign flagged as malignant)')  # noqa: E501

        # ROC Curve
        st.markdown('<div class="section-label">ROC Curve</div>', unsafe_allow_html=True)
        with st.container(border=True):
            render_roc_curve_explorer(error_data['roc_data'], decision_threshold)

        # Confusion Matrix + Error Distribution
        st.markdown('<div class="section-label">Error Breakdown</div>', unsafe_allow_html=True)
        col_cm, col_ed = st.columns(2)
        with col_cm:
            with st.container(border=True):
                st.markdown('<div class="card-header">Confusion Matrix</div>', unsafe_allow_html=True)
                render_confusion_matrix(error_data['cm'])
        with col_ed:
            with st.container(border=True):
                st.markdown('<div class="card-header">Error Probability Distribution</div>', unsafe_allow_html=True)  # noqa: E501
                render_error_distribution(error_data['misclassified'])

        # Misclassification Log
        st.markdown('<div class="section-label">Misclassification Log</div>', unsafe_allow_html=True)
        with st.container(border=True):
            if not error_data['misclassified'].empty:
                st.dataframe(error_data['misclassified'], use_container_width=True, hide_index=True)
            else:
                st.success("No misclassifications found at the current threshold.")

    # ══════════════════════════════════════════════════════════════════════════
    #  TAB 5 — Research Lab
    # ══════════════════════════════════════════════════════════════════════════
    with tab_lab:
        st.markdown(
            '<div class="section-heading">Research Lab: Synthetic Stress Testing</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="section-subtext">Evaluate model generalisation by simulating '
            'clinical edge cases via Gaussian multivariate sampling.</div>',
            unsafe_allow_html=True,
        )

        lab_left, lab_right = st.columns([1, 2])

        with lab_left:
            with st.container(border=True):
                st.markdown('<div class="card-header">Simulation Parameters</div>', unsafe_allow_html=True)

                n_gen = st.slider(
                    "Samples to generate", min_value=5, max_value=50, value=10,
                    help="Number of synthetic patient profiles to create per run.",
                )
                target_class = st.selectbox(
                    "Target profile",
                    options=["Mixed Distribution", "Benign Profile", "Malignant Profile"],
                    help="Constrain synthetic samples toward a specific clinical distribution.",
                )
                target_val = None
                if target_class == "Benign Profile":
                    target_val = 0
                if target_class == "Malignant Profile":
                    target_val = 1

                run_synth = st.button("Run Stress Test", key="synth_btn", use_container_width=True)

            if run_synth:
                with st.spinner("Generating synthetic samples..."):
                    st.session_state.synthetic_samples = generate_synthetic_samples(
                        df, expected_features, n_gen, target_val
                    )
                    st.session_state.synthetic_results = evaluate_generalization(
                        model, scaler, st.session_state.synthetic_samples, expected_features
                    )
                    st.success(f"Generated {n_gen} synthetic samples successfully.")

            # Show summary if available
            if 'synthetic_results' in st.session_state:
                sr = st.session_state.synthetic_results
                if 'Predicted_Class' in sr.columns:
                    n_mal = (sr['Predicted_Class'] == 'Malignant').sum()
                    n_ben = (sr['Predicted_Class'] == 'Benign').sum()
                    st.markdown('<hr class="card-divider">', unsafe_allow_html=True)
                    m1, m2 = st.columns(2)
                    m1.metric('Predicted Malignant', n_mal)
                    m2.metric('Predicted Benign', n_ben)

        with lab_right:
            if 'synthetic_results' in st.session_state:
                with st.container(border=True):
                    st.markdown('<div class="card-header">PCA Manifold Projection</div>', unsafe_allow_html=True)  # noqa: E501
                    render_synthetic_pca_plot(df, st.session_state.synthetic_results, expected_features)

        if 'synthetic_results' in st.session_state:
            st.markdown('<div class="section-label">Synthetic Sample Batch Results</div>', unsafe_allow_html=True)  # noqa: E501
            with st.container(border=True):
                st.dataframe(st.session_state.synthetic_results, use_container_width=True, hide_index=True)

    # ══════════════════════════════════════════════════════════════════════════
    #  TAB 6 — Dataset Explorer
    # ══════════════════════════════════════════════════════════════════════════
    with tab_explore:
        render_dataset_explorer(df, expected_features)

        st.markdown('<div class="section-label" style="margin-top:24px;">Feature Correlation</div>', unsafe_allow_html=True)  # noqa: E501
        with st.container(border=True):
            render_correlation_heatmap(df, expected_features)

        st.markdown('<div class="section-label">Dataset Statistics</div>', unsafe_allow_html=True)
        render_stats(df, expected_features)

    # ── Footer ─────────────────────────────────────────────────────────────────
    render_footer()


if __name__ == "__main__":
    main()
