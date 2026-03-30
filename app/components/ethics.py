import streamlit as st
import plotly.graph_objects as go

def render_ethics_disclaimer():
    """Renders a prominent, compact medical disclaimer."""
    st.warning(
        "**Medical Disclaimer:** This application is for educational and research purposes only. "
        "It is not intended to provide medical advice, diagnosis, or treatment. "
        "Always consult a qualified physician or healthcare provider."
    )


def render_ethics_content(df):
    """Document-style Model Card with numbered sections and dividers."""

    st.markdown('<div class="section-heading">Model Card &mdash; Breast Cancer Diagnostic System (v2.2)</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtext">A structured accountability document for the AI model deployed in this application.</div>', unsafe_allow_html=True)

    # Identity card
    st.markdown("""
        <div class="card-muted" style="margin-top:12px;">
            <div style="display:flex;gap:12px;flex-wrap:wrap;">
                <span class="badge badge-info">Random Forest Classifier</span>
                <span class="badge badge-neutral">Binary Classification</span>
                <span class="badge badge-neutral">Scikit-Learn v1.x</span>
                <span class="badge badge-neutral">v2.2 Clinical Control + XAI</span>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # Section 1
    st.markdown('<div class="section-label">1. Intended Use</div>', unsafe_allow_html=True)
    with st.container(border=True):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Intended Audience**")
            st.markdown("""
            - Medical researchers and data scientists
            - Clinical informatics students
            - Healthcare AI educators
            """)
        with col2:
            st.markdown("**Out of Scope**")
            st.markdown("""
            - Primary clinical diagnosis without specialist oversight
            - Surgical or treatment planning
            - Environments requiring CE / FDA certification
            """)

    # Section 2 — Dataset
    st.markdown('<div class="section-label">2. Dataset & Factors</div>', unsafe_allow_html=True)
    col_d1, col_d2 = st.columns([3, 2])

    with col_d1:
        with st.container(border=True):
            st.markdown("**Wisconsin Diagnostic Breast Cancer (WDBC)**")
            st.markdown("""
            - **Source:** University of Wisconsin–Madison
            - **Provenance:** Wolberg, Street & Mangasarian (1995)
            - **Size:** 569 instances, 30 numerical features
            - **Features:** Extracted via Fine Needle Aspiration (FNA) cytology
            - **Note:** Reflects a specific Wisconsin-based population.
              Generalizability across demographics is undocumented.
            """)

    with col_d2:
        counts = df['diagnosis'].value_counts()
        malignant_pct = (counts.get(1, 0) / len(df)) * 100

        fig = go.Figure(go.Indicator(
            mode='gauge+number',
            value=malignant_pct,
            title={'text': 'Malignant Representation (%)', 'font': {'size': 12, 'color': '#0f172a'}},
            number={'font': {'color': '#0f172a', 'size': 26}, 'suffix': '%'},
            gauge={
                'axis': {'range': [0, 100], 'tickcolor': '#94a3b8'},
                'bar': {'color': '#dc2626'},
                'bgcolor': '#f8fafc',
                'borderwidth': 0,
                'steps': [{'range': [0, 100], 'color': '#f1f5f9'}],
                'threshold': {'line': {'color': '#475569', 'width': 2}, 'value': 37.3},
            },
        ))
        fig.update_layout(
            paper_bgcolor='#ffffff', height=200,
            font=dict(family='Inter, sans-serif', color='#0f172a'),
            margin=dict(l=10, r=10, t=40, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)

    # Section 3 — Performance
    st.markdown('<div class="section-label">3. Performance & Evaluation</div>', unsafe_allow_html=True)
    with st.container(border=True):
        c1, c2, c3 = st.columns(3)
        c1.metric('Binary Accuracy', '~96%', help='Held-out 20% test set')
        c2.metric('Primary Metric', 'AUC-ROC', help='Area under the Receiver Operating Characteristic curve')
        c3.metric('Robustness', 'Ensemble Variance', help='Measured via tree-level prediction spread; see Model Monitoring tab')

    # Section 4 — Ethics
    st.markdown('<div class="section-label">4. Ethical Considerations</div>', unsafe_allow_html=True)
    with st.container(border=True):
        col_e1, col_e2 = st.columns(2)
        with col_e1:
            st.markdown("**Transparency**")
            st.markdown(
                "Uses both local (SHAP) and global (PCA, ROC) interpretability methods "
                "to explain individual predictions and overall model behavior."
            )
            st.markdown("**Accountability**")
            st.markdown(
                "User-controlled decision threshold allows clinical teams to adjust "
                "sensitivity vs. specificity based on operational requirements."
            )
        with col_e2:
            st.markdown("**Data Privacy**")
            st.markdown(
                "Fully local execution — patient measurements are processed "
                "in-memory and are not persisted to any external storage or cloud service."
            )
            st.markdown("**Fairness**")
            st.markdown(
                "Dataset demographic scope is limited. Independent validation on "
                "diverse populations is required before any clinical deployment."
            )

    # Section 5 — Limitations
    st.markdown('<div class="section-label">5. Technical Limitations</div>', unsafe_allow_html=True)
    with st.container(border=True):
        limitations = [
            ("Digital Biopsy Only", "Predictions rely on numeric FNA measurements. Genetic, genomic, and lifestyle risk factors are not included."),
            ("Class Imbalance", "Dataset contains 357 Benign vs. 212 Malignant samples. This may bias the model toward higher specificity."),
            ("Time-Invariant", "The model does not account for longitudinal tumor growth rates or treatment history."),
            ("Distribution Shift", "Performance may degrade on samples from populations or imaging equipment not represented in the training data."),
        ]
        for title, body in limitations:
            st.markdown(f"**{title}** — {body}")

    st.markdown('<div class="section-label">6. Version History</div>', unsafe_allow_html=True)
    with st.container(border=True):
        st.markdown("""
        | Version | Change |
        |---------|--------|
        | v2.2 | Clinical threshold control, SHAP explainability, PDF reporting |
        | v2.0 | Ensemble model, drift monitoring, error analysis |
        | v1.0 | Initial Random Forest baseline |
        """)
