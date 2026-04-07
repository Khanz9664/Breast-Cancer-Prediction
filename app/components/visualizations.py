import pandas as pd
import streamlit as st
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from sklearn.decomposition import PCA

# ── Design token Plotly defaults ──────────────────────────────────────────────
_FONT = dict(family='Inter, -apple-system, sans-serif', color='#0f172a', size=12)
_TITLE_FONT = dict(family='Inter, sans-serif', color='#0f172a', size=14, weight=600) if False else dict(family='Inter, sans-serif', color='#0f172a', size=14)  # noqa: E501

CLINICAL_LAYOUT = dict(
    paper_bgcolor='#ffffff',
    plot_bgcolor='#f8fafc',
    font=_FONT,
    title_font=dict(family='Inter, sans-serif', color='#0f172a', size=14),
    legend=dict(bgcolor='rgba(0,0,0,0)', borderwidth=0, font=dict(size=12, color='#475569')),
    margin=dict(l=16, r=16, t=44, b=16),
    hoverlabel=dict(bgcolor='#0f172a', font_color='#ffffff', font_size=12, bordercolor='#0f172a'),
)

# Design-token color palette
C_PRIMARY = '#3b82f6'
C_MALIGNANT = '#dc2626'
C_BENIGN = '#16a34a'
C_WARNING = '#f59e0b'
C_NEUTRAL = '#64748b'
C_GRID = '#f1f5f9'
C_AXIS = '#e2e8f0'


def _apply(fig, height=380, **extra):
    """Apply clinical layout to a Plotly figure and return it."""
    layout = {**CLINICAL_LAYOUT, 'height': height, **extra}
    fig.update_layout(**layout)
    fig.update_xaxes(
        gridcolor=C_GRID, linecolor=C_AXIS, zerolinecolor=C_AXIS,
        tickfont=dict(size=11, color='#64748b'),
    )
    fig.update_yaxes(
        gridcolor=C_GRID, linecolor=C_AXIS, zerolinecolor=C_AXIS,
        tickfont=dict(size=11, color='#64748b'),
    )
    return fig


# ── Charts ────────────────────────────────────────────────────────────────────

def render_probability_chart(proba):
    fig = go.Figure(data=[go.Bar(
        x=['Benign', 'Malignant'],
        y=[proba[0], proba[1]],
        marker_color=[C_BENIGN, C_MALIGNANT],
        marker_line=dict(width=0),
        text=[f'{proba[0]:.1%}', f'{proba[1]:.1%}'],
        textposition='outside',
        textfont=dict(size=13, color='#0f172a', weight=600 if False else None),
    )])
    _apply(fig, height=300, title='Prediction Probability Distribution', showlegend=False)
    fig.update_yaxes(range=[0, 1.15], tickformat='.0%')
    st.plotly_chart(fig, use_container_width=True)


def render_correlation_heatmap(df, features):
    st.markdown('<div class="section-label">Feature Correlation Matrix</div>', unsafe_allow_html=True)
    corr = df[list(features)].corr()
    fig = go.Figure(go.Heatmap(
        z=corr.values,
        x=[c.replace('_', ' ').title() for c in corr.columns],
        y=[c.replace('_', ' ').title() for c in corr.columns],
        colorscale='RdBu',
        zmid=0,
        text=corr.values,
        texttemplate='%{text:.2f}',
        textfont=dict(size=9),
        hoverongaps=False,
    ))
    _apply(fig, height=640, plot_bgcolor='#ffffff',
           title='Feature Correlation Matrix (Pearson)')
    st.plotly_chart(fig, use_container_width=True)


def render_shap_plot(shap_values, feature_names):
    shap_flat = np.array(shap_values).flatten()
    n = min(len(feature_names), len(shap_flat))
    feature_names, shap_flat = feature_names[:n], shap_flat[:n]

    df_shap = pd.DataFrame({'Feature': [f.replace('_', ' ').title() for f in feature_names],
                            'Contribution': shap_flat})
    df_shap['Abs'] = df_shap['Contribution'].abs()
    df_shap = df_shap.sort_values('Abs', ascending=True).tail(10)
    df_shap['Color'] = df_shap['Contribution'].apply(lambda x: C_MALIGNANT if x > 0 else C_BENIGN)

    fig = go.Figure(go.Bar(
        x=df_shap['Contribution'], y=df_shap['Feature'],
        orientation='h', marker_color=df_shap['Color'],
        marker_line=dict(width=0),
    ))
    _apply(fig, height=380,
           title='Top 10 Feature Contributions (SHAP)',
           xaxis_title='SHAP Value  (negative = Benign  |  positive = Malignant)',
           yaxis_title='')
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        'Red bars indicate features driving toward Malignant; '
        'green bars support a Benign classification.'
    )


def render_drift_monitoring(drift_data):
    col1, col2 = st.columns([1, 2])
    with col1:
        fig = go.Figure(go.Indicator(
            mode='gauge+number',
            value=drift_data['health_score'],
            title={'text': 'System Health Score', 'font': {'size': 13, 'color': '#0f172a'}},
            number={'font': {'color': '#0f172a', 'size': 30}},
            gauge={
                'axis': {'range': [0, 100], 'tickcolor': '#94a3b8'},
                'bar': {'color': C_PRIMARY},
                'bgcolor': '#f8fafc',
                'borderwidth': 0,
                'steps': [
                    {'range': [0, 50],   'color': '#fee2e2'},
                    {'range': [50, 80],  'color': '#fef3c7'},
                    {'range': [80, 100], 'color': '#dcfce7'},
                ],
                'threshold': {'line': {'color': '#0f172a', 'width': 2}, 'value': 80},
            }
        ))
        fig.update_layout(
            paper_bgcolor='#ffffff', height=220,
            font=dict(family='Inter, sans-serif', color='#0f172a'),
            margin=dict(l=10, r=10, t=40, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        if drift_data['is_drifting']:
            st.warning(
                '**Data Drift Detected.** The current input deviates significantly from the '
                'training data distribution. Exercise additional caution when interpreting predictions.'
            )
        else:
            st.success(
                '**Data Stability Confirmed.** The current input falls within the expected '
                'statistical range of the training dataset.'
            )

        metrics = drift_data['metrics']
        drift_df = pd.DataFrame([
            {'Feature': f.replace('_', ' ').title(), 'Z-Score': round(m['z_score'], 3), 'Status': m['status']}
            for f, m in metrics.items()
        ]).sort_values('Z-Score', key=abs, ascending=False).head(5)
        st.dataframe(drift_df, use_container_width=True, hide_index=True)


def render_history_charts(history_df):
    if history_df.empty:
        st.info('No prediction history available. Run analyses to start tracking trends.')
        return

    st.markdown('<div class="section-heading">Prediction Trends</div>', unsafe_allow_html=True)
    history_df['timestamp'] = pd.to_datetime(history_df['timestamp'])
    daily = history_df.groupby(
        [history_df['timestamp'].dt.date, 'prediction']
    ).size().unstack(fill_value=0)

    fig = px.line(
        daily, title='Daily Prediction Frequency',
        color_discrete_map={'Malignant': C_MALIGNANT, 'Benign': C_BENIGN},
    )
    fig.update_traces(line_width=2)
    _apply(fig, height=320)
    st.plotly_chart(fig, use_container_width=True)


def render_radar_chart(user_input, df, features):
    radar_features = [f for f in features if 'mean' in f][:6]
    if not radar_features:
        st.info('Insufficient features for radar chart.')
        return

    norm_user, norm_avg = [], []
    for f in radar_features:
        mn, mx, mu = df[f].min(), df[f].max(), df[f].mean()
        d = (mx - mn) or 1
        norm_user.append((user_input[f].iloc[0] - mn) / d)
        norm_avg.append((mu - mn) / d)

    theta = [f.replace('_mean', '').replace('_', ' ').title() for f in radar_features]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=norm_user, theta=theta, fill='toself', name='Current Sample',
        line_color=C_PRIMARY, fillcolor='rgba(59,130,246,0.12)',
    ))
    fig.add_trace(go.Scatterpolar(
        r=norm_avg, theta=theta, fill='toself', name='Population Mean',
        line_color=C_NEUTRAL, fillcolor='rgba(100,116,139,0.06)', line_dash='dot',
    ))
    fig.update_layout(
        polar=dict(
            bgcolor='#f8fafc',
            radialaxis=dict(visible=True, range=[0, 1], gridcolor='#e2e8f0',
                            tickfont=dict(size=10, color='#64748b')),
            angularaxis=dict(gridcolor='#e2e8f0'),
        ),
        paper_bgcolor='#ffffff', height=400,
        font=dict(family='Inter, sans-serif', color='#0f172a'),
        title='Tumor Profile vs. Population Mean',
        title_font=dict(color='#0f172a', size=14),
        showlegend=True,
        margin=dict(l=20, r=20, t=44, b=20),
    )
    st.plotly_chart(fig, use_container_width=True)


def render_pca_plot(user_input, df, features):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(df[features])
    u_pca = pca.transform(user_input[features])

    pca_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
    pca_df['Diagnosis'] = df['diagnosis'].map({0: 'Benign', 1: 'Malignant'})

    fig = px.scatter(
        pca_df, x='PC1', y='PC2', color='Diagnosis',
        color_discrete_map={'Benign': C_BENIGN, 'Malignant': C_MALIGNANT},
        opacity=0.5, title='PCA Projection with Current Sample',
    )
    fig.add_trace(go.Scatter(
        x=[u_pca[0, 0]], y=[u_pca[0, 1]],
        mode='markers+text',
        marker=dict(size=14, color=C_PRIMARY, symbol='diamond',
                    line=dict(width=2, color='white')),
        name='Current Sample',
        text=['Current'], textposition='top center',
        textfont=dict(size=11, color=C_PRIMARY),
    ))
    _apply(fig, height=400)
    st.plotly_chart(fig, use_container_width=True)


def render_robustness_chart(rob_data):
    mean_val = np.mean(rob_data['all_tree_probs'])
    fig = px.histogram(
        x=rob_data['all_tree_probs'], nbins=20,
        labels={'x': 'Predicted Probability (Malignant)'},
        color_discrete_sequence=[C_PRIMARY],
        opacity=0.8, title='Ensemble Tree Agreement',
    )
    fig.add_vline(
        x=mean_val, line_dash='dash', line_color=C_MALIGNANT, line_width=2,
        annotation_text=f'Mean: {mean_val:.1%}',
        annotation_font_color=C_MALIGNANT,
        annotation_font_size=12,
    )
    _apply(fig, height=300, yaxis_title='Tree Count')
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)


def render_sensitivity_plot(curve_data, feature_name):
    display = feature_name.replace('_', ' ').title()
    fig = px.line(
        x=curve_data['x'], y=curve_data['y'],
        labels={'x': display, 'y': 'Malignancy Probability'},
        title=f'Sensitivity: {display}',
    )
    fig.update_traces(line_color=C_PRIMARY, line_width=2.5)
    fig.add_trace(go.Scatter(
        x=[curve_data['current_val']], y=[curve_data['current_prob']],
        mode='markers',
        marker=dict(size=11, color=C_MALIGNANT, symbol='circle',
                    line=dict(width=2, color='white')),
        name='Current Value',
    ))
    _apply(fig, height=300, yaxis_range=[0, 1],
           yaxis_tickformat='.0%')
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        f"Predicted risk as '{display}' varies across its full clinical range "
        '— all other features held constant at current values.'
    )


def render_synthetic_pca_plot(original_df, synthetic_df, features):
    pca = PCA(n_components=2)
    original_pca = pca.fit_transform(original_df[features])
    synth_pca = pca.transform(synthetic_df[features])

    fig = px.scatter(
        x=original_pca[:, 0], y=original_pca[:, 1],
        color=original_df['diagnosis'].map({0: 'Benign', 1: 'Malignant'}),
        color_discrete_map={'Benign': '#94a3b8', 'Malignant': '#cbd5e1'},
        opacity=0.25, labels={'x': 'PC1', 'y': 'PC2'},
        title='Synthetic Samples on Dataset Manifold',
    )
    fig.add_trace(go.Scatter(
        x=synth_pca[:, 0], y=synth_pca[:, 1],
        mode='markers',
        marker=dict(
            size=10, symbol='diamond',
            color=synthetic_df['Predicted_Class'].map({'Benign': C_BENIGN, 'Malignant': C_MALIGNANT}),
            line=dict(width=1, color='white'),
        ),
        name='Synthetic Samples',
    ))
    _apply(fig, height=400)
    st.plotly_chart(fig, use_container_width=True)


def render_roc_curve_explorer(roc_data, current_threshold):
    fpr, tpr, thresholds = roc_data['fpr'], roc_data['tpr'], roc_data['thresholds']
    idx = np.argmin(np.abs(thresholds - current_threshold))
    cur_fpr, cur_tpr = fpr[idx], tpr[idx]

    fig = px.area(
        x=fpr, y=tpr,
        title=f"Receiver Operating Characteristic  (AUC: {roc_data['auc']:.4f})",
        labels={'x': 'False Positive Rate (1 − Specificity)', 'y': 'True Positive Rate (Sensitivity)'},
    )
    fig.update_traces(fillcolor='rgba(59,130,246,0.10)', line_color=C_PRIMARY, line_width=2)
    fig.add_shape(type='line', line=dict(dash='dot', color='#94a3b8', width=1), x0=0, x1=1, y0=0, y1=1)
    fig.add_trace(go.Scatter(
        x=[cur_fpr], y=[cur_tpr],
        mode='markers+text',
        marker=dict(size=13, color=C_MALIGNANT, symbol='diamond', line=dict(width=2, color='white')),
        name='Operating Point',
        text=[f'T = {current_threshold:.2f}'], textposition='top center',
        textfont=dict(size=11, color=C_MALIGNANT),
    ))
    fig.add_hline(y=cur_tpr, line_dash='dot', line_color='#e2e8f0', line_width=1)
    fig.add_vline(x=cur_fpr, line_dash='dot', line_color='#e2e8f0', line_width=1)
    _apply(fig, height=440, xaxis_range=[0, 1], yaxis_range=[0, 1.05],
           xaxis_tickformat='.0%', yaxis_tickformat='.0%')
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    col1.metric('Sensitivity (Recall)', f'{cur_tpr:.1%}')
    col2.metric('Specificity', f'{1 - cur_fpr:.1%}')
    st.caption(
        'Adjust the Malignancy Threshold in Clinical Configuration to '
        'move the operating point along the ROC curve.'
    )


def render_confusion_matrix(cm):
    z = cm
    x = ['Predicted Benign', 'Predicted Malignant']
    y = ['Actual Benign',    'Actual Malignant']
    text = [[str(v) for v in row] for row in z]

    fig = ff.create_annotated_heatmap(
        z, x=x, y=y, annotation_text=text,
        colorscale=[[0, '#f0fdf4'], [0.5, '#bfdbfe'], [1, '#1e3a8a']],
        showscale=True,
    )
    fig.update_layout(
        paper_bgcolor='#ffffff', plot_bgcolor='#f8fafc',
        font=dict(family='Inter, sans-serif', color='#0f172a'),
        height=360, margin=dict(l=16, r=16, t=40, b=16),
    )
    st.plotly_chart(fig, use_container_width=True)


def render_error_distribution(misclassified):
    if misclassified.empty:
        st.info('No misclassifications found at the current threshold. Excellent model performance.')
        return

    fig = px.box(
        misclassified, x='Error_Type', y='Probability',
        color='Error_Type',
        color_discrete_map={'False Positive': C_WARNING, 'False Negative': C_MALIGNANT},
        points='all', title='Predictive Uncertainty in Misclassified Samples',
    )
    _apply(fig, height=340)
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
