import streamlit as st
import plotly.express as px


def render_dataset_explorer(df, features):
    """Clean, card-based dataset explorer with filter controls and summary metrics."""

    st.markdown('<div class="section-heading">Dataset Explorer</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-subtext">Filter, analyse, and visualise the WDBC training dataset.</div>',
        unsafe_allow_html=True,
    )

    col_left, col_right = st.columns([1, 3])

    # ── Filters Panel ──────────────────────────────────────────────────────────
    with col_left:
        with st.container(border=True):
            st.markdown('<div class="card-header">Filters</div>', unsafe_allow_html=True)

            diag_opts = st.multiselect(
                'Diagnosis',
                options=['Benign', 'Malignant'],
                default=['Benign', 'Malignant'],
                help='Filter samples by primary diagnosis classification.',
            )
            diag_map = {'Benign': 0, 'Malignant': 1}
            filtered_df = df[df['diagnosis'].isin([diag_map[d] for d in diag_opts])].copy()

            filter_feat = st.selectbox(
                'Feature range filter',
                options=features,
                help='Restrict displayed samples to a value range for this feature.',
            )
            mn = float(df[filter_feat].min())
            mx = float(df[filter_feat].max())
            rng = st.slider(
                filter_feat.replace('_', ' ').title(),
                min_value=mn, max_value=mx, value=(mn, mx),
            )
            filtered_df = filtered_df[
                (filtered_df[filter_feat] >= rng[0]) &
                (filtered_df[filter_feat] <= rng[1])
            ]

            # Summary tiles
            st.markdown('<hr class="card-divider">', unsafe_allow_html=True)
            m1, m2 = st.columns(2)
            m1.metric('Shown', len(filtered_df))
            m2.metric('Total', len(df))

        # Stats card
        with st.container(border=True):
            st.markdown('<div class="card-header">Summary Statistics</div>', unsafe_allow_html=True)
            st.dataframe(
                filtered_df[features].describe().T[['mean', 'std', 'min', 'max']].round(3),
                use_container_width=True,
            )

    # ── Visualisation Panel ────────────────────────────────────────────────────
    with col_right:
        with st.container(border=True):
            st.markdown('<div class="card-header">Feature Scatter Analysis</div>', unsafe_allow_html=True)

            ax1, ax2 = st.columns(2)
            x_feat = ax1.selectbox('X Axis', options=features, index=0, key='exp_x')
            y_feat = ax2.selectbox('Y Axis', options=features,
                                   index=min(4, len(features) - 1), key='exp_y')

            fig = px.scatter(
                filtered_df,
                x=x_feat, y=y_feat,
                color=filtered_df['diagnosis'].map({0: 'Benign', 1: 'Malignant'}),
                color_discrete_map={'Benign': '#16a34a', 'Malignant': '#dc2626'},
                opacity=0.65,
                marginal_x='histogram', marginal_y='box',
                title=f'{x_feat.replace("_", " ").title()} vs {y_feat.replace("_", " ").title()}',
                hover_data=features[:5],
            )
            fig.update_layout(
                paper_bgcolor='#ffffff', plot_bgcolor='#f8fafc',
                font=dict(family='Inter, sans-serif', color='#0f172a', size=12),
                height=460, margin=dict(l=16, r=16, t=44, b=16),
            )
            fig.update_xaxes(gridcolor='#f1f5f9', linecolor='#e2e8f0')
            fig.update_yaxes(gridcolor='#f1f5f9', linecolor='#e2e8f0')
            st.plotly_chart(fig, use_container_width=True)

    # ── Raw Data Table ─────────────────────────────────────────────────────────
    st.markdown(
        '<div class="section-label" style="margin-top:20px;">Filtered Sample Data</div>',
        unsafe_allow_html=True,
    )
    with st.container(border=True):
        display_df = filtered_df.copy()
        display_df['diagnosis'] = display_df['diagnosis'].map({0: 'Benign', 1: 'Malignant'})
        display_df = display_df.rename(columns={'diagnosis': 'Diagnosis'})
        st.dataframe(display_df, use_container_width=True, hide_index=True)
