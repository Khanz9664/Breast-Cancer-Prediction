import streamlit as st


def inject_custom_css():
    """Inject the full design-token CSS system into the Streamlit app."""
    st.markdown("""
        <style>
        /* ════════════════════════════════════════════════════════════════
           DESIGN TOKENS
        ════════════════════════════════════════════════════════════════ */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

        :root {
            /* Colors */
            --primary:           #3b82f6;
            --primary-dark:      #2563eb;
            --primary-light:     #eff6ff;
            --bg:                #ffffff;
            --bg-secondary:      #f8fafc;
            --bg-muted:          #f1f5f9;
            --border:            #e2e8f0;
            --border-strong:     #cbd5e1;

            /* Text */
            --text:              #0f172a;
            --text-secondary:    #475569;
            --text-muted:        #64748b;

            /* Semantic */
            --success:           #16a34a;
            --success-bg:        #f0fdf4;
            --success-border:    #bbf7d0;
            --warning:           #f59e0b;
            --warning-bg:        #fffbeb;
            --warning-border:    #fde68a;
            --error:             #dc2626;
            --error-bg:          #fef2f2;
            --error-border:      #fecaca;
            --info:              #2563eb;
            --info-bg:           #eff6ff;
            --info-border:       #bfdbfe;

            /* Spacing */
            --radius:            8px;
            --radius-sm:         6px;
            --radius-lg:         10px;
            --shadow-sm:         0 1px 3px rgba(15,23,42,0.08), 0 1px 2px rgba(15,23,42,0.04);
            --shadow:            0 2px 6px rgba(15,23,42,0.08), 0 1px 3px rgba(15,23,42,0.06);
        }

        /* ════════════════════════════════════════════════════════════════
           GLOBAL RESET
        ════════════════════════════════════════════════════════════════ */
        .stApp {
            background: var(--bg) !important;
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            color: var(--text);
        }

        #MainMenu { visibility: hidden; }
        footer    { visibility: hidden; }
        header    { visibility: hidden; }

        .main .block-container {
            padding: 0 2rem 3rem 2rem;
            max-width: 1360px;
        }

        /* ════════════════════════════════════════════════════════════════
           TYPOGRAPHY SCALE
        ════════════════════════════════════════════════════════════════ */

        /* Override Streamlit's h1–h3 so they respect our scale */
        h1, .h1 {
            font-size: 28px !important;
            font-weight: 700 !important;
            letter-spacing: -0.025em !important;
            color: var(--text) !important;
            line-height: 1.25 !important;
            margin: 0 0 4px 0 !important;
        }
        h2, .h2 {
            font-size: 22px !important;
            font-weight: 600 !important;
            letter-spacing: -0.015em !important;
            color: var(--text) !important;
            line-height: 1.3 !important;
            margin: 0 0 8px 0 !important;
        }
        h3, .h3, [data-testid="stHeading"] h3 {
            font-size: 18px !important;
            font-weight: 600 !important;
            letter-spacing: -0.01em !important;
            color: var(--text) !important;
            line-height: 1.4 !important;
        }
        p, li, .body-text {
            font-size: 15px;
            font-weight: 400;
            color: var(--text-secondary);
            line-height: 1.65;
        }
        .label-text {
            font-size: 13px;
            font-weight: 500;
            color: var(--text-secondary);
        }
        .caption-text {
            font-size: 12px;
            font-weight: 400;
            color: var(--text-muted);
        }
        /* Streamlit's caption */
        [data-testid="stCaptionContainer"] p {
            font-size: 12px !important;
            color: var(--text-muted) !important;
        }
        [data-testid="stMarkdownContainer"] p {
            font-size: 15px;
            color: var(--text-secondary);
        }

        /* ════════════════════════════════════════════════════════════════
           PAGE HEADER
        ════════════════════════════════════════════════════════════════ */
        .page-header {
            padding: 1.75rem 0 1.25rem 0;
            border-bottom: 1px solid var(--border);
            margin-bottom: 0;
        }
        .page-header-title {
            font-size: 28px;
            font-weight: 700;
            color: var(--text);
            letter-spacing: -0.025em;
            margin: 0 0 4px 0;
            line-height: 1.25;
        }
        .page-header-subtitle {
            font-size: 14px;
            font-weight: 400;
            color: var(--text-muted);
            margin: 0;
        }
        .page-header-meta {
            display: flex;
            align-items: center;
            gap: 8px;
            margin-top: 10px;
        }

        /* ════════════════════════════════════════════════════════════════
           SECTION LABELS
        ════════════════════════════════════════════════════════════════ */
        .section-label {
            font-size: 11px;
            font-weight: 600;
            letter-spacing: 0.07em;
            text-transform: uppercase;
            color: var(--text-muted);
            padding-bottom: 8px;
            border-bottom: 1px solid var(--border);
            margin: 24px 0 16px 0;
        }
        .section-heading {
            font-size: 18px;
            font-weight: 600;
            color: var(--text);
            letter-spacing: -0.01em;
            margin: 0 0 4px 0;
        }
        .section-subtext {
            font-size: 13px;
            color: var(--text-muted);
            margin: 0 0 16px 0;
        }

        /* ════════════════════════════════════════════════════════════════
           CARD COMPONENTS
        ════════════════════════════════════════════════════════════════ */
        .card {
            background: var(--bg);
            border: 1px solid var(--border);
            border-radius: var(--radius);
            padding: 20px 24px;
            box-shadow: var(--shadow-sm);
            margin-bottom: 16px;
        }
        .card-sm {
            background: var(--bg);
            border: 1px solid var(--border);
            border-radius: var(--radius);
            padding: 14px 18px;
            box-shadow: var(--shadow-sm);
            margin-bottom: 12px;
        }
        .card-muted {
            background: var(--bg-secondary);
            border: 1px solid var(--border);
            border-radius: var(--radius);
            padding: 16px 20px;
            margin-bottom: 14px;
        }
        .card-header {
            font-size: 13px;
            font-weight: 600;
            letter-spacing: 0.04em;
            text-transform: uppercase;
            color: var(--text-muted);
            margin-bottom: 12px;
            padding-bottom: 10px;
            border-bottom: 1px solid var(--border);
        }
        .card-divider {
            border: none;
            border-top: 1px solid var(--border);
            margin: 16px 0;
        }

        /* ════════════════════════════════════════════════════════════════
           STATUS BADGES
        ════════════════════════════════════════════════════════════════ */
        .badge {
            display: inline-block;
            padding: 2px 10px;
            border-radius: 99px;
            font-size: 12px;
            font-weight: 600;
            letter-spacing: 0.02em;
        }
        .badge-success { background: var(--success-bg); color: var(--success); border: 1px solid var(--success-border); }  # noqa: E501
        .badge-warning { background: var(--warning-bg); color: #92400e; border: 1px solid var(--warning-border); }  # noqa: E501
        .badge-error   { background: var(--error-bg);   color: var(--error);   border: 1px solid var(--error-border); }  # noqa: E501
        .badge-info    { background: var(--info-bg);    color: var(--info);    border: 1px solid var(--info-border); }  # noqa: E501
        .badge-neutral { background: var(--bg-muted);   color: var(--text-secondary); border: 1px solid var(--border); }  # noqa: E501

        /* ════════════════════════════════════════════════════════════════
           RESULT / DIAGNOSTIC CARDS
        ════════════════════════════════════════════════════════════════ */
        .result-card {
            background: var(--bg);
            border: 1px solid var(--border);
            border-radius: var(--radius-lg);
            padding: 24px 28px;
            text-align: center;
            box-shadow: var(--shadow);
        }
        .result-card-label {
            font-size: 11px;
            font-weight: 600;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: var(--text-muted);
            margin-bottom: 8px;
        }
        .result-value {
            font-size: 36px;
            font-weight: 700;
            letter-spacing: -0.03em;
            line-height: 1.1;
            margin: 0;
        }
        .result-value.malignant { color: var(--error); }
        .result-value.benign    { color: var(--success); }
        .result-confidence {
            font-size: 14px;
            color: var(--text-muted);
            margin-top: 6px;
        }
        .result-card.malignant-card { border-left: 4px solid var(--error); }
        .result-card.benign-card    { border-left: 4px solid var(--success); }

        /* ════════════════════════════════════════════════════════════════
           PROBABILITY CELLS
        ════════════════════════════════════════════════════════════════ */
        .prob-row    { display: flex; gap: 12px; margin: 14px 0; }
        .prob-cell {
            flex: 1;
            background: var(--bg);
            border: 1px solid var(--border);
            border-radius: var(--radius);
            padding: 16px 20px;
            text-align: center;
            box-shadow: var(--shadow-sm);
        }
        .prob-cell.malignant { border-top: 3px solid var(--error); }
        .prob-cell.benign    { border-top: 3px solid var(--success); }
        .prob-cell-label {
            font-size: 11px;
            font-weight: 600;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            color: var(--text-muted);
            margin-bottom: 6px;
        }
        .prob-cell-value {
            font-size: 28px;
            font-weight: 700;
            letter-spacing: -0.02em;
            color: var(--text);
        }

        /* ════════════════════════════════════════════════════════════════
           ROBUSTNESS CARD
        ════════════════════════════════════════════════════════════════ */
        .robustness-card {
            background: var(--bg-secondary);
            border: 1px solid var(--border);
            border-radius: var(--radius);
            padding: 16px 20px;
            margin: 12px 0;
        }
        .robustness-title {
            font-size: 11px;
            font-weight: 600;
            letter-spacing: 0.07em;
            text-transform: uppercase;
            color: var(--text-muted);
            margin-bottom: 10px;
        }
        .robustness-row    { display: flex; justify-content: space-between; align-items: center; }
        .robustness-level  { font-size: 20px; font-weight: 700; color: var(--text); }
        .robustness-score  { font-size: 13px; color: var(--text-muted); }
        .robustness-ci     { font-size: 12px; color: var(--text-muted); margin-top: 6px; font-family: 'Courier New', monospace; }  # noqa: E501

        /* ════════════════════════════════════════════════════════════════
           CLINICAL INSIGHT ROWS
        ════════════════════════════════════════════════════════════════ */
        .insight-row {
            padding: 10px 14px;
            border-radius: var(--radius-sm);
            border-left: 3px solid;
            border: 1px solid;
            margin-bottom: 8px;
            font-size: 14px;
            line-height: 1.55;
        }
        .insight-row strong { display: block; font-size: 12.5px; font-weight: 600; margin-bottom: 2px; }
        .insight-row.high   { background: var(--error-bg);   border-color: var(--error-border);   color: #7f1d1d; border-left: 3px solid var(--error); }  # noqa: E501
        .insight-row.medium { background: var(--warning-bg); border-color: var(--warning-border); color: #78350f; border-left: 3px solid var(--warning); }  # noqa: E501
        .insight-row.low    { background: var(--success-bg); border-color: var(--success-border); color: #14532d; border-left: 3px solid var(--success); }  # noqa: E501

        /* ════════════════════════════════════════════════════════════════
           METRIC TILES (for custom stats row)
        ════════════════════════════════════════════════════════════════ */
        .stat-tile {
            background: var(--bg);
            border: 1px solid var(--border);
            border-radius: var(--radius);
            padding: 16px 20px;
            box-shadow: var(--shadow-sm);
        }
        .stat-tile-label {
            font-size: 11px;
            font-weight: 600;
            letter-spacing: 0.07em;
            text-transform: uppercase;
            color: var(--text-muted);
            margin-bottom: 6px;
        }
        .stat-tile-value {
            font-size: 26px;
            font-weight: 700;
            color: var(--text);
            letter-spacing: -0.02em;
        }
        .stat-tile-value.success { color: var(--success); }
        .stat-tile-value.error   { color: var(--error); }
        .stat-tile-value.primary { color: var(--primary); }

        /* ════════════════════════════════════════════════════════════════
           BUTTONS
        ════════════════════════════════════════════════════════════════ */
        .stButton > button {
            background-color: var(--primary) !important;
            color: #ffffff !important;
            border: 1px solid var(--primary) !important;
            border-radius: var(--radius-sm) !important;
            padding: 8px 20px !important;
            font-size: 14px !important;
            font-weight: 500 !important;
            letter-spacing: 0.01em !important;
            box-shadow: 0 1px 2px rgba(59,130,246,0.20) !important;
            transition: background-color 0.15s ease, box-shadow 0.15s ease !important;
            width: 100% !important;
        }
        .stButton > button:hover {
            background-color: var(--primary-dark) !important;
            border-color: var(--primary-dark) !important;
            box-shadow: 0 3px 8px rgba(37,99,235,0.25) !important;
        }
        .stButton > button:focus {
            outline: 2px solid var(--primary) !important;
            outline-offset: 2px !important;
        }

        /* Download button */
        .stDownloadButton > button {
            background-color: var(--bg) !important;
            color: var(--primary) !important;
            border: 1px solid var(--primary) !important;
            border-radius: var(--radius-sm) !important;
            font-size: 14px !important;
            font-weight: 500 !important;
            width: 100% !important;
        }
        .stDownloadButton > button:hover {
            background-color: var(--primary-light) !important;
        }

        /* ════════════════════════════════════════════════════════════════
           TABS
        ════════════════════════════════════════════════════════════════ */
        .stTabs [data-baseweb="tab-list"] {
            gap: 0 !important;
            border-bottom: 1px solid var(--border) !important;
            background: transparent !important;
            padding: 0 !important;
        }
        .stTabs [data-baseweb="tab"] {
            background: transparent !important;
            border: none !important;
            border-bottom: 2px solid transparent !important;
            color: var(--text-muted) !important;
            font-size: 14px !important;
            font-weight: 500 !important;
            padding: 10px 18px !important;
            margin-bottom: -1px !important;
        }
        .stTabs [aria-selected="true"] {
            color: var(--primary) !important;
            border-bottom: 2px solid var(--primary) !important;
            background: transparent !important;
            font-weight: 600 !important;
        }
        .stTabs [data-baseweb="tab-panel"] {
            padding: 20px 0 0 0 !important;
        }

        /* ════════════════════════════════════════════════════════════════
           SIDEBAR
        ════════════════════════════════════════════════════════════════ */
        [data-testid="stSidebar"] {
            background-color: var(--bg-secondary) !important;
            border-right: 1px solid var(--border) !important;
        }
        [data-testid="stSidebar"] > div { padding: 0 !important; }
        .sidebar-inner { padding: 1.25rem 1rem 2rem 1rem; }

        .sidebar-panel-label {
            font-size: 10px;
            font-weight: 600;
            letter-spacing: 0.1em;
            text-transform: uppercase;
            color: var(--text-muted);
        }
        .sidebar-panel-title {
            font-size: 17px;
            font-weight: 700;
            color: var(--text);
            margin: 4px 0 0 0;
        }
        .sidebar-section-label {
            font-size: 11px;
            font-weight: 700;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            color: var(--text-muted);
            padding: 4px 0;
            border-bottom: 1px solid var(--border);
            margin-bottom: 10px;
        }

        /* Slider labels */
        [data-testid="stSidebar"] label {
            font-size: 13px !important;
            font-weight: 500 !important;
            color: var(--text-secondary) !important;
        }
        [data-testid="stSidebar"] .stSlider > div > div > div {
            background: var(--primary) !important;
        }

        /* ════════════════════════════════════════════════════════════════
           NATIVE STREAMLIT METRICS
        ════════════════════════════════════════════════════════════════ */
        [data-testid="stMetric"] {
            background: var(--bg) !important;
            border: 1px solid var(--border) !important;
            border-radius: var(--radius) !important;
            padding: 14px 18px !important;
            box-shadow: var(--shadow-sm) !important;
        }
        [data-testid="stMetricLabel"] p {
            font-size: 11px !important;
            font-weight: 600 !important;
            letter-spacing: 0.06em !important;
            text-transform: uppercase !important;
            color: var(--text-muted) !important;
        }
        [data-testid="stMetricValue"] {
            font-size: 22px !important;
            font-weight: 700 !important;
            color: var(--text) !important;
        }

        /* ════════════════════════════════════════════════════════════════
           EXPANDERS
        ════════════════════════════════════════════════════════════════ */
        [data-testid="stExpander"] {
            border: 1px solid var(--border) !important;
            border-radius: var(--radius) !important;
            background: var(--bg) !important;
            box-shadow: var(--shadow-sm) !important;
            margin-bottom: 10px !important;
        }
        [data-testid="stExpander"] summary {
            font-size: 14px !important;
            font-weight: 500 !important;
            color: var(--text) !important;
            padding: 12px 16px !important;
        }

        /* ════════════════════════════════════════════════════════════════
           DATAFRAMES
        ════════════════════════════════════════════════════════════════ */
        [data-testid="stDataFrame"] {
            border: 1px solid var(--border) !important;
            border-radius: var(--radius) !important;
            overflow: hidden !important;
        }
        iframe[title="st_aggrid"] { border-radius: var(--radius); }

        /* ════════════════════════════════════════════════════════════════
           ALERTS (st.info / st.warning / st.error / st.success)
        ════════════════════════════════════════════════════════════════ */
        [data-testid="stAlert"] {
            border-radius: var(--radius-sm) !important;
        }

        /* ════════════════════════════════════════════════════════════════
           CONTAINERS (st.container border=True)
        ════════════════════════════════════════════════════════════════ */
        [data-testid="stVerticalBlockBorderWrapper"] {
            border: 1px solid var(--border) !important;
            border-radius: var(--radius) !important;
            background: var(--bg) !important;
            box-shadow: var(--shadow-sm) !important;
            padding: 16px !important;
        }

        /* ════════════════════════════════════════════════════════════════
           INSTRUCTIONS PANEL
        ════════════════════════════════════════════════════════════════ */
        .instructions-panel {
            background: var(--info-bg);
            border: 1px solid var(--info-border);
            border-left: 3px solid var(--info);
            border-radius: var(--radius-sm);
            padding: 12px 16px;
            margin: 16px 0;
        }
        .instructions-panel p {
            font-size: 14px;
            color: #1e3a8a;
            margin: 0;
        }
        .instructions-panel strong { color: #1e40af; }

        /* ════════════════════════════════════════════════════════════════
           FOOTER
        ════════════════════════════════════════════════════════════════ */
        .app-footer {
            border-top: 1px solid var(--border);
            padding: 20px 0 12px 0;
            margin-top: 40px;
            text-align: center;
        }
        .app-footer p {
            font-size: 12px;
            color: var(--text-muted);
            margin: 0 0 6px 0;
        }
        .app-footer a {
            color: var(--primary-dark);
            text-decoration: none;
            margin: 0 10px;
            font-size: 12px;
            font-weight: 500;
        }
        .app-footer a:hover { text-decoration: underline; }

        /* ════════════════════════════════════════════════════════════════
           SCROLLBAR
        ════════════════════════════════════════════════════════════════ */
        ::-webkit-scrollbar            { width: 6px; height: 6px; }
        ::-webkit-scrollbar-track      { background: var(--bg-secondary); }
        ::-webkit-scrollbar-thumb      { background: var(--border-strong); border-radius: 3px; }
        ::-webkit-scrollbar-thumb:hover{ background: var(--text-muted); }
        </style>
    """, unsafe_allow_html=True)


# ── Page Header ──────────────────────────────────────────────────────────────

def render_hero():
    """Render the compact clinical page header with version badge."""
    st.markdown("""
        <div class="page-header">
            <div class="page-header-title">Breast Cancer Diagnostic Support</div>
            <div class="page-header-subtitle">
                Machine learning analysis of Fine Needle Aspiration (FNA) cytology measurements
            </div>
            <div class="page-header-meta">
                <span class="badge badge-info">v2.2 Research</span>
                <span class="badge badge-neutral">Random Forest Ensemble</span>
                <span class="badge badge-neutral">WDBC Dataset</span>
            </div>
        </div>
    """, unsafe_allow_html=True)


def render_instructions():
    """Render concise usage instructions."""
    st.markdown("""
        <div class="instructions-panel">
            <p>
                <strong>Workflow:</strong> Configure tumor measurements in the
                <strong>Clinical Configuration</strong> sidebar &rarr; click
                <strong>Run Analysis</strong> to generate a prediction &rarr; review
                AI results, SHAP explanations, and clinical insights below.
            </p>
        </div>
    """, unsafe_allow_html=True)


def render_footer():
    """Render the application footer."""
    st.markdown("""
        <div class="app-footer">
            <p>Breast Cancer Diagnostic Support System &mdash; For research and educational use only.</p>
            <p>
                <a href="mailto:shahid9664@gmail.com">Email</a>
                <a href="https://khanz9664.github.io/portfolio" target="_blank">Portfolio</a>
                <a href="https://github.com/Khanz9664" target="_blank">GitHub</a>
                <a href="https://www.linkedin.com/in/shahid-ul-islam-13650998/" target="_blank">LinkedIn</a>
            </p>
        </div>
    """, unsafe_allow_html=True)


def render_stats(df, features):
    """Render clean statistical summary tiles for the dataset."""
    malignant = int(df['diagnosis'].sum())
    benign = len(df) - malignant

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f'<div class="stat-tile"><div class="stat-tile-label">Total Samples</div><div class="stat-tile-value">{len(df)}</div></div>', unsafe_allow_html=True)  # noqa: E501
    with c2:
        st.markdown(f'<div class="stat-tile"><div class="stat-tile-label">Malignant</div><div class="stat-tile-value error">{malignant}</div></div>', unsafe_allow_html=True)  # noqa: E501
    with c3:
        st.markdown(f'<div class="stat-tile"><div class="stat-tile-label">Benign</div><div class="stat-tile-value success">{benign}</div></div>', unsafe_allow_html=True)  # noqa: E501
    with c4:
        st.markdown(f'<div class="stat-tile"><div class="stat-tile-label">Features</div><div class="stat-tile-value primary">{len(features)}</div></div>', unsafe_allow_html=True)  # noqa: E501
