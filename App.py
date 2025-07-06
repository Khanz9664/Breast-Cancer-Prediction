import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# ------------------ Streamlit Page Configuration ------------------
st.set_page_config(
    page_title="AI Cancer Diagnostics",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------ Modern Custom CSS ------------------
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        
        /* Root variables for consistent theming */
        :root {
            --primary-gradient: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            --secondary-gradient: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            --success-gradient: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
            --warning-gradient: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
            --dark-bg: #0f1419;
            --card-bg: rgba(255, 255, 255, 0.08);
            --text-primary: #ffffff;
            --text-secondary: #a0a9c0;
            --border-color: rgba(255, 255, 255, 0.1);
        }
        
        /* Global styles */
        .stApp {
            background: linear-gradient(135deg, #0f1419 0%, #1a1f36 50%, #2d1b69 100%);
            font-family: 'Inter', sans-serif;
        }
        
        /* Hide Streamlit elements */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        
        /* Main container */
        .main .block-container {
            padding: 2rem 1rem;
            max-width: 1200px;
        }
        
        /* Glassmorphism cards */
        .glass-card {
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(20px);
            border-radius: 20px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            padding: 2rem;
            margin: 1rem 0;
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
            transition: all 0.3s ease;
        }
        
        .glass-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 30px 60px rgba(0, 0, 0, 0.2);
        }
        
        /* Header styles */
        .hero-section {
            text-align: center;
            padding: 3rem 0;
            background: var(--primary-gradient);
            border-radius: 25px;
            margin-bottom: 3rem;
            position: relative;
            overflow: hidden;
        }
        
        .hero-section::before {
            content: '';
            position: absolute;
            top: -50%;
            left: -50%;
            width: 200%;
            height: 200%;
            background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
            animation: float 6s ease-in-out infinite;
        }
        
        @keyframes float {
            0%, 100% { transform: translateY(0px) rotate(0deg); }
            50% { transform: translateY(-20px) rotate(180deg); }
        }
        
        .hero-title {
            font-size: 3.5rem;
            font-weight: 700;
            color: white;
            margin-bottom: 1rem;
            text-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
            position: relative;
            z-index: 1;
        }
        
        .hero-subtitle {
            font-size: 1.4rem;
            color: rgba(255, 255, 255, 0.9);
            font-weight: 300;
            position: relative;
            z-index: 1;
        }
        
        /* Sidebar styling */
        .css-1d391kg {
            background: linear-gradient(180deg, #1a1f36 0%, #2d1b69 100%);
        }
        
        .css-1d391kg .css-1v0mbdj {
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(10px);
            border-radius: 15px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            margin: 1rem 0;
            padding: 1rem;
        }
        
        /* Button styling */
        .stButton > button {
            background: var(--secondary-gradient);
            color: white;
            border: none;
            border-radius: 15px;
            padding: 0.8rem 2rem;
            font-weight: 600;
            font-size: 1.1rem;
            transition: all 0.3s ease;
            box-shadow: 0 10px 30px rgba(245, 87, 108, 0.3);
            width: 100%;
        }
        
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 15px 40px rgba(245, 87, 108, 0.4);
        }
        
        /* Prediction result styling */
        .prediction-card {
            background: var(--success-gradient);
            border-radius: 20px;
            padding: 2rem;
            text-align: center;
            color: white;
            margin: 2rem 0;
            box-shadow: 0 15px 35px rgba(79, 172, 254, 0.3);
        }
        
        .prediction-title {
            font-size: 2rem;
            font-weight: 700;
            margin-bottom: 1rem;
        }
        
        .prediction-result {
            font-size: 3rem;
            font-weight: 800;
            margin: 1rem 0;
            text-shadow: 0 2px 10px rgba(0, 0, 0, 0.3);
        }
        
        /* Probability cards */
        .prob-container {
            display: flex;
            gap: 1rem;
            margin: 2rem 0;
        }
        
        .prob-card {
            flex: 1;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 15px;
            padding: 1.5rem;
            text-align: center;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        
        .prob-label {
            font-size: 1.1rem;
            font-weight: 600;
            color: rgba(255, 255, 255, 0.8);
            margin-bottom: 0.5rem;
        }
        
        .prob-value {
            font-size: 2.5rem;
            font-weight: 700;
            color: white;
        }
        
        .malignant {
            background: linear-gradient(135deg, #ff6b6b 0%, #ee5a52 100%);
        }
        
        .benign {
            background: linear-gradient(135deg, #51cf66 0%, #40c057 100%);
        }
        
        /* Input summary styling */
        .input-summary {
            background: rgba(255, 255, 255, 0.05);
            border-radius: 15px;
            padding: 1.5rem;
            margin: 2rem 0;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        
        /* Section headers */
        .section-header {
            font-size: 1.8rem;
            font-weight: 600;
            color: white;
            margin: 2rem 0 1rem 0;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        
        .section-header::before {
            content: '';
            width: 4px;
            height: 30px;
            background: var(--secondary-gradient);
            border-radius: 2px;
        }
        
        /* Instructions styling */
        .instructions {
            background: rgba(255, 255, 255, 0.03);
            border-radius: 15px;
            padding: 1.5rem;
            margin: 2rem 0;
            border-left: 4px solid #667eea;
        }
        
        .instructions h3 {
            color: #667eea;
            margin-bottom: 1rem;
            font-weight: 600;
        }
        
        .instructions ul {
            color: rgba(255, 255, 255, 0.8);
            line-height: 1.6;
        }
        
        /* Footer styling */
        .footer {
            background: rgba(255, 255, 255, 0.05);
            border-radius: 20px;
            padding: 2rem;
            margin-top: 3rem;
            text-align: center;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        
        .footer a {
            color: #667eea;
            text-decoration: none;
            transition: color 0.3s ease;
        }
        
        .footer a:hover {
            color: #764ba2;
        }
        
        /* Responsive design */
        @media (max-width: 768px) {
            .hero-title {
                font-size: 2.5rem;
            }
            
            .hero-subtitle {
                font-size: 1.2rem;
            }
            
            .prob-container {
                flex-direction: column;
            }
        }
        
        /* Animations */
        @keyframes fadeInUp {
            from {
                opacity: 0;
                transform: translateY(30px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        .animate-fade-in {
            animation: fadeInUp 0.8s ease-out;
        }
        
        /* Slider customization */
        .stSlider > div > div > div {
            background: var(--primary-gradient);
        }
        
        /* Success/Error message styling */
        .stSuccess {
            background: var(--success-gradient);
            color: white;
            border-radius: 15px;
            border: none;
        }
        
        .stInfo {
            background: rgba(102, 126, 234, 0.2);
            border: 1px solid rgba(102, 126, 234, 0.3);
            border-radius: 15px;
            color: white;
        }
        
        /* DataFrame styling */
        .dataframe {
            background: rgba(255, 255, 255, 0.05);
            border-radius: 15px;
            backdrop-filter: blur(10px);
        }
        
        .dataframe th {
            background: rgba(102, 126, 234, 0.3) !important;
            color: white !important;
            border: none !important;
        }
        
        .dataframe td {
            background: rgba(255, 255, 255, 0.02) !important;
            color: rgba(255, 255, 255, 0.9) !important;
            border: none !important;
        }
    </style>
""", unsafe_allow_html=True)

# ------------------ Hero Section ------------------
st.markdown("""
    <div class="hero-section">
        <div class="hero-title">🧬 AI Cancer Diagnostics</div>
        <div class="hero-subtitle">
            Advanced Machine Learning for Breast Cancer Detection
        </div>
    </div>
""", unsafe_allow_html=True)

# ------------------ Instructions ------------------
st.markdown("""
    <div class="instructions">
        <h3>📋 How to Use This Tool</h3>
        <ul>
            <li>🎛️ <strong>Configure:</strong> Use the sidebar sliders to input tumor characteristics</li>
            <li>🔍 <strong>Analyze:</strong> Click the "Run AI Analysis" button to get predictions</li>
            <li>📊 <strong>Interpret:</strong> View detailed probability scores and feature correlations</li>
        </ul>
    </div>
""", unsafe_allow_html=True)

# ------------------ Load Model and Scaler ------------------
@st.cache_resource
def load_model_and_scaler():
    try:
        bundle = joblib.load("breast_cancer_model_v2.pkl")
        if isinstance(bundle, dict):
            return bundle["model"], bundle.get("scaler", None)
        return bundle, None
    except Exception as e:
        st.error("🚨 Error loading model. Make sure 'breast_cancer_model_v2.pkl' is present.")
        st.stop()

# ------------------ Load Data ------------------
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("data.csv")
        df = df.drop(columns=["id"], errors='ignore')
        df['diagnosis'] = df['diagnosis'].apply(lambda x: 1 if x == 'M' else 0)
        return df
    except Exception as e:
        st.error("🚨 Error loading data. Make sure 'data.csv' is present.")
        st.stop()

model, loaded_scaler = load_model_and_scaler()
df = load_data()

# ------------------ Use Features from Scaler ------------------
expected_features = (
    loaded_scaler.feature_names_in_
    if loaded_scaler and hasattr(loaded_scaler, "feature_names_in_")
    else df.drop(columns=["diagnosis"]).columns
)

X = df[expected_features]
scaler = loaded_scaler if loaded_scaler else StandardScaler().fit(X)

# ------------------ Sidebar Input ------------------
st.sidebar.markdown("""
    <div style="text-align: center; padding: 1rem; background: rgba(255,255,255,0.05); border-radius: 15px; margin-bottom: 2rem;">
        <h2 style="color: white; margin: 0;">🎛️ Feature Controls</h2>
        <p style="color: rgba(255,255,255,0.7); margin: 0.5rem 0 0 0;">Adjust tumor characteristics</p>
    </div>
""", unsafe_allow_html=True)

def get_user_input():
    input_dict = {}
    
    # Group features by category for better organization
    mean_features = [f for f in expected_features if f.endswith('_mean')]
    se_features = [f for f in expected_features if f.endswith('_se')]
    worst_features = [f for f in expected_features if f.endswith('_worst')]
    
    categories = [
        ("📊 Mean Values", mean_features),
        ("📈 Standard Error", se_features),
        ("⚠️ Worst Values", worst_features)
    ]
    
    for category_name, features in categories:
        if features:
            st.sidebar.markdown(f"**{category_name}**")
            for feature in features:
                display_name = feature.replace('_', ' ').title()
                input_dict[feature] = st.sidebar.slider(
                    label=display_name,
                    min_value=float(X[feature].min()),
                    max_value=float(X[feature].max()),
                    value=float(X[feature].mean()),
                    step=0.01,
                    key=feature
                )
            st.sidebar.markdown("---")
    
    return pd.DataFrame(input_dict, index=[0])

user_input = get_user_input()

# ------------------ Show User Input ------------------
st.markdown('<div class="section-header">📋 Input Summary</div>', unsafe_allow_html=True)
st.markdown('<div class="input-summary">', unsafe_allow_html=True)
st.dataframe(user_input.style.format(precision=2).background_gradient(cmap="viridis"))
st.markdown('</div>', unsafe_allow_html=True)

# ------------------ Prediction Section ------------------
st.markdown('<div class="section-header">🧠 AI Analysis</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    if st.button("🚀 Run AI Analysis", key="predict_btn"):
        with st.spinner("🔄 Processing data through neural networks..."):
            user_input_aligned = user_input[expected_features]
            scaled_input = scaler.transform(user_input_aligned)

            prediction = model.predict(scaled_input)[0]
            proba = model.predict_proba(scaled_input)[0]

            result = "Malignant" if prediction == 1 else "Benign"
            result_color = "#ff6b6b" if prediction == 1 else "#51cf66"
            
            st.markdown(f"""
                <div class="prediction-card">
                    <div class="prediction-title">🎯 Analysis Complete</div>
                    <div class="prediction-result" style="color: {result_color};">
                        {result}
                    </div>
                    <p style="font-size: 1.2rem; margin: 0; opacity: 0.9;">
                        AI Confidence: {max(proba):.1%}
                    </p>
                </div>
            """, unsafe_allow_html=True)

            # Probability breakdown
            st.markdown(f"""
                <div class="prob-container">
                    <div class="prob-card malignant">
                        <div class="prob-label">Malignant Risk</div>
                        <div class="prob-value">{proba[1]:.1%}</div>
                    </div>
                    <div class="prob-card benign">
                        <div class="prob-label">Benign Probability</div>
                        <div class="prob-value">{proba[0]:.1%}</div>
                    </div>
                </div>
            """, unsafe_allow_html=True)

            # Create probability visualization
            fig = go.Figure(data=[
                go.Bar(
                    x=['Benign', 'Malignant'],
                    y=[proba[0], proba[1]],
                    marker_color=['#51cf66', '#ff6b6b'],
                    text=[f'{proba[0]:.1%}', f'{proba[1]:.1%}'],
                    textposition='auto',
                )
            ])
            
            fig.update_layout(
                title="Prediction Probability Distribution",
                title_font_color="white",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font_color="white",
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)

# ------------------ Feature Importance Visualization ------------------
st.markdown('<div class="section-header">📊 Feature Analysis</div>', unsafe_allow_html=True)

# Create correlation heatmap with Plotly
correlation_matrix = df[expected_features].corr()

fig_heatmap = go.Figure(data=go.Heatmap(
    z=correlation_matrix.values,
    x=correlation_matrix.columns,
    y=correlation_matrix.columns,
    colorscale='RdBu',
    zmid=0,
    text=correlation_matrix.values,
    texttemplate="%{text:.2f}",
    textfont={"size": 10},
    hoverongaps=False
))

fig_heatmap.update_layout(
    title="Feature Correlation Matrix",
    title_font_color="white",
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font_color="white",
    width=800,
    height=600
)

st.plotly_chart(fig_heatmap, use_container_width=True)

# ------------------ Statistics Overview ------------------
st.markdown('<div class="section-header">📈 Dataset Statistics</div>', unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
        <div class="glass-card" style="text-align: center;">
            <h3 style="color: #667eea; margin-bottom: 0.5rem;">📊 Total Samples</h3>
            <p style="font-size: 2rem; font-weight: 700; color: white; margin: 0;">{len(df)}</p>
        </div>
    """, unsafe_allow_html=True)

with col2:
    malignant_count = df['diagnosis'].sum()
    st.markdown(f"""
        <div class="glass-card" style="text-align: center;">
            <h3 style="color: #ff6b6b; margin-bottom: 0.5rem;">⚠️ Malignant</h3>
            <p style="font-size: 2rem; font-weight: 700; color: white; margin: 0;">{malignant_count}</p>
        </div>
    """, unsafe_allow_html=True)

with col3:
    benign_count = len(df) - malignant_count
    st.markdown(f"""
        <div class="glass-card" style="text-align: center;">
            <h3 style="color: #51cf66; margin-bottom: 0.5rem;">✅ Benign</h3>
            <p style="font-size: 2rem; font-weight: 700; color: white; margin: 0;">{benign_count}</p>
        </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
        <div class="glass-card" style="text-align: center;">
            <h3 style="color: #f093fb; margin-bottom: 0.5rem;">🔬 Features</h3>
            <p style="font-size: 2rem; font-weight: 700; color: white; margin: 0;">{len(expected_features)}</p>
        </div>
    """, unsafe_allow_html=True)

# ------------------ Footer ------------------
st.markdown("""
    <div class="footer">
        <h3 style="color: white; margin-bottom: 1rem;">👨‍💻 Created with ❤️ by Shahid Ul Islam</h3>
        <div style="display: flex; justify-content: center; gap: 2rem; flex-wrap: wrap;">
            <a href="mailto:shahid9664@gmail.com" style="display: flex; align-items: center; gap: 0.5rem;">
                📧 shahid9664@gmail.com
            </a>
            <a href="https://khanz9664.github.io/portfolio" target="_blank" style="display: flex; align-items: center; gap: 0.5rem;">
                🌐 Portfolio
            </a>
            <a href="https://github.com/Khanz9664" target="_blank" style="display: flex; align-items: center; gap: 0.5rem;">
                💻 GitHub
            </a>
            <a href="https://www.linkedin.com/in/shahid-ul-islam-13650998/" target="_blank" style="display: flex; align-items: center; gap: 0.5rem;">
                🔗 LinkedIn
            </a>
        </div>
        <p style="margin-top: 1rem; color: rgba(255,255,255,0.6);">
            🚀 Powered by Machine Learning & Modern Web Technologies
        </p>
    </div>
""", unsafe_allow_html=True)
