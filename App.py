import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


st.set_page_config(
    page_title="Breast Cancer Prediction",
    page_icon="🎗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load the trained model
model = joblib.load('breast_cancer_model.pkl')

# Load and preprocess the dataset
df = pd.read_csv('data.csv')
df = df.drop(columns=['id'])  # Drop non-informative columns

# Encode target column if not already encoded
df['diagnosis'] = df['diagnosis'].apply(lambda x: 1 if x == 'M' else 0)

# Sidebar for user input
st.sidebar.header("🔧 Configure Input Parameters")
st.sidebar.markdown("""---""")

def user_input_features():
    """Capture user input for prediction."""
    input_features = {}
    for feature in df.columns[:-1]:  # Exclude target column
        min_val = float(df[feature].min())
        max_val = float(df[feature].max())
        mean_val = float(df[feature].mean())
        input_features[feature] = st.sidebar.slider(
            f"{feature.replace('_', ' ').title()}", min_val, max_val, mean_val
        )
    return pd.DataFrame(input_features, index=[0])

# Gather user input
user_data = user_input_features()

# Display user input
def display_user_input():
    st.markdown("""### User Input Data""")
    st.dataframe(user_data.style.format(precision=2).background_gradient(cmap="Blues"))

# Scale features
scaler = StandardScaler()
x_scaled = scaler.fit_transform(df.iloc[:, :-1])

# Header and description
st.title("🎗️ Breast Cancer Prediction App")
st.markdown("""
This app predicts whether a breast cancer diagnosis is **Malignant** (M) or **Benign** (B) based on tumor features.

#### Instructions:
1. Use the sliders in the sidebar to input tumor features.
2. Click **Predict** to view the results.
3. Explore feature importance and correlation heatmap below.
---
""")

# Prediction
col1, col2 = st.columns([2, 1])
with col1:
    st.markdown("### Configure and Predict")
    if st.button("💡 Predict"):
        input_scaled = scaler.transform(user_data)
        prediction = model.predict(input_scaled)[0]
        prediction_proba = model.predict_proba(input_scaled)[0]
        
        st.markdown("""#### Prediction Result""")
        result = "Malignant" if prediction == 1 else "Benign"
        st.success(f"The tumor is predicted to be **{result}**.")
        
        st.markdown("""#### Prediction Probabilities""")
        st.write(f"- **Malignant (M):** {prediction_proba[1]:.2f}")
        st.write(f"- **Benign (B):** {prediction_proba[0]:.2f}")

# Feature importance visualization
st.markdown("### 🔍 Feature Importance")
feature_importances = pd.DataFrame({
    "Feature": df.columns[:-1].str.replace('_', ' ').str.title(),
    "Importance": model.feature_importances_
}).sort_values(by="Importance", ascending=False)

fig = px.bar(
    feature_importances,
    x="Importance",
    y="Feature",
    orientation="h",
    title="Feature Importance",
    color="Importance",
    color_continuous_scale="Blues"
)
st.plotly_chart(fig, use_container_width=True)

# Correlation heatmap
st.markdown("### 🌐 Correlation Heatmap")
correlation_matrix = df.corr()
fig, ax = plt.subplots(figsize=(12, 8))
sns.heatmap(
    correlation_matrix, 
    annot=True, 
    fmt=".2f", 
    cmap="coolwarm", 
    linewidths=0.5,
    cbar_kws={"shrink": 0.8}
)
st.pyplot(fig)

# Summary and footer
st.markdown("""---
### Summary
- Configure tumor features using the sidebar sliders.
- Click **Predict** to analyze the tumor type and probabilities.
- Visualize the importance of features and their relationships.

**Developed with ❤️ by Shahid Ul Islam**
""")

