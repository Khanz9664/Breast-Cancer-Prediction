import streamlit as st
import pandas as pd
import os

@st.cache_data
def load_data():
    """Load the breast cancer dataset from the data directory."""
    data_path = os.path.join("data", "data.csv")
    try:
        df = pd.read_csv(data_path)
        # Basic preprocessing consistent with original App.py
        df = df.drop(columns=["id"], errors='ignore')
        if 'diagnosis' in df.columns:
            df['diagnosis'] = df['diagnosis'].apply(lambda x: 1 if x == 'M' else 0)
        return df
    except Exception as e:
        st.error(f"Error loading data from {data_path}. Please ensure the file exists.")
        st.stop()
