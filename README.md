![Watch the video on YouTube](vid.mp4)

---

# Breast-Cancer-Prediction

This repository contains a machine learning project aimed at predicting the likelihood of breast cancer based on a dataset of medical features. The goal is to create a predictive model that assists in the early diagnosis of breast cancer, helping medical professionals make informed decisions.

---

## Overview
Breast cancer is one of the most common cancers in women worldwide. Early detection and accurate diagnosis are crucial for effective treatment and improved survival rates. This project utilizes machine learning algorithms to classify breast cancer cases as either malignant or benign based on medical diagnostic features.

---

### Key Details:
The dataset used in this app is sourced from Kaggle and includes 30 features extracted from breast cancer tumor samples. It contains information on both malignant and benign cases.
- **Number of Samples**: 569
- **Number of Features**: 30 (e.g., radius, texture, smoothness, compactness)
- **Target Variable**: Diagnosis (Malignant or Benign)

---

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/Khanz9664/Breast-Cancer-Prediction.git
   cd Breast-Cancer-Prediction

---

## How to Run the App
1. Ensure the `breast_cancer_model_v2.pkl` and `data.csv` files are in the project directory.
2. Open Terminal in Project Directory
3. Start the Streamlit app using this Command:
   ```bash
   streamlit run app.py
   ```
4. Open your web browser and go to `http://localhost:8501` to interact with the app.

---

## Usage
1. Adjust the sliders in the sidebar to input tumor feature values.
2. Click on the **Predict** button to get the prediction results.
3. Explore:
   - **Feature Importance:** Understand which features contributed the most to the prediction.
   - **Feature Correlation Heatmap:** Gain insights into relationships between features.
