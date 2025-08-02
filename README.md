# Breast Cancer Prediction

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-2C2D72?style=for-the-badge&logo=pandas&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)

A web application built with Streamlit to predict breast cancer based on tumor features. This project uses a Random Forest Classifier model trained on the Breast Cancer Wisconsin (Diagnostic) dataset.

---

## Features

-   [cite_start]**Interactive Prediction**: Users can input tumor features through sliders in the sidebar to get a real-time prediction of whether a tumor is malignant or benign. [cite: 1]
-   [cite_start]**Data Visualization**: The application displays a correlation heatmap and distribution plots of the dataset features. [cite: 1]
-   [cite_start]**Model Explainability**: The underlying Jupyter Notebook uses SHAP (SHapley Additive exPlanations) for model explainability. [cite: 2]
-   [cite_start]**User-Friendly Interface**: A simple and intuitive web interface powered by Streamlit. [cite: 1]

---

## Dataset

[cite_start]The model is trained on the **Breast Cancer Wisconsin (Diagnostic) Dataset**. [cite: 2] This dataset contains features computed from a digitized image of a fine needle aspirate (FNA) of a breast mass. [cite_start]They describe characteristics of the cell nuclei present in the image. [cite: 2]

-   [cite_start]**Number of Instances**: 569 [cite: 2]
-   [cite_start]**Number of Attributes**: 30 numeric, predictive attributes and the class [cite: 2]
-   [cite_start]**Class Distribution**: 212 Malignant, 357 Benign [cite: 1]

---

## Model

[cite_start]The prediction model is a **Random Forest Classifier** from the scikit-learn library. [cite: 2]

-   [cite_start]**Model Training**: The model is trained on a scaled version of the dataset, split into 80% for training and 20% for testing. [cite: 2]
-   [cite_start]**Model Evaluation**: The model's performance is evaluated using a classification report, confusion matrix, and ROC-AUC score. [cite: 2]
-   [cite_start]**Model Serialization**: The trained model is saved as `breast_cancer_model.pkl` using `joblib` for use in the Streamlit application. [cite: 1, 2]

---

## Technologies Used

-   **Python**: The core programming language for the project.
-   [cite_start]**Streamlit**: For creating and deploying the web application. [cite: 1]
-   [cite_start]**Pandas**: For data manipulation and analysis. [cite: 1, 2]
-   [cite_start]**Numpy**: For numerical operations. [cite: 1, 2]
-   [cite_start]**Scikit-learn**: For implementing the machine learning model and preprocessing. [cite: 2]
-   [cite_start]**Joblib**: For saving and loading the trained model. [cite: 1]
-   [cite_start]**Plotly and Seaborn**: For creating interactive and static visualizations. [cite: 1, 2]
-   [cite_start]**Jupyter Notebook**: For model development and experimentation. [cite: 2]

---

## Installation and Setup

1.  **Clone the repository**:
    ```bash
    git clone [https://github.com/Khanz9664/Breast-Cancer-Prediction.git](https://github.com/Khanz9664/Breast-Cancer-Prediction.git)
    cd Breast-Cancer-Prediction
    ```

2.  **Create a virtual environment**:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

---

## Usage

To run the Streamlit application, execute the following command in your terminal:

```bash
streamlit run App.py
```
Here is the readme.md file for your Breast Cancer Prediction project.

Markdown

# Breast Cancer Prediction

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-2C2D72?style=for-the-badge&logo=pandas&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)

A web application built with Streamlit to predict breast cancer based on tumor features. This project uses a Random Forest Classifier model trained on the Breast Cancer Wisconsin (Diagnostic) dataset.

---

## Features

-   [cite_start]**Interactive Prediction**: Users can input tumor features through sliders in the sidebar to get a real-time prediction of whether a tumor is malignant or benign. [cite: 1]
-   [cite_start]**Data Visualization**: The application displays a correlation heatmap and distribution plots of the dataset features. [cite: 1]
-   [cite_start]**Model Explainability**: The underlying Jupyter Notebook uses SHAP (SHapley Additive exPlanations) for model explainability. [cite: 2]
-   [cite_start]**User-Friendly Interface**: A simple and intuitive web interface powered by Streamlit. [cite: 1]

---

## Dataset

[cite_start]The model is trained on the **Breast Cancer Wisconsin (Diagnostic) Dataset**. [cite: 2] This dataset contains features computed from a digitized image of a fine needle aspirate (FNA) of a breast mass. [cite_start]They describe characteristics of the cell nuclei present in the image. [cite: 2]

-   [cite_start]**Number of Instances**: 569 [cite: 2]
-   [cite_start]**Number of Attributes**: 30 numeric, predictive attributes and the class [cite: 2]
-   [cite_start]**Class Distribution**: 212 Malignant, 357 Benign [cite: 1]

---

## Model

[cite_start]The prediction model is a **Random Forest Classifier** from the scikit-learn library. [cite: 2]

-   [cite_start]**Model Training**: The model is trained on a scaled version of the dataset, split into 80% for training and 20% for testing. [cite: 2]
-   [cite_start]**Model Evaluation**: The model's performance is evaluated using a classification report, confusion matrix, and ROC-AUC score. [cite: 2]
-   [cite_start]**Model Serialization**: The trained model is saved as `breast_cancer_model.pkl` using `joblib` for use in the Streamlit application. [cite: 1, 2]

---

## Technologies Used

-   **Python**: The core programming language for the project.
-   [cite_start]**Streamlit**: For creating and deploying the web application. [cite: 1]
-   [cite_start]**Pandas**: For data manipulation and analysis. [cite: 1, 2]
-   [cite_start]**Numpy**: For numerical operations. [cite: 1, 2]
-   [cite_start]**Scikit-learn**: For implementing the machine learning model and preprocessing. [cite: 2]
-   [cite_start]**Joblib**: For saving and loading the trained model. [cite: 1]
-   [cite_start]**Plotly and Seaborn**: For creating interactive and static visualizations. [cite: 1, 2]
-   [cite_start]**Jupyter Notebook**: For model development and experimentation. [cite: 2]

---

## Installation and Setup

1.  **Clone the repository**:
    ```bash
    git clone [https://github.com/Khanz9664/Breast-Cancer-Prediction.git](https://github.com/Khanz9664/Breast-Cancer-Prediction.git)
    cd Breast-Cancer-Prediction
    ```

2.  **Create a virtual environment**:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

---

## Usage

To run the Streamlit application, execute the following command in your terminal:

```bash
streamlit run App.py
```
This will open the application in your default web browser. You can then use the sliders in the sidebar to input the tumor features and see the prediction.

-----

## Screenshots

![Screenshot](screenshots/s1.jpg)
![Screenshot](screenshots/s2.jpg)
![Screenshot](screenshots/s3.jpg)
![Screenshot](screenshots/s4.jpg)
![Screenshot](screenshots/s5.jpg)

-----

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

```
```
