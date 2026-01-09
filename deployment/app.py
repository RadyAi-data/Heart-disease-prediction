import subprocess
import sys

# Ensure joblib is installed
try:
    import joblib
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "joblib"])
import os
import joblib
import streamlit as st
import pandas as pd

# -------------------------------
# Load the trained pipeline
# -------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, '..', 'models', 'heart_disease_pipeline.pkl')

try:
    pipeline = joblib.load(MODEL_PATH)
except FileNotFoundError:
    st.error("Trained model not found. Make sure heart_disease_pipeline.pkl exists in models/")
    st.stop()

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("❤️ Heart Disease Risk Prediction")
st.write("Enter patient information below:")

# Collect raw input features (13 total)
input_data = {}
input_data['age'] = st.number_input("Age", min_value=1, max_value=120, value=50)
input_data['sex'] = st.selectbox("Sex", options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
input_data['cp'] = st.selectbox(
    "Chest Pain Type",
    options=[0, 1, 2, 3],
    format_func=lambda x: ["Typical angina", "Atypical angina", "Non-anginal pain", "Asymptomatic"][x]
)
input_data['trestbps'] = st.number_input("Resting Blood Pressure (mm Hg)", min_value=80, max_value=200, value=120)
input_data['chol'] = st.number_input("Serum Cholesterol (mg/dl)", min_value=100, max_value=600, value=200)
input_data['fbs'] = st.selectbox("Fasting Blood Sugar > 120 mg/dl", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
input_data['restecg'] = st.selectbox("Resting ECG results", options=[0, 1, 2])
input_data['thalach'] = st.number_input("Maximum Heart Rate Achieved", min_value=60, max_value=220, value=150)
input_data['exang'] = st.selectbox("Exercise Induced Angina", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
input_data['oldpeak'] = st.number_input("ST depression (oldpeak)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
input_data['slope'] = st.selectbox("Slope of ST segment", options=[0, 1, 2], format_func=lambda x: ["Upsloping", "Flat", "Downsloping"][x])
input_data['ca'] = st.selectbox("Number of major vessels (0–3)", options=[0, 1, 2, 3])
input_data['thal'] = st.selectbox("Thalassemia", options=[0, 1, 2], format_func=lambda x: ["Normal", "Fixed defect", "Reversible defect"][x])

# -------------------------------
# Prediction
# -------------------------------
if st.button("Predict Risk"):
    input_df = pd.DataFrame([input_data])  # convert dict to DataFrame

    try:
        prediction = pipeline.predict(input_df)[0]
        proba = pipeline.predict_proba(input_df)[0][1]  # probability of class 1

        if prediction == 1:
            st.error(f"High risk of heart disease! Probability: {proba:.1%}")
            st.write(" Recommendation: Consult a cardiologist for further evaluation.")
        else:
            st.success(f" Low risk of heart disease. Probability: {proba:.1%}")
            st.write(" Recommendation: Maintain a healthy lifestyle and regular checkups.")

    except Exception as e:
        st.error(f"Prediction error: {e}")

        st.write("Check that your input features match the training dataset.")

