import os
import joblib
import streamlit as st
import pandas as pd

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="HeartGuard AI",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# 2. LOAD MODEL (Cached for Performance)
# ==========================================
@st.cache_resource
def load_model():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, '..', 'models', 'heart_disease_pipeline.pkl')
    
    try:
        return joblib.load(model_path)
    except FileNotFoundError:
        return None

pipeline = load_model()

# ==========================================
# 3. SIDEBAR INFO
# ==========================================
with st.sidebar:
    st.title("❤️ HeartGuard AI")
    st.markdown("""
    **Medical Diagnostic Tool**
    
    This AI system estimates the likelihood of heart disease based on 13 clinical parameters.
    
    ---
    **Author:** RadyAi Data
    **Model:** Random Forest Classifier
    **Accuracy:** ~85% on Test Data
    """)
    st.info("ℹ️ Note: This tool is for screening purposes only and does not replace a doctor's diagnosis.")

# ==========================================
# 4. MAIN APP UI
# ==========================================
st.title("🫀 Patient Health Analysis")
st.markdown("### Enter Clinical Vitals")

if pipeline is None:
    st.error("❌ Model file not found! Please ensure `models/heart_disease_pipeline.pkl` exists.")
    st.stop()

# --- INPUT FORM (Organized in Columns) ---
with st.form("prediction_form"):
    st.markdown("#### 1. Patient Demographics & Vitals")
    c1, c2, c3, c4 = st.columns(4)
    
    with c1:
        age = st.number_input("Age", min_value=1, max_value=120, value=50)
    with c2:
        sex = st.selectbox("Sex", options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
    with c3:
        trestbps = st.number_input("Resting BP (mm Hg)", 80, 200, 120, help="Resting Blood Pressure")
    with c4:
        chol = st.number_input("Cholesterol (mg/dl)", 100, 600, 200)

    st.markdown("#### 2. Clinical History")
    c5, c6, c7 = st.columns(3)
    
    with c5:
        cp = st.selectbox(
            "Chest Pain Type", options=[0, 1, 2, 3],
            format_func=lambda x: ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"][x]
        )
    with c6:
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    with c7:
        restecg = st.selectbox("Resting ECG", options=[0, 1, 2], help="0: Normal, 1: ST-T wave abnormality, 2: Left ventricular hypertrophy")

    st.markdown("#### 3. Stress Test & Imaging")
    c8, c9, c10 = st.columns(3)
    
    with c8:
        thalach = st.number_input("Max Heart Rate", 60, 220, 150)
    with c9:
        exang = st.selectbox("Exercise Induced Angina", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    with c10:
        oldpeak = st.number_input("ST Depression", 0.0, 10.0, 1.0, step=0.1)

    c11, c12, c13 = st.columns(3)
    with c11:
        slope = st.selectbox("ST Slope", options=[0, 1, 2], format_func=lambda x: ["Upsloping", "Flat", "Downsloping"][x])
    with c12:
        ca = st.selectbox("Major Vessels (0-3)", options=[0, 1, 2, 3])
    with c13:
        thal = st.selectbox("Thalassemia", options=[0, 1, 2], format_func=lambda x: ["Normal", "Fixed Defect", "Reversible Defect"][x])

    # Submit Button
    submit_button = st.form_submit_button("🛡️ Analyze Risk Profile", type="primary")

# ==========================================
# 5. PREDICTION LOGIC
# ==========================================
if submit_button:
    # Dictionary must match the exact column names the model was trained on
    input_data = {
        'age': age, 'sex': sex, 'cp': cp, 'trestbps': trestbps, 'chol': chol,
        'fbs': fbs, 'restecg': restecg, 'thalach': thalach, 'exang': exang,
        'oldpeak': oldpeak, 'slope': slope, 'ca': ca, 'thal': thal
    }

    input_df = pd.DataFrame([input_data])

    try:
        # Get prediction and probability
        prediction = pipeline.predict(input_df)[0]
        proba = pipeline.predict_proba(input_df)[0][1]

        st.divider()
        
        # --- RESULTS DISPLAY ---
        col_res, col_graph = st.columns([1, 2])
        
        with col_res:
            if prediction == 1:
                st.error("🚨 **High Risk Detected**")
                st.metric("Risk Probability", f"{proba:.1%}", delta="High", delta_color="inverse")
            else:
                st.success("✅ **Low Risk Detected**")
                st.metric("Risk Probability", f"{proba:.1%}", delta="Safe", delta_color="normal")
        
        with col_graph:
            st.caption("Risk Confidence Meter")
            st.progress(proba)
            if prediction == 1:
                st.warning("⚠️ Recommendation: Patient shows signs indicative of heart disease. Please refer to a cardiologist immediately.")
            else:
                st.success("👍 Recommendation: Maintain a healthy lifestyle. Regular annual checkups are advised.")

    except Exception as e:
        st.error(f"❌ An error occurred during prediction: {e}")
