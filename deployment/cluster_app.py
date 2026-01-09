import os
import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Absolute paths to project files
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
kmeans_path = os.path.join(BASE_DIR, 'models', 'kmeans_cluster_model.pkl')
pca_path = os.path.join(BASE_DIR, 'models', 'pca_model.pkl')
data_path = os.path.join(BASE_DIR, 'data', 'heart_disease_clustered.csv')

# Load models and data
kmeans_model = joblib.load(kmeans_path)
pca_model = joblib.load(pca_path)
df = pd.read_csv(data_path)

# Streamlit UI
st.title('Heart Disease Risk Group Assessment')
st.write('Discover which risk group you belong to based on medical indicators.')

st.header('Patient Information')

sex_option = st.selectbox('Sex', ['Female', 'Male'])
sex = 1 if sex_option == 'Male' else 0

cp_option = st.selectbox('Chest Pain Type', [
    'Typical angina', 
    'Atypical angina', 
    'Non-anginal pain', 
    'Asymptomatic'
])
cp = ['Typical angina', 'Atypical angina', 'Non-anginal pain', 'Asymptomatic'].index(cp_option) + 1

exang_option = st.selectbox('Exercise Induced Angina?', ['No', 'Yes'])
exang = 1 if exang_option == 'Yes' else 0

slope_option = st.selectbox('Slope of Peak Exercise ST Segment', [
    'Upsloping', 
    'Flat', 
    'Downsloping'
])
slope = ['Upsloping', 'Flat', 'Downsloping'].index(slope_option) + 1

st.subheader('Medical Test Results')
thal = st.slider('Thalassemia (3=normal, 6=fixed defect, 7=reversible defect)', 3.0, 7.0, 3.0)
ca = st.slider('Number of Major Vessels Colored by Fluoroscopy (0-3)', 0.0, 3.0, 0.0)
oldpeak = st.slider('ST Depression Induced by Exercise (0-6.2)', 0.0, 6.2, 1.0)

if st.button('Assess My Risk Group'):
    input_data = pd.DataFrame([[thal, ca, exang, oldpeak, cp, slope, sex]],
                              columns=['thal', 'ca', 'exang', 'oldpeak', 'cp', 'slope', 'sex'])
    
    cluster = kmeans_model.predict(input_data)[0]
    input_2d = pca_model.transform(input_data)

    if cluster == 0:
        st.warning('HIGH-RISK GROUP')
        st.write('Your medical profile places you in the high-risk group.')
        st.write('74.8% of patients in this group have heart disease.')
        st.write('Key characteristics: Abnormal thalassemia, significant ST depression, more severe symptoms.')
    else:
        st.success('LOW-RISK GROUP')
        st.write('Your medical profile places you in the low-risk group.')
        st.write('22.6% of patients in this group have heart disease.')
        st.write('Key characteristics: Normal thalassemia, minimal ST depression, milder symptoms.')

    st.subheader('Your Position in Patient Groups')

    X = df.drop(['target', 'cluster'], axis=1, errors='ignore')
    X_2d = pca_model.transform(X)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(X_2d[:, 0], X_2d[:, 1], c=df['cluster'], cmap='viridis', alpha=0.3, label='All Patients')
    ax.scatter(input_2d[0, 0], input_2d[0, 1], c='red', marker='*', s=300, edgecolors='black', label='You')

    ax.set_xlabel('Principal Component 1 (Mainly Thalassemia)')
    ax.set_ylabel('Principal Component 2 (Mainly ST Depression)')
    ax.set_title('Your Position Among Patient Risk Groups')
    ax.legend()
    ax.grid(True)

    st.pyplot(fig)

st.header('About Risk Groups')
st.write("""
This tool uses unsupervised machine learning to group patients based on their medical characteristics.

The model identified two distinct groups:

- High-Risk Group: 74.8% heart disease prevalence
- Low-Risk Group: 22.6% heart disease prevalence

Key indicators include thalassemia status, ST depression, chest pain type, and number of blocked vessels.
""")