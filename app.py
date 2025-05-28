import streamlit as st
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# Load model and scaler
model = load_model('heart_failure_model.keras')
scaler = joblib.load('scaler.pkl')

# App title
st.title("🫀 Heart Failure Prediction App")
st.write("Enter patient details below to predict the likelihood of heart failure.")

# Input fields
age = st.number_input("Age", 0, 130, 50)
anaemia = st.selectbox("Anaemia", [0, 1])
creatinine_phosphokinase = st.number_input("Creatinine Phosphokinase", 0, 10000, 250)
diabetes = st.selectbox("Diabetes", [0, 1])
ejection_fraction = st.slider("Ejection Fraction (%)", 10, 80, 35)
high_blood_pressure = st.selectbox("High Blood Pressure", [0, 1])
platelets = st.number_input("Platelets", 50000, 1000000, 250000)
serum_creatinine = st.number_input("Serum Creatinine", 0.1, 10.0, 1.0)
serum_sodium = st.number_input("Serum Sodium", 100, 150, 135)
sex = st.selectbox("Sex", [0, 1])
smoking = st.selectbox("Smoking", [0, 1])
time = st.slider("Follow-up Time (Days)", 0, 300, 100)

# Predict
if st.button("Predict"):
    input_data = np.array([[age, anaemia, creatinine_phosphokinase, diabetes,
                            ejection_fraction, high_blood_pressure, platelets,
                            serum_creatinine, serum_sodium, sex, smoking, time]])
    
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)

    if prediction[0][0] > 0.5:
        st.error("⚠️ High Risk of Heart Failure")
    else:
        st.success("✅ Low Risk of Heart Failure")
