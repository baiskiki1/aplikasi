import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
import joblib

# Load model & encoder
model = load_model('bmi_model (1).h5')
encoder = joblib.load('encoder_bmi.pkl')

st.title("BMI Classification App (Deep Learning)")

weight = st.number_input("Berat Badan (kg):", min_value=30.0, max_value=150.0, step=0.5)
height = st.number_input("Tinggi Badan (meter):", min_value=1.0, max_value=2.5, step=0.01)

if st.button("Prediksi BMI"):
    data = np.array([[weight, height]])
    pred = model.predict(data)
    result = encoder.inverse_transform([np.argmax(pred)])
    st.success(f"Hasil Prediksi BMI Anda: {result[0]}")
