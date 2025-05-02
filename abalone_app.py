import streamlit as st
import tensorflow as tf  # type: ignore
import numpy as np
import joblib

# Load scaler dan label encoder
scaler = joblib.load('scaler.pkl')
label_encoder = joblib.load('label_encoder.pkl')  # Optional, jika target dikategorikan

# Load model TFLite
interpreter = tf.lite.Interpreter(model_path="abalone_uci.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Judul Aplikasi
st.title("Abalone Age Prediction System")
st.write("Masukkan karakteristik abalone untuk memprediksi usianya (jumlah cincin).")

# Input form pengguna
sex = st.selectbox("Jenis Kelamin", options=["M", "F", "I"])
length = st.number_input("Panjang (Length)", min_value=0.0, max_value=100.0, value=100.0)
diameter = st.number_input("Diameter", min_value=0.0, max_value=100.0, value=0.4)
height = st.number_input("Tinggi", min_value=0.0, max_value=200.0, value=0.15)
whole_weight = st.number_input("Berat Utuh", min_value=0.0, max_value=200.0, value=1.0)
shucked_weight = st.number_input("Berat Daging", min_value=0.0, max_value=200.0, value=0.5)
viscera_weight = st.number_input("Berat Viscera", min_value=0.0, max_value=200.0, value=0.2)
shell_weight = st.number_input("Berat Cangkang", min_value=0.0, max_value=200.0, value=0.3)

if st.button("Prediksi Usia Abalone"):
    # Encode jenis kelamin
    sex_map = {'M': 2, 'F': 0, 'I': 1}
    sex_encoded = sex_map[sex]

    # Susun input
    input_data = np.array([[sex_encoded, length, diameter, height,
                            whole_weight, shucked_weight, viscera_weight, shell_weight]])
    input_scaled = scaler.transform(input_data).astype(np.float32)

    # Inference dengan TFLite
    interpreter.set_tensor(input_details[0]['index'], input_scaled)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]['index'])

    predicted_rings = prediction[0][0]  # Karena model regresi
    estimated_age = predicted_rings + 1.5  # Estimasi usia abalone (Rings + 1.5)

    st.success(f"Prediksi jumlah cincin: **{predicted_rings:.2f}**")
    st.info(f"Estimasi usia abalone: **{estimated_age:.2f} tahun**")
