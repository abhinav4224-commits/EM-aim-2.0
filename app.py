import streamlit as st
import pickle
import os
import numpy as np

st.set_page_config(page_title="Earnings Manipulation Predictor")

# Always resolve files relative to this app.py file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")

# Load model
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# Load scaler
with open(SCALER_PATH, "rb") as f:
    scaler = pickle.load(f)

st.title("Earnings Manipulation Risk Predictor")

dsri = st.number_input("DSRI", value=1.0)
gmi  = st.number_input("GMI", value=1.0)
aqi  = st.number_input("AQI", value=1.0)
sgi  = st.number_input("SGI", value=1.0)
depi = st.number_input("DEPI", value=1.0)
sgai = st.number_input("SGAI", value=1.0)
tata = st.number_input("TATA", value=0.0)
lvgi = st.number_input("LVGI", value=1.0)

if st.button("Predict"):
    X = np.array([[dsri, gmi, aqi, sgi, depi, sgai, tata, lvgi]])
    X_scaled = scaler.transform(X)

    pred = model.predict(X_scaled)[0]
    prob = model.predict_proba(X_scaled)[0][1]

    if pred == 1:
        st.error(f"High Manipulation Risk (Probability: {prob:.2f})")
    else:
        st.success(f"Low Manipulation Risk (Probability: {prob:.2f})")
