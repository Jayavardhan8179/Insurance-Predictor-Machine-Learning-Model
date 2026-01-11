import streamlit as st
import numpy as np
import pandas as pd
import joblib

st.set_page_config(page_title="Insurance Charges Prediction App", layout="centered")
st.title("💰 Insurance Charges Prediction App")

st.write("Enter the details below to predict medical insurance charges.")

# ------------------------------------
# USER INPUT FIELDS (BASED ON YOUR DATASET)
# ------------------------------------

age = st.number_input(
    "Age",
    min_value=18, max_value=100, step=1
)

sex = st.selectbox(
    "Sex",
    ["male", "female"]
)

bmi = st.number_input(
    "BMI (Body Mass Index)",
    min_value=10.0, max_value=50.0, step=0.1
)

children = st.number_input(
    "Number of Children",
    min_value=0, max_value=5, step=1
)

smoker = st.selectbox(
    "Smoker",
    ["yes", "no"]
)

region = st.selectbox(
    "Region",
    ["southwest", "southeast", "northwest", "northeast"]
)

# ------------------------------------
# PREDICT BUTTON
# ------------------------------------

if st.button("Predict Insurance Charges"):
    model = joblib.load("RF_model.pkl")

    # Columns EXACTLY used during training
    columns = ['age', 'gender', 'bmi', 'children', 'smoker', 'region']

    # Build input row from user inputs
    sample_input = pd.DataFrame([[
        age,
        sex,
        bmi,
        children,
        smoker,
        region
    ]], columns=columns)

    # Predict
    prediction = model.predict(sample_input)

    st.success(f"Predicted Insurance Charges: ₹ {prediction[0]:,.2f}")
