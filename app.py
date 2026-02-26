import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="AI Symptom Checker", page_icon="🩺")

st.title("🩺 AI Symptom Checker Chatbot")
st.caption("⚠️ Educational purpose only — not medical advice.")

# Load model + columns
model = joblib.load("symptom_model.pkl")
symptom_columns = joblib.load("symptom_columns.pkl")

def predict_disease(symptoms_list):
    row = np.zeros(len(symptom_columns))

    for s in symptoms_list:
        s = s.strip().lower().replace(" ", "_")
        if s in symptom_columns:
            row[symptom_columns.index(s)] = 1

    input_df = pd.DataFrame([row], columns=symptom_columns)
    return model.predict(input_df)[0]

# ✅ Only ONE multiselect
selected = st.multiselect("Select symptoms:", symptom_columns)

if st.button("Predict"):
    if len(selected) == 0:
        st.error("Please select at least one symptom.")
    else:
        result = predict_disease(selected)
        st.success(f"Possible Disease: {result}")
        st.warning("This is not a medical diagnosis.")
