
import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.title("🩺 AI Symptom Checker Chatbot")
st.write("⚠️ Educational purpose only — not medical advice.")

# Load model
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

# Chat input
user_input = st.text_input("Enter symptoms separated by comma:")

if st.button("Predict"):
    symptoms = [x.strip() for x in user_input.split(",")]
    result = predict_disease(symptoms)

    st.success(f"Possible Disease: {result}")
    st.warning("This is not a medical diagnosis.")
