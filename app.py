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

def predict_top3(symptoms_list):
    row = np.zeros(len(symptom_columns))

    for s in symptoms_list:
        s = s.strip().lower().replace(" ", "_")
        if s in symptom_columns:
            row[symptom_columns.index(s)] = 1

    input_df = pd.DataFrame([row], columns=symptom_columns)

    # probabilities for each disease
    probs = model.predict_proba(input_df)[0]
    classes = model.classes_

    # top 3 indices (highest probabilities)
    top_idx = np.argsort(probs)[-3:][::-1]

    top3 = [(classes[i], float(probs[i])) for i in top_idx]
    return top3

# ✅ Only ONE multiselect
selected = st.multiselect("Select symptoms:", symptom_columns)

if st.button("Predict"):
    if len(selected) == 0:
        st.error("Please select at least one symptom.")
    else:
        top3 = predict_top3(selected)

st.success("Top possible conditions:")

for disease, p in top3:
    st.write(f"• **{disease}** — {p*100:.1f}%")

st.warning("This is not a medical diagnosis.")
   
