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

# Symptom categories
symptom_groups = {
    "Skin": [
        "itching", "skin_rash", "nodal_skin_eruptions",
        "skin_peeling", "blister", "silver_like_dusting",
        "red_sore_around_nose", "yellow_crust_ooze"
    ],
    "Fever / General": [
        "shivering", "chills", "fatigue", "sweating",
        "high_fever", "malaise", "restlessness", "lethargy"
    ],
    "Stomach / Digestive": [
        "stomach_pain", "acidity", "ulcers_on_tongue",
        "vomiting", "nausea", "abdominal_pain", "indigestion",
        "constipation", "diarrhoea"
    ],
    "Respiratory": [
        "continuous_sneezing", "cough", "breathlessness",
        "chest_pain", "phlegm", "runny_nose", "sinus_pressure"
    ],
    "Joint / Muscle": [
        "joint_pain", "muscle_wasting", "muscle_pain",
        "movement_stiffness", "swelling_joints", "stiff_neck"
    ],
    "Urinary": [
        "burning_micturition", "spotting_urination",
        "bladder_discomfort", "foul_smell_of_urine"
    ],
    "Head / Neurological": [
        "headache", "dizziness", "loss_of_balance",
        "lack_of_concentration"
    ]
}

# Keep only symptoms that exist in model
for group in symptom_groups:
    symptom_groups[group] = [s for s in symptom_groups[group] if s in symptom_columns]

def predict_top3(symptoms_list):
    row = np.zeros(len(symptom_columns))

    for s in symptoms_list:
        s = s.strip().lower().replace(" ", "_")
        if s in symptom_columns:
            row[symptom_columns.index(s)] = 1

    input_df = pd.DataFrame([row], columns=symptom_columns)

    probs = model.predict_proba(input_df)[0]
    classes = model.classes_

    top_idx = np.argsort(probs)[-3:][::-1]
    top3 = [(classes[i], float(probs[i])) for i in top_idx]
    return top3

# Store final selected symptoms across categories
if "final_selected" not in st.session_state:
    st.session_state.final_selected = []

# Category selection
category = st.selectbox("Select symptom category:", list(symptom_groups.keys()))

# Symptoms from selected category
current_selected = st.multiselect(
    f"Select symptoms from {category}:",
    symptom_groups[category]
)

# Add symptoms button
if st.button("Add Symptoms"):
    for symptom in current_selected:
        if symptom not in st.session_state.final_selected:
            st.session_state.final_selected.append(symptom)

# Show all selected symptoms
st.write("### Selected Symptoms")
if st.session_state.final_selected:
    st.write(", ".join(st.session_state.final_selected))
else:
    st.write("No symptoms added yet.")

# Clear symptoms button
if st.button("Clear All Symptoms"):
    st.session_state.final_selected = []

# Predict button
if st.button("Predict"):
    if len(st.session_state.final_selected) == 0:
        st.error("Please add at least one symptom.")
    else:
        top3 = predict_top3(st.session_state.final_selected)
        best_disease, best_prob = top3[0]

               confidence = round(min(best_prob * 100, 95.0), 1)

                confidence = round(min(best_prob * 100, 95.0), 1)
        confidence = round(min(best_prob * 100, 95.0), 1)

               confidence = round(min(best_prob * 100, 95.0), 1)

                confidence = round(min(best_prob * 100, 95.0), 1)

        if confidence < 60:
            st.warning("The symptoms you selected do not give a clear prediction. Please consult a healthcare professional.")
            st.warning("This is not a medical diagnosis.")
        else:
            st.success(f"Possible Disease: {best_disease}")

            st.write("### Prediction Level")
            st.progress(int(confidence))
            st.write(f"{confidence}%")

            st.warning("This is not a medical diagnosis.")
