import streamlit as st
import requests
import numpy as np
import joblib

# Load the trained model
model = joblib.load("prognosis_model.joblib")

# Set page title and icon
st.set_page_config(page_title="Disease Prediction App", page_icon="🏥", layout="centered")

# Add a healthcare logo
st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/5/55/Emblem_of_the_World_Health_Organization.svg/512px-Emblem_of_the_World_Health_Organization.svg.png", width=100)

# Title and description
st.title("🏥 AI-Powered Disease Prediction")
st.write("Enter your details and symptoms to predict the possible disease.")

# User inputs
name = st.text_input("👤 Enter your Name")
age = st.number_input("🎂 Enter your Age", min_value=1, max_value=120, step=1)

# Symptoms selection
st.write("### ✅ Select Symptoms")
symptom_list = [
    "Fatigue", "Headache", "High Fever", "Nausea", "Loss of Appetite",
    "Cough", "Shortness of Breath", "Chest Pain", "Dizziness", "Sore Throat"
]
selected_symptoms = st.multiselect("Choose symptoms", symptom_list)

# Convert symptoms into binary format (1 if selected, 0 if not)
symptoms_vector = [1 if symptom in selected_symptoms else 0 for symptom in symptom_list]

# Predict button
if st.button("🩺 Predict Disease"):
    if not name or not age or not selected_symptoms:
        st.error("⚠️ Please enter all details!")
    else:
        # Convert input into the required format
        input_data = np.array(symptoms_vector).reshape(1, -1)
        
        # Make prediction using the loaded model
        prediction = model.predict(input_data)[0]
        
        st.success(f"🩺 **Prediction for {name} (Age {age}):** {prediction}")

st.write("---")
st.write("🔬 This app uses a **machine learning model** trained on symptom data.")


    
    

           
