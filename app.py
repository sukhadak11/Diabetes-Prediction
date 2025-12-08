import streamlit as st
import numpy as np
import pickle

# Load model and scaler
model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

st.title("Diabetes Prediction using Logistic Regression")
st.write("Enter patient details below to predict diabetes risk.")

# Input fields
pregnancies = st.number_input('Pregnancies', min_value=0, max_value=20, value=0)
glucose = st.number_input('Glucose Level', min_value=0, max_value=300, value=120)
blood_pressure = st.number_input('Blood Pressure', min_value=0, max_value=200, value=70)
skin_thickness = st.number_input('Skin Thickness', min_value=0, max_value=100, value=20)
insulin = st.number_input('Insulin Level', min_value=0, max_value=900, value=80)
bmi = st.number_input('BMI', min_value=0.0, max_value=70.0, value=25.0)
dpf = st.number_input('Diabetes Pedigree Function', min_value=0.0, max_value=3.0, value=0.3)
age = st.number_input('Age', min_value=1, max_value=120, value=30)

# Prediction
if st.button("Predict"):
    # Prepare input
    input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness,
                            insulin, bmi, dpf, age]])
    
    # Scale input data
    scaled_data = scaler.transform(input_data)
    
    # Predict
    prediction = model.predict(scaled_data)[0]
    
    if prediction == 1:
        st.error("The model predicts: **High Risk of Diabetes**")
    else:
        st.success("The model predicts: **Low Risk of Diabetes**")