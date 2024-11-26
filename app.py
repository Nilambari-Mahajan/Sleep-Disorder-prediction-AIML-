import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load trained model
model = joblib.load('rf_sleep_model_with_bp.pkl')

# Load dataset to extract occupation options
dataset = pd.read_csv('Sleep_health_and_lifestyle_dataset.csv')

# Extract unique occupations
occupations = dataset['Occupation'].unique()

# Define input preprocessing function
def preprocess_input(data):
    # Encode categorical data
    encoder = LabelEncoder()
    categorical_columns = ['Gender', 'Occupation', 'BMI Category']
    for col in categorical_columns:
        data[col] = encoder.fit_transform(data[col])
    
    # Scale numerical data
    scaler = StandardScaler()
    numerical_columns = ['Age', 'Sleep Duration', 'Quality of Sleep',
                         'Physical Activity Level', 'Stress Level', 'Heart Rate', 
                         'Daily Steps', 'Systolic', 'Diastolic']
    data[numerical_columns] = scaler.fit_transform(data[numerical_columns])
    return data

# Streamlit app
st.title("Sleep Disorder Prediction")

# Input form
st.header("Input Features")
gender = st.selectbox("Gender", ["Male", "Female"])
age = st.number_input("Age", min_value=18, max_value=100, step=1)
occupation = st.selectbox("Occupation", occupations)
sleep_duration = st.number_input("Sleep Duration (hours)", min_value=0.0, max_value=24.0, step=0.1)
quality_of_sleep = st.slider("Quality of Sleep (1-10)", 1, 10)
physical_activity = st.slider("Physical Activity Level (1-100)", 1, 100)
stress_level = st.slider("Stress Level (1-10)", 1, 10)
bmi_category = st.selectbox("BMI Category", ["Underweight", "Normal", "Overweight", "Obese"])
heart_rate = st.number_input("Heart Rate (bpm)", min_value=40, max_value=200, step=1)
daily_steps = st.number_input("Daily Steps", min_value=0, max_value=20000, step=100)
systolic = st.number_input("Systolic Blood Pressure(Range: 100-150) ", min_value=80, max_value=200, step=1)
diastolic = st.number_input("Diastolic Blood Pressure(Range: 50-100)", min_value=40, max_value=120, step=1)

# Prediction button
if st.button("Predict"):
    # Create input data
    input_data = pd.DataFrame({
        "Gender": [gender],
        "Age": [age],
        "Occupation": [occupation],
        "Sleep Duration": [sleep_duration],
        "Quality of Sleep": [quality_of_sleep],
        "Physical Activity Level": [physical_activity],
        "Stress Level": [stress_level],
        "BMI Category": [bmi_category],
        "Heart Rate": [heart_rate],
        "Daily Steps": [daily_steps],
        "Systolic": [systolic],
        "Diastolic": [diastolic]
    })
    
    # Preprocess input data
    input_data = preprocess_input(input_data)
    
    # Make prediction
    prediction = model.predict(input_data)
    sleep_disorders = {0: "None", 1: "Insomnia", 2: "Sleep Apnea"}  # Adjust based on your dataset
    result = sleep_disorders[prediction[0]]
    st.success(f"Predicted Sleep Disorder: {result}")
