import streamlit as st
import joblib
import numpy as np
import json
import os

#Load model and scaler
BASE = os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(BASE, 'best_model.pkl'))
scaler = joblib.load(os.path.join(BASE, 'scaler.pkl'))

with open(os.path.join(BASE, 'feature_names.json'), 'r', encoding='utf-8') as f:
    feature_names = json.load(f)
	
#Dasboard title
st.title('Student Performance Prediction System')
st.write('Enter student details below to Predict Pass or Fail')

#Input felds
gender = st.selectbox('Gender', ['Male', 'Female'])
age = st.number_input('Age', min_value=15, max_value=30, value=20) 
parental_education = st.number_input('Parental  Education Level', min_value=0, max_value=5, value=2)
family_income = st.number_input('Family Income', min_value=0.0, value=50000.00)
daily_study_hours = st.number_input('Daily Study Hours', min_value=0.0, max_value=24.0, value=4.0)
attendance_rate = st.number_input('Attendance Rate (%)', min_value=0.0, max_value=100.0, value=80.0)
sleep_hours = st.number_input('Sleep Hours', min_value=0.0, max_value=24.0, value=7.0)
stress_level = st.number_input('Stress Level', min_value=0.0, max_value=10.0, value=5.0)
motivation_score = st.number_input('Motivation Score', min_value=0.0, max_value=10.0, value=5.0 ) 
private_tutoring = st.selectbox('Private Tutoring', ['Yes', 'N0'])
internet_quality = st.number_input('Internet Quality', min_value=0.0, max_value=10.0, value=5.0) 
math_score = st.number_input('Math Score', min_value=0.0, max_value=100.0, value=70.0)
reading_score = st.number_input('Reading Score', min_value=0.0, max_value=100.0, value=70.0) 
writing_score = st.number_input('Writing Score', min_value=0.0, max_value=100.0, value=70.0) 

#Prepare input
gender_val = 1 if gender == 'Male' else 0
tutoring_val = 1 if private_tutoring =='Yes' else 0

input_data = np.array([[gender_val, age,
    parental_education, family_income,
    daily_study_hours, attendance_rate,
    sleep_hours, stress_level,
    motivation_score, tutoring_val, 
    internet_quality, math_score, 
    reading_score, writing_score]])
    
#Scale input
input_scaled = scaler.transform(input_data)

#Predict button
if st.button('Predict'): 
    prediction = model.predict(input_scaled)
    if prediction[0] == 1:
        st.success('Result:PASS')
        st.balloons() 
    else:
        st.error('Result: FAIL')
        st.write('This student may need academic support.')   
        
 
        
        




