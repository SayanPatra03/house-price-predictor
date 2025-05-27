import streamlit as st
import pickle
import numpy as np

# Load model and scaler
model = pickle.load(open('model/house_price_model.pkl', 'rb'))
scaler = pickle.load(open('model/scaler.pkl', 'rb'))

st.set_page_config(page_title="🏠 House Price Predictor")
st.title("🏠 Boston House Price Predictor")
st.write("Enter details to predict the house price")

# User inputs
rm = st.number_input("Average number of rooms (RM)", min_value=1.0, max_value=10.0, value=5.0)
lstat = st.number_input("% lower status population (LSTAT)", min_value=1.0, max_value=40.0, value=10.0)
ptratio = st.number_input("Pupil-teacher ratio (PTRATIO)", min_value=10.0, max_value=30.0, value=18.0)

if st.button("Predict Price"):
    input_data = np.array([[rm, lstat, ptratio]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    st.success(f"🏡 Predicted House Price: ${prediction[0]*1000:.2f}")