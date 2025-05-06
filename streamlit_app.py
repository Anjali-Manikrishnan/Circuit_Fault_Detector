import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Load and prepare data
@st.cache_data
def load_model():
    df = pd.read_csv("circuit_fault_data.csv")
    X = df[['Input Voltage (V)', 'Output Current (A)']]
    y = df['Fault Type']
    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)
    return model

model = load_model()

# UI
st.title("🔌 Circuit Fault Detector")
st.write("Enter the input voltage and output current to detect possible faults in the circuit.")

vin = st.number_input("Input Voltage (V)", min_value=0.0, max_value=10.0, step=0.1)
iout = st.number_input("Output Current (A)", min_value=0.0, max_value=10.0, step=0.1)

if st.button("Predict Fault"):
    prediction = model.predict([[vin, iout]])[0]
    st.success(f"✅ Predicted Fault: **{prediction}**")
