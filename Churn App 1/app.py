import streamlit as st
import pandas as pd
import pickle

# Load trained pipeline model
model = pickle.load(open("churn_model.pkl", "rb"))

st.set_page_config(page_title="Customer Churn Predictor")

st.title("📊 Telco Customer Churn Prediction")
st.write("Enter customer details to predict churn")

# ---- User Inputs ---- #

gender = st.selectbox("Gender", ["Male", "Female"])
senior = st.selectbox("Senior Citizen", [0, 1])
partner = st.selectbox("Partner", ["Yes", "No"])
dependents = st.selectbox("Dependents", ["Yes", "No"])

tenure = st.number_input("Tenure (months)", min_value=0)
monthly = st.number_input("Monthly Charges", min_value=0.0)
total = st.number_input("Total Charges", min_value=0.0)

contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
payment = st.selectbox("Payment Method", [
    "Electronic check", 
    "Mailed check", 
    "Bank transfer (automatic)", 
    "Credit card (automatic)"
])

# ---- Create DataFrame (must match training columns) ---- #

input_data = pd.DataFrame([{
    "gender": gender,
    "SeniorCitizen": senior,
    "Partner": partner,
    "Dependents": dependents,
    "tenure": tenure,
    "MonthlyCharges": monthly,
    "TotalCharges": total,
    "Contract": contract,
    "PaymentMethod": payment
}])

# ---- Prediction ---- #

if st.button("Predict Churn"):
    prediction = model.predict(input_data)[0]

    if prediction == 1:
        st.error("❌ Customer is likely to churn")
    else:
        st.success("✅ Customer is likely to stay")
