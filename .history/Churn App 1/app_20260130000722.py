import streamlit as st
import pandas as pd
import pickle
import os

# Load trained pipeline model
model_path = "churn_model.pkl"
if not os.path.exists(model_path):
    st.error("⚠️ Model file not found. Please ensure 'churn_model.pkl' exists in the app directory.")
    st.stop()

try:
    model = pickle.load(open(model_path, "rb"))
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

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

phone_service = st.selectbox("Phone Service", ["Yes", "No"])
multiple_lines = st.selectbox("Multiple Lines", ["No phone service", "No", "Yes"])
internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])

contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
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
    "PhoneService": phone_service,
    "MultipleLines": multiple_lines,
    "InternetService": internet_service,
    "OnlineSecurity": online_security,
    "OnlineBackup": online_backup,
    "DeviceProtection": device_protection,
    "TechSupport": tech_support,
    "StreamingTV": streaming_tv,
    "StreamingMovies": streaming_movies,
    "Contract": contract,
    "PaperlessBilling": paperless_billing,
    "PaymentMethod": payment,
    "MonthlyCharges": monthly,
    "TotalCharges": total
}])

# ---- Prediction ---- #

if st.button("Predict Churn"):
    try:
        prediction = model.predict(input_data)[0]

        if prediction == 1:
            st.error("❌ Customer is likely to churn")
        else:
            st.success("✅ Customer is likely to stay")
    except Exception as e:
        st.error(f"Error making prediction: {e}")
