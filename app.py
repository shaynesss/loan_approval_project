import streamlit as st
import pandas as pd
import joblib

model = joblib.load("XGBmodel_approval_model.pkl")

st.title("Loan Approval Prediction")
st.write("Applicant Details:")

age = st.number_input("Age", min_value = 18, max_value = 90)
years_employed = st.number_input("Years Employed", min_value = 0)
annual_income = st.number_input("Annual Income", min_value = 0, step = 1000)
credit_score = st.number_input("Credit Score", min_value = 200, max_value = 850, value = 650)
current_debt = st.number_input("Current Debt", min_value = 0, step = 1000)
loan_amount = st.number_input("Loan Amount", min_value = 0, step = 1000)
occupation_status = st.selectbox("Occupation Status", ["Employed", "Student", "Self-Employed"])
loan_intent = st.selectbox("Loan Intent", ["Business", "Home Improvement", "Debt Consolidation", "Education", "Personal", "Medical"])
product_type = st.selectbox("Product Type", ["Credit Card", "Personal Loan", "Line of Credit"])


input_df = pd.DataFrame([{
    "age": age,
    "years_employed": years_employed,
    "annual_income": annual_income,
    "credit_score": credit_score,
    "current_debt": current_debt,
    "loan_amount": loan_amount,
    "occupation_status": occupation_status,
    "loan_intent": loan_intent,
    "product_type": product_type   
}])

st.sidebar.write("Application Preview:", input_df)

if st.button("Predict"):
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(input_df)[0][1]
        st.metric("APPROVAL PROBABILITY", f"{proba:.2%}")
        st.success("Approved" if proba >= 0.5 else "Rejected")
