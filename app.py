import streamlit as st
import pandas as pd
import numpy as np
from src.pipeline import PredictPipeline, CustomData
from src.utils import load_object

st.title("Customer Churn Prediction")

st.sidebar.header("Enter Customer Details")

gender = st.sidebar.selectbox('Gender', ['Male', 'Female'])
SeniorCitizen = st.sidebar.selectbox('Senior Citizen', ['Yes', 'No'])
Partner = st.sidebar.selectbox('Partner', ['Yes', 'No'])
Dependents = st.sidebar.selectbox('Dependents', ['Yes', 'No'])
PhoneService = st.sidebar.selectbox('Phone Service', ['Yes', 'No'])
MultipleLines = st.sidebar.selectbox('Multiple Lines', ['Yes', 'No', 'No phone service'])
InternetService = st.sidebar.selectbox('Internet Service', ['DSL', 'Fiber optic', 'No'])
OnlineSecurity = st.sidebar.selectbox('Online Security', ['Yes', 'No', 'No internet service'])
OnlineBackup = st.sidebar.selectbox('Online Backup', ['Yes', 'No', 'No internet service'])
DeviceProtection = st.sidebar.selectbox('Device Protection', ['Yes', 'No', 'No internet service'])
TechSupport = st.sidebar.selectbox('Tech Support', ['Yes', 'No', 'No internet service'])
StreamingTV = st.sidebar.selectbox('Streaming TV', ['Yes', 'No', 'No internet service'])
StreamingMovies = st.sidebar.selectbox('Streaming Movies', ['Yes', 'No', 'No internet service'])
Contract = st.sidebar.selectbox('Contract', ['Month-to-month', 'One year', 'Two year'])
PaperlessBilling = st.sidebar.selectbox('Paperless Billing', ['Yes', 'No'])
PaymentMethod = st.sidebar.selectbox('Payment Method', ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'])
tenure = st.sidebar.slider('Tenure', 0, 80, 1)
MonthlyCharges = st.sidebar.slider('Monthly Charges', 0.0, 120.0, 1.0)
TotalCharges = st.sidebar.slider('Total Charges', 0.0, 9000.0, 1.0)

input_data = CustomData(
    gender=gender,
    SeniorCitizen=SeniorCitizen,
    Partner=Partner,
    Dependents=Dependents,
    PhoneService=PhoneService,
    MultipleLines=MultipleLines,
    InternetService=InternetService,
    OnlineSecurity=OnlineSecurity,
    OnlineBackup=OnlineBackup,
    DeviceProtection=DeviceProtection,
    TechSupport=TechSupport,
    StreamingTV=StreamingTV,
    StreamingMovies=StreamingMovies,
    Contract=Contract,
    PaperlessBilling=PaperlessBilling,
    PaymentMethod=PaymentMethod,
    tenure=tenure,
    MonthlyCharges=MonthlyCharges,
    TotalCharges=TotalCharges
)

input_df = input_data.get_data_as_data_frame()

if st.sidebar.button('Predict'):
    st.subheader("Customer Details")
    st.write(input_df)

    predict_pipeline = PredictPipeline()
    prediction = predict_pipeline.predict(input_df)
    st.write("Prediction Result")
    if prediction[0] == 1:
        st.error("DANGER!!! Customer will churn")

    else:
        st.success("SAFE: Customer will not churn")


