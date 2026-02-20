import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Page configuration
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
    layout="wide"
)

# Load Models and Scaler
@st.cache_resource
def load_artifacts():
    # Adjusted paths for running from project root: streamlit run data/app/app.py
    model_path = "data/models/logistic_regression.pkl"
    scaler_path = "data/models/scaler.pkl"
    
    # Fallback paths if run from inside data/app/
    if not os.path.exists(model_path):
        model_path = "../models/logistic_regression.pkl"
        scaler_path = "../models/scaler.pkl"
        
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

try:
    model, scaler = load_artifacts()
except Exception as e:
    st.error(f"Error loading models: {e}. Please ensure you've run the training script.")
    st.stop()

# Title and Description
st.title("🚀 Telco Customer Churn Prediction")
st.markdown("""
Predict whether a customer will leave based on their demographics and service usage. 
This app uses a **Logistic Regression** model (80% Accuracy).
""")

# Sidebar for User Inputs
st.sidebar.header("👤 Customer Information")

def user_input_features():
    # Categorical Inputs
    gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
    senior_citizen = st.sidebar.selectbox("Senior Citizen", ["No", "Yes"])
    partner = st.sidebar.selectbox("Partner", ["No", "Yes"])
    dependents = st.sidebar.selectbox("Dependents", ["No", "Yes"])
    
    st.sidebar.header("📞 Service Usage")
    phone_service = st.sidebar.selectbox("Phone Service", ["No", "Yes"])
    multiple_lines = st.sidebar.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
    internet_service = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    online_security = st.sidebar.selectbox("Online Security", ["No", "Yes", "No internet service"])
    online_backup = st.sidebar.selectbox("Online Backup", ["No", "Yes", "No internet service"])
    device_protection = st.sidebar.selectbox("Device Protection", ["No", "Yes", "No internet service"])
    tech_support = st.sidebar.selectbox("Tech Support", ["No", "Yes", "No internet service"])
    streaming_tv = st.sidebar.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
    streaming_movies = st.sidebar.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
    
    st.sidebar.header("💳 Billing & Contract")
    contract = st.sidebar.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless_billing = st.sidebar.selectbox("Paperless Billing", ["No", "Yes"])
    payment_method = st.sidebar.selectbox("Payment Method", [
        "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
    ])
    
    tenure = st.sidebar.slider("Tenure (months)", 0, 72, 12)
    monthly_charges = st.sidebar.slider("Monthly Charges ($)", 18.0, 120.0, 70.0)
    total_charges = st.sidebar.number_input("Total Charges ($)", min_value=0.0, value=tenure * monthly_charges)
    
    data = {
        'gender': gender,
        'SeniorCitizen': 1 if senior_citizen == "Yes" else 0,
        'Partner': partner,
        'Dependents': dependents,
        'tenure': tenure,
        'PhoneService': phone_service,
        'MultipleLines': multiple_lines,
        'InternetService': internet_service,
        'OnlineSecurity': online_security,
        'OnlineBackup': online_backup,
        'DeviceProtection': device_protection,
        'TechSupport': tech_support,
        'StreamingTV': streaming_tv,
        'StreamingMovies': streaming_movies,
        'Contract': contract,
        'PaperlessBilling': paperless_billing,
        'PaymentMethod': payment_method,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges
    }
    return pd.DataFrame([data])

input_df = user_input_features()

# Display Input Data
st.subheader("Selected Parameters")
st.write(input_df)

# Preprocessing for Prediction
def preprocess_input(df, scaler, model_columns):
    # 1. Scale Numeric
    numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    df[numeric_cols] = scaler.transform(df[numeric_cols])
    
    # 2. Categorical Encoding (dummies)
    # We must match the column order and names of the training data
    df_encoded = pd.get_dummies(df)
    
    # Reindex to match model expectation (add missing columns with 0)
    for col in model_columns:
        if col not in df_encoded.columns:
            df_encoded[col] = 0
            
    # Ensure correct order
    df_encoded = df_encoded[model_columns]
    return df_encoded

# Get the columns from the model
# (Logistic Regression coefficients have names if we use model.feature_names_in_)
try:
    model_columns = model.feature_names_in_
    processed_input = preprocess_input(input_df.copy(), scaler, model_columns)
    
    # Prediction
    prediction_prob = model.predict_proba(processed_input)[0, 1]
    prediction = model.predict(processed_input)[0]
    
    # Main Prediction Display
    st.divider()
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Prediction")
        if prediction == 1:
            st.error("🚨 **High Risk of Churn!**")
        else:
            st.success("✅ **Low Risk of Churn**")
            
        st.metric(label="Churn Probability", value=f"{prediction_prob*100:.1f}%")

    with col2:
        st.subheader("Risk Insights")
        if prediction_prob > 0.7:
            st.warning("Customer is likely to leave in the next 1-2 months.")
        elif prediction_prob > 0.4:
            st.info("Customer shows signs of dissatisfaction.")
        else:
            st.success("Customer loyalty seems strong.")

    # Phase 5: Feature Importance Visualization
    st.divider()
    st.subheader("📈 Key Churn Drivers (Global)")
    
    # Extract importance from LR coefficients
    importance = pd.DataFrame({
        'Feature': model_columns,
        'Importance': model.coef_[0]
    }).sort_values(by='Importance', ascending=False)
    
    # Plotting
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=importance.head(10), x='Importance', y='Feature', palette='RdYlGn_r')
    plt.title("Top Factors Increasing Churn Risk")
    st.pyplot(fig)
    
except Exception as e:
    st.error(f"Error during prediction: {e}")

st.markdown("---")
st.caption("Developed by Antigravity | Customer Churn ML Project Milestone 1")
