import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import sys

# Standardized Path Configuration
# This ensures accuracy whether run from root, src, or via a deployment bridge
CURRENT_FILE_PATH = os.path.abspath(__file__)
BASE_DIR = os.path.dirname(CURRENT_FILE_PATH)

# If this file is being executed from a subfolder, move up to find the root
if os.path.basename(BASE_DIR) in ["src", "app", "data"]:
    BASE_DIR = os.path.dirname(BASE_DIR)
if os.path.basename(BASE_DIR) in ["app", "data"]: # Handle nested case like data/app/
    BASE_DIR = os.path.dirname(BASE_DIR)
SRC_PATH = os.path.join(BASE_DIR, "src")
sys.path.append(SRC_PATH)

from preprocess import preprocess_data

# Page configuration
st.set_page_config(
    page_title="Customer Churn Prediction Dashboard",
    layout="wide"
)

# Application styling
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #007bff;
        color: white;
    }
    .stMetric {
        background-color: white;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    [data-testid="stMetricLabel"] > div {
        color: #4b4b4b !important;
    }
    [data-testid="stMetricValue"] > div {
        color: #1a1a1a !important;
    }
</style>
""", unsafe_allow_html=True)

# Resource loading
@st.cache_resource
def load_all_artifacts():
    models_path = os.path.join(BASE_DIR, "notebooks/models")
    metrics_path = os.path.join(BASE_DIR, "results/metrics.json")
    
    # Model deserialization
    lr_model = joblib.load(os.path.join(models_path, "logistic_regression.pkl"))
    dt_model = joblib.load(os.path.join(models_path, "decision_tree.pkl"))
    
    # Preprocessing artifacts
    scaler = joblib.load(os.path.join(models_path, "scaler.pkl"))
    feature_names = joblib.load(os.path.join(models_path, "feature_names.pkl"))
    
    # Performance metrics
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
        
    return {"lr": lr_model, "dt": dt_model}, scaler, feature_names, metrics

try:
    models, scaler, feature_names, metrics = load_all_artifacts()
except Exception as e:
    st.error(f"Error loading system artifacts: {e}")
    st.stop()

# Sidebar controls
st.sidebar.title("Navigation")
st.sidebar.markdown("---")

selected_model_key = st.sidebar.radio(
    "Select Prediction Model",
    ["Logistic Regression", "Decision Tree"]
)
model_id = "logistic_regression" if selected_model_key == "Logistic Regression" else "decision_tree"
current_model = models["lr"] if model_id == "logistic_regression" else models["dt"]

# Model performance summary
st.sidebar.markdown("### Model Scorecard")
m = metrics[model_id]
st.sidebar.info(f"""
**{selected_model_key}**
- **Accuracy:** {m['accuracy']*100:.1f}%
- **Precision:** {m['precision']*100:.1f}%
- **Recall:** {m['recall']*100:.1f}%
- **F1-Score:** {m['f1']*100:.1f}%
""")

# Dataset information
st.sidebar.markdown("### Dataset Overview")
st.sidebar.write("""
- **Total Samples:** 7,043
- **Churn Rate:** 26.5%
- **Target:** Customer Churn
""")

# Main dashboard interface
st.title("Customer Churn Prediction Dashboard")
tab1, tab2 = st.tabs(["Predict Churn", "Model Performance"])

with tab1:
    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("Customer Profile")
        with st.expander("Demographics", expanded=True):
            gender = st.selectbox("Gender", ["Male", "Female"])
            senior = st.selectbox("Senior Citizen", ["No", "Yes"])
            partner = st.selectbox("Partner", ["No", "Yes"])
            dependents = st.selectbox("Dependents", ["No", "Yes"])
            tenure = st.slider("Tenure (months)", 0, 72, 12)
        with st.expander("Services"):
            phone = st.selectbox("Phone Service", ["No", "Yes"])
            multiple = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
            internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
            security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
            backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
            protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
            support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
            tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
            movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
        with st.expander("Billing & Contract"):
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
            paperless = st.selectbox("Paperless Billing", ["No", "Yes"])
            payment = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
            monthly = st.number_input("Monthly Charges ($)", 18.0, 120.0, 70.0)

    with col2:
        st.subheader("Prediction Results")
        input_data = {
            'gender': gender, 'SeniorCitizen': 1 if senior == "Yes" else 0,
            'Partner': partner, 'Dependents': dependents, 'tenure': tenure,
            'PhoneService': phone, 'MultipleLines': multiple, 'InternetService': internet,
            'OnlineSecurity': security, 'OnlineBackup': backup, 'DeviceProtection': protection,
            'TechSupport': support, 'StreamingTV': tv, 'StreamingMovies': movies,
            'Contract': contract, 'PaperlessBilling': paperless, 'PaymentMethod': payment,
            'MonthlyCharges': monthly
        }
        input_df = pd.DataFrame([input_data])
        processed_input = preprocess_data(input_df, is_training=False, scaler=scaler, feature_cols=feature_names)
        
        prob = current_model.predict_proba(processed_input)[0, 1]
        
        # Risk classification logic
        if prob > 0.5: risk, color = "HIGH", "#dc3545"
        elif prob > 0.25: risk, color = "MEDIUM", "#ffc107"
        else: risk, color = "LOW", "#28a745"
        
        res_col1, res_col2 = st.columns(2)
        res_col1.metric("Churn Probability", f"{prob*100:.1f}%")
        res_col2.markdown(f"#### Risk Level: <span style='color:{color}'>{risk}</span>", unsafe_allow_html=True)
        st.progress(prob)
        
        st.write("---")
        st.subheader("Insights & Actions")
        if risk == "HIGH":
            st.warning("Retention Alert: This customer is highly likely to leave.")
            st.info("Suggested Actions:\n- Provide a personalized loyalty discount.\n- Offer a multi-year contract extension.\n- Conduct a proactive service quality check.")
        elif risk == "MEDIUM":
            st.info("Cautionary Alert: This customer shows signs of potential churn.")
            st.write("Suggested Actions:\n- Send a promotional retention offer.\n- Engage customer with a satisfaction survey.\n- Suggest beneficial service upgrades.")
        else:
            st.success("Loyalty Confirmed: Customer is unlikely to churn.")
            st.write("Suggested Actions:\n- Upsell premium value-added services.\n- Enroll in loyalty rewards program.")

with tab2:
    st.subheader(f"Performance Metrics: {selected_model_key}")
    m = metrics[model_id]
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    m_col1.metric("Accuracy", f"{m['accuracy']*100:.1f}%")
    m_col2.metric("Precision", f"{m['precision']*100:.1f}%")
    m_col3.metric("Recall", f"{m['recall']*100:.1f}%")
    m_col4.metric("F1-Score", f"{m['f1']*100:.1f}%")
    
    st.write("---")
    st.subheader("Visual Analysis")
    v_col1, v_col2 = st.columns(2)
    with v_col1:
        st.write("Confusion Matrix")
        cm_path = os.path.join(BASE_DIR, f"reports/cm_{model_id}.png")
        if os.path.exists(cm_path): st.image(cm_path)
    with v_col2:
        st.write("ROC Curves")
        roc_path = os.path.join(BASE_DIR, "reports/roc_curves.png")
        if os.path.exists(roc_path): st.image(roc_path)

st.sidebar.markdown("---")
st.sidebar.caption("Final Year Project Submission")
