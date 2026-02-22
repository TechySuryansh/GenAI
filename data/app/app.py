import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns

# Import preprocessing logic
import sys
# Add src to path if needed
# sys.path.append(os.path.join(os.getcwd(), 'data/src'))
# from preprocess import preprocess_data
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_PATH = os.path.join(BASE_DIR, "src")

import sys
sys.path.append(SRC_PATH)

from preprocess import preprocess_data
# Page configuration
st.set_page_config(
    page_title="ChurnGuard | AI Predictor",
    page_icon="🛡️",
    layout="wide"
)

# Custom CSS for a premium feel
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
    /* Target labels and values specifically for visibility */
    [data-testid="stMetricLabel"] > div {
        color: #4b4b4b !important;
    }
    [data-testid="stMetricValue"] > div {
        color: #1a1a1a !important;
    }
    .risk-high {
        color: #dc3545;
        font-weight: bold;
    }
    .risk-low {
        color: #28a745;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Load Artifacts
@st.cache_resource
def load_all_artifacts():
    base_path = "data/models"
    metrics_path = "data/results/metrics.json"
    
    # Load Models
    lr_model = joblib.load(os.path.join(base_path, "logistic_regression.pkl"))
    dt_model = joblib.load(os.path.join(base_path, "decision_tree.pkl"))
    
    # Load Preprocessing
    scaler = joblib.load(os.path.join(base_path, "scaler.pkl"))
    feature_names = joblib.load(os.path.join(base_path, "feature_names.pkl"))
    
    # Load Metrics
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
        
    return {"lr": lr_model, "dt": dt_model}, scaler, feature_names, metrics

try:
    models, scaler, feature_names, metrics = load_all_artifacts()
except Exception as e:
    st.error(f"⚠️ Error loading system artifacts: {e}. Please run the training pipeline first.")
    st.stop()

# --- Sidebar ---
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", width=100)
st.sidebar.title("ChurnGuard AI")
st.sidebar.markdown("---")

selected_model_key = st.sidebar.radio(
    "Select Prediction Model",
    ["Logistic Regression", "Decision Tree"],
    help="Logistic Regression offers better probability calibration, while Decision Trees are highly interpretable."
)
model_id = "logistic_regression" if selected_model_key == "Logistic Regression" else "decision_tree"
current_model = models["lr"] if model_id == "logistic_regression" else models["dt"]

# --- Main Interface ---
st.title("🛡️ Customer Retention Dashboard")
st.markdown(f"Currently using: **{selected_model_key}**")

tab1, tab2 = st.tabs(["🔮 Predict Churn", "📊 Model Performance"])

with tab1:
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("👤 Customer Profile")
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
            payment = st.selectbox("Payment Method", [
                "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
            ])
            monthly = st.number_input("Monthly Charges ($)", 18.0, 120.0, 70.0)
            total = st.number_input("Total Charges ($)", 0.0, 9000.0, float(tenure * monthly))

    with col2:
        st.subheader("📈 Prediction Results")
        
        # Prepare Input
        input_data = {
            'gender': gender, 'SeniorCitizen': 1 if senior == "Yes" else 0,
            'Partner': partner, 'Dependents': dependents, 'tenure': tenure,
            'PhoneService': phone, 'MultipleLines': multiple, 'InternetService': internet,
            'OnlineSecurity': security, 'OnlineBackup': backup, 'DeviceProtection': protection,
            'TechSupport': support, 'StreamingTV': tv, 'StreamingMovies': movies,
            'Contract': contract, 'PaperlessBilling': paperless, 'PaymentMethod': payment,
            'MonthlyCharges': monthly, 'TotalCharges': total
        }
        input_df = pd.DataFrame([input_data])
        
        # Preprocess
        processed_input = preprocess_data(input_df, is_training=False, scaler=scaler, feature_cols=feature_names)
        
        # Predict
        prob = current_model.predict_proba(processed_input)[0, 1]
        risk = "HIGH" if prob > 0.5 else "LOW"
        
        # Display
        res_col1, res_col2 = st.columns(2)
        with res_col1:
            st.metric("Churn Probability", f"{prob*100:.1f}%")
        with res_col2:
            color = "#dc3545" if risk == "HIGH" else "#28a745"
            st.markdown(f"#### Risk Level: <span style='color:{color}'>{risk}</span>", unsafe_allow_html=True)
            
        st.progress(prob)
        
        # Recommendations
        st.write("---")
        st.subheader("💡 Key Insights & Actions")
        if risk == "HIGH":
            st.warning("🚨 **Retention Alert:** This customer is likely to leave.")
            st.info("**Suggested Actions:**\n- Provide a personalized discount on Monthly Charges.\n- Offer a 1-year contract extension.\n- Check for technical issues in their service region.")
        else:
            st.success("✅ **Loyalty Confirmed:** Customer is unlikely to churn.")
            st.write("**Suggested Actions:**\n- Upsell new streaming features.\n- Ask for a referral or review.")

        # Feature Importance for this model
        st.write("---")
        st.subheader("📊 Model Feature Importance (Global)")
        if model_id == "logistic_regression":
            importance = pd.DataFrame({'Feature': feature_names, 'Importance': current_model.coef_[0]})
        else:
            importance = pd.DataFrame({'Feature': feature_names, 'Importance': current_model.feature_importances_})
        
        importance = importance.sort_values(by='Importance', ascending=False).head(10)
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.barplot(data=importance, x='Importance', y='Feature', palette='viridis')
        st.pyplot(fig)

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
        st.write("**Confusion Matrix**")
        cm_path = f"reports/figures/cm_{model_id}.png"
        if os.path.exists(cm_path):
            st.image(cm_path)
        else:
            st.info("Confusion matrix visual not found.")
            
    with v_col2:
        st.write("**ROC Curves (All Models)**")
        roc_path = "reports/figures/roc_curves.png"
        if os.path.exists(roc_path):
            st.image(roc_path)
        else:
            st.info("ROC Curve visual not found.")

st.sidebar.markdown("---")
st.sidebar.caption("© 2026 ChurnGuard AI | Milestone 1")
