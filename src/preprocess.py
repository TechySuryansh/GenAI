import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

def load_and_clean_data(file_path):
    """
    Load CSV, convert types, and handle missing values.
    """
    df = pd.read_csv(file_path)
    
    # 1. Drop customerID (non-predictive)
    df = df.drop('customerID', axis=1)
    
    # 3. Drop TotalCharges (Redundant with tenure and MonthlyCharges)
    df = df.drop('TotalCharges', axis=1)
    
    return df

def preprocess_data(df, is_training=True, scaler=None, feature_cols=None):
    """
    Perform encoding, scaling and splitting.
    If is_training is False, use provided scaler and feature_cols.
    """
    # 1. Encode Target variable (Churn) if training
    if is_training:
        le = LabelEncoder()
        df['Churn'] = le.fit_transform(df['Churn']) # No=0, Yes=1
    
    # 2. Identify Numeric and Categorical columns
    numeric_cols = ['tenure', 'MonthlyCharges']
    
    # 3. Scaling Numeric features
    for col in numeric_cols:
        if col not in df.columns:
            print(f"Warning: Numeric column {col} missing during inference. Adding as 0.")
            df[col] = 0
            
    if is_training:
        scaler = StandardScaler()
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    else:
        df[numeric_cols] = scaler.transform(df[numeric_cols])
    
    # 4. One-Hot Encoding
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    if 'customerID' in categorical_cols: categorical_cols.remove('customerID')
    if 'Churn' in categorical_cols: categorical_cols.remove('Churn')
    
    # Debug info for Streamlit logs
    if not is_training:
        print(f"Inference input columns: {df.columns.tolist()}")
        print(f"Detected categorical columns: {categorical_cols}")

    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    
    if is_training:
        # Save feature columns (excluding target)
        feature_cols = [c for c in df.columns if c != 'Churn']
        X = df[feature_cols]
        y = df['Churn']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        return X_train, X_test, y_train, y_test, scaler, feature_cols
    else:
        # Align features for inference
        for col in feature_cols:
            if col not in df.columns:
                df[col] = 0
        df = df[feature_cols]
        return df

if __name__ == "__main__":
    # Test script
    import os
    DATA_PATH = "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
        
    try:
        data = load_and_clean_data(DATA_PATH)
        X_train, X_test, y_train, y_test, scaler, feature_names = preprocess_data(data)
        print("Preprocessing Successful")
        print(f"X_train shape: {X_train.shape}")
        print(f"Features: {len(feature_names)}")
    except Exception as e:
        print(f"Error: {e}")
