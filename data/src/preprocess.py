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
    
    # 2. Convert TotalCharges to numeric
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    
    # 3. Handle missing values (median imputation)
    df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())
    
    return df

def preprocess_data(df):
    """
    Perform encoding, scaling and splitting.
    """
    # 1. Encode Target variable (Churn)
    le = LabelEncoder()
    df['Churn'] = le.fit_transform(df['Churn']) # No=0, Yes=1
    
    # 2. Identiy Numeric and Categorical columns
    numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    # 3. Scaling Numeric features
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    
    # 4. One-Hot Encoding for categorical features
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    
    # 5. Split into X and y
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    # Test script - usage from project root: python data/src/preprocess.py
    import os
    DATA_PATH = "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
    if not os.path.exists(DATA_PATH):
        # Fallback for running from inside data/src/
        DATA_PATH = "../WA_Fn-UseC_-Telco-Customer-Churn.csv"
    try:
        data = load_and_clean_data(DATA_PATH)
        X_train, X_test, y_train, y_test = preprocess_data(data)
        print("✅ Preprocessing Successful")
        print(f"X_train shape: {X_train.shape}")
        print(f"y_train distribution: \n{y_train.value_counts(normalize=True)}")
    except Exception as e:
        print(f"❌ Error: {e}")
