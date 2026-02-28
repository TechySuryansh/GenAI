# ChurnGuard AI

ChurnGuard AI is a Customer Churn Predictor built with Python and Streamlit. It uses machine learning to predict whether a customer will churn based on demographic data, service subscriptions (internet, phone, etc.), and billing contracts. 

It provides an interactive, easy-to-use dashboard to profile customers and outputs the churn probability along with a "Risk Level" and actionable insights.

## Project Structure

```text
GenAI/
├── data/
│   ├── WA_Fn-UseC_-Telco-Customer-Churn.csv  # Raw dataset
│   ├── app/
│   │   └── app.py                            # Streamlit frontend dashboard
│   ├── models/                               # Trained models, scalers, and encoders (.pkl files)
│   ├── processed/                            # Train/test split data (.csv files)
│   ├── results/
│   │   └── metrics.json                      # Performance metrics
│   └── src/
│       ├── preprocess.py                     # Data cleaning, encoding, and scaling
│       ├── train.py                          # Model training pipelines (Logistic Regression, Decision Tree)
│       └── evaluate.py                       # Evaluation scripts to generate matrices & ROC curves
├── reports/
│   └── figures/                              # Generated CM and ROC plots (.png files)
└── requirements.txt                          # Python dependencies
```

## Features

- **Interactive UI**: A sleek, premium dashboard built using Streamlit.
- **Data Preprocessing**: Automatically handles missing values, performs one-hot encoding for categorical variables, and scales bounding numeric columns.
- **Machine Learning Models**: 
  - Logistic Regression Pipeline with hyperparameter tuning via GridSearchCV for probability calibration.
  - Decision Tree Classifier used for model interpretability.
- **Model Interpretability**: Dynamically visuals the global importance of each individual feature in predicting churn.
- **Evaluation Methods**: Generates test metrics (Accuracy, Precision, Recall, F1-Score), Confusion Matrices, and interactive ROC-AUC curves.

## Installation

1. Clone this repository or navigate to this project folder.
2. Ensure you have Python installed.
3. Install the required dependencies by running:
```bash
pip install -r requirements.txt
```

## Usage

### 1. Training the Model

If you have fresh data or want to re-train the underlying models, navigate to the main directory and run the training script:

```bash
python data/src/train.py
```

This will:
- Preprocess the dataset (`WA_Fn-UseC_-Telco-Customer-Churn.csv`).
- Save the trained pipelines and artifact scalers into `data/models/`.
- Output evaluation metrics into `data/results/metrics.json`.
- Save testing splits into `data/processed/`.

### 2. Evaluating the Model

Once trained, generate the ROC curves and confusion matrices by running:

```bash
python data/src/evaluate.py
```
This script will read from the generated `data/processed` test splits and drop visualization artifacts (`.png`) in the `reports/figures/` folder.

### 3. Launching the App

Run the Streamlit application to start profiling customers via the web interface:

```bash
streamlit run data/app/app.py
```

Open your local browser to view and interact with the ChurnGuard AI dashboard interface!
