from preprocess import load_and_clean_data, preprocess_data
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import os
import json

def train_models(X_train, y_train):
    """
    Train Logistic Regression and Decision Tree models with GridSearch.
    """
    results = {}
    
    # 1. Logistic Regression
    print("Training Logistic Regression...")
    lr_params = {'C': [0.01, 0.1, 1, 10]}
    lr_grid = GridSearchCV(LogisticRegression(max_iter=1000), lr_params, cv=5, scoring='f1')
    lr_grid.fit(X_train, y_train)
    results['logistic_regression'] = lr_grid.best_estimator_
    
    # 2. Decision Tree
    print("Training Decision Tree...")
    dt_params = {
        'max_depth': [3, 5, 7, 10],
        'min_samples_split': [2, 5, 10]
    }
    dt_grid = GridSearchCV(DecisionTreeClassifier(random_state=42), dt_params, cv=5, scoring='f1')
    dt_grid.fit(X_train, y_train)
    results['decision_tree'] = dt_grid.best_estimator_
    
    return results

def evaluate_and_save_metrics(models, X_test, y_test, directory='data/results'):
    """
    Evaluate models and save metrics to JSON.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
        
    metrics_report = {}
    for name, model in models.items():
        y_pred = model.predict(X_test)
        metrics_report[name] = {
            'accuracy': round(accuracy_score(y_test, y_pred), 4),
            'precision': round(precision_score(y_test, y_pred), 4),
            'recall': round(recall_score(y_test, y_pred), 4),
            'f1': round(f1_score(y_test, y_pred), 4)
        }
    
    with open(os.path.join(directory, 'metrics.json'), 'w') as f:
        json.dump(metrics_report, f, indent=4)
    print(f"✅ Metrics saved to {directory}/metrics.json")

def save_artifacts(models, scaler, feature_names, directory='data/models'):
    """
    Save trained models, scaler, and features to disk.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    for name, model in models.items():
        path = os.path.join(directory, f"{name}.pkl")
        joblib.dump(model, path)
        print(f"✅ Model saved: {path}")
    
    joblib.dump(scaler, os.path.join(directory, "scaler.pkl"))
    joblib.dump(feature_names, os.path.join(directory, "feature_names.pkl"))
    print("✅ Preprocessing artifacts saved")

if __name__ == "__main__":
    DATA_PATH = "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
    
    # 1. Load and Preprocess
    data = load_and_clean_data(DATA_PATH)
    X_train, X_test, y_train, y_test, scaler, feature_names = preprocess_data(data)
    
    # 2. Train
    models = train_models(X_train, y_train)
    
    # 3. Evaluate and Save Metrics
    evaluate_and_save_metrics(models, X_test, y_test)
    
    # 4. Save Models and Preprocessing Artifacts
    save_artifacts(models, scaler, feature_names)
    
    # 5. Save test data for potential manual evaluation
    if not os.path.exists('data/processed'):
        os.makedirs('data/processed')
    X_test.to_csv('data/processed/X_test.csv', index=False)
    y_test.to_csv('data/processed/y_test.csv', index=False)
