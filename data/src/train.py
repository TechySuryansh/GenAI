from preprocess import load_and_clean_data, preprocess_data
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
import joblib
import os

def train_models(X_train, y_train):
    """
    Train Logistic Regression and Decision Tree models with GridSearch.
    """
    results = {}
    
    # 1. Logistic Regression
    print("Training Logistic Regression...")
    lr_params = {'C': [0.01, 0.1, 1, 10, 100]}
    lr_grid = GridSearchCV(LogisticRegression(max_iter=1000), lr_params, cv=5, scoring='f1')
    lr_grid.fit(X_train, y_train)
    results['logistic_regression'] = lr_grid.best_estimator_
    print(f"Best LR Params: {lr_grid.best_params_}")
    
    # 2. Decision Tree
    print("Training Decision Tree...")
    dt_params = {
        'max_depth': [3, 5, 10, None],
        'min_samples_split': [2, 5, 10]
    }
    dt_grid = GridSearchCV(DecisionTreeClassifier(random_state=42), dt_params, cv=5, scoring='f1')
    dt_grid.fit(X_train, y_train)
    results['decision_tree'] = dt_grid.best_estimator_
    print(f"Best DT Params: {dt_grid.best_params_}")
    
    return results

def save_models(models, directory='data/models'):
    """
    Save trained models to disk.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    for name, model in models.items():
        path = os.path.join(directory, f"{name}.pkl")
        joblib.dump(model, path)
        print(f"✅ Model saved: {path}")

if __name__ == "__main__":
    DATA_PATH = "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
    
    # Load and Preprocess
    data = load_and_clean_data(DATA_PATH)
    X_train, X_test, y_train, y_test = preprocess_data(data)
    
    # Save test data for evaluation
    if not os.path.exists('data/processed'):
        os.makedirs('data/processed')
    X_test.to_csv('data/processed/X_test.csv', index=False)
    y_test.to_csv('data/processed/y_test.csv', index=False)
    print("✅ Test data saved for evaluation")
    
    # Train
    trained_models = train_models(X_train, y_train)
    
    # Save
    save_models(trained_models)
