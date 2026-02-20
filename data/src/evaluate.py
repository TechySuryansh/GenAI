import pandas as pd
import joblib
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import os

def evaluate_models(models_dir, X_test_path, y_test_path):
    """
    Load models and evaluate them on test data.
    """
    X_test = pd.read_csv(X_test_path)
    y_test = pd.read_csv(y_test_path).values.ravel()
    
    results = {}
    
    model_files = [f for f in os.listdir(models_dir) if f.endswith('.pkl')]
    
    for model_file in model_files:
        model_name = model_file.replace('.pkl', '')
        model_path = os.path.join(models_dir, model_file)
        model = joblib.load(model_path)
        
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        print(f"\n--- Evaluation for {model_name} ---")
        print(classification_report(y_test, y_pred))
        
        auc = roc_auc_score(y_test, y_prob)
        print(f"ROC-AUC Score: {auc:.4f}")
        
        results[model_name] = {
            'y_pred': y_pred,
            'y_prob': y_prob,
            'auc': auc
        }
    
    return results, y_test

def plot_visualizations(results, y_test, output_dir='reports/figures'):
    """
    Plot confusion matrices and ROC curves.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    for name, data in results.items():
        # 1. Confusion Matrix
        cm = confusion_matrix(y_test, data['y_pred'])
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {name}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.savefig(os.path.join(output_dir, f'cm_{name}.png'))
        plt.close()
        
        # 2. ROC Curve
        fpr, tpr, _ = roc_curve(y_test, data['y_prob'])
        plt.plot(fpr, tpr, label=f"{name} (AUC = {data['auc']:.2f})")
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'roc_curves.png'))
    plt.close()
    print(f"✅ Visualizations saved to {output_dir}")

if __name__ == "__main__":
    MODELS_DIR = "data/models"
    X_TEST = "data/processed/X_test.csv"
    Y_TEST = "data/processed/y_test.csv"
    
    try:
        eval_results, y_true = evaluate_models(MODELS_DIR, X_TEST, Y_TEST)
        plot_visualizations(eval_results, y_true)
    except Exception as e:
        print(f"❌ Error: {e}")
