import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

from config import RESULTS_DIR

def evaluate_model(model, test_gen, model_type='cnn'):
    """Comprehensive evaluation for both CNN and ML models"""
    if model_type == 'cnn':
        y_true = test_gen.classes
        y_pred = np.argmax(model.predict(test_gen), axis=1)
        y_proba = model.predict(test_gen)
    else:
        X_test, y_true = test_gen
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
    
    # Multi-class metrics
    report = classification_report(y_true, y_pred, output_dict=True)
    cm = confusion_matrix(y_true, y_pred)
    
    # Visualization
    plt.figure(figsize=(15,15))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix ({model_type.upper()})')
    plt.savefig(RESULTS_DIR / f'confusion_matrix_{model_type}.png')
    
    # ROC-AUC for binary health status if possible
    if y_proba is not None and len(np.unique(y_true)) == 2:
        report['roc_auc'] = roc_auc_score(y_true, y_proba[:,1])
    
    return report

def compare_results(cnn_report, ml_reports):
    """Generate comparative analysis"""
    comparison = {
        'CNN': {
            'accuracy': cnn_report['accuracy'],
            'f1_weighted': cnn_report['weighted avg']['f1-score']
        }
    }
    
    for name, report in ml_reports.items():
        comparison[name] = {
            'accuracy': report['accuracy'],
            'f1_weighted': report['weighted avg']['f1-score']
        }
    
    # Create comparison plot
    metrics_df = pd.DataFrame(comparison).T
    metrics_df.plot(kind='bar', figsize=(10,6))
    plt.title('Model Comparison')
    plt.ylabel('Score')
    plt.ylim(0.5, 1.0)
    plt.savefig(RESULTS_DIR / 'model_comparison.png')
    
    return comparison

def plot_confusion_matrix(y_true, y_pred, class_names):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(15,15))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.savefig(RESULTS_DIR / 'confusion_matrix.png')
    plt.close()
    