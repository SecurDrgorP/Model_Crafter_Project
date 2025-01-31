import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from pathlib import Path
from typing import Dict
from config import RESULTS_DIR, logger

def plot_comparison(cnn_preds: np.ndarray, 
                    y_true: np.ndarray,
                    ml_results: Dict[str, float],
                    save_dir: Path) -> None:
    try:
        plt.figure(figsize=(15, 6))
        
        # Model accuracy comparison
        plt.subplot(1, 2, 1)
        models = ['CNN'] + list(ml_results.keys())
        accuracies = [np.mean(cnn_preds == y_true)] + list(ml_results.values())
        plt.bar(models, accuracies)
        plt.title('Model Accuracy Comparison')
        plt.ylabel('Accuracy')
        plt.ylim(0, 1)
        
        # Confusion matrix for CNN
        plt.subplot(1, 2, 2)
        cm = confusion_matrix(y_true, cnn_preds)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('CNN Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        
        plt.tight_layout()
        plt.savefig(save_dir / 'model_comparison.png')
        plt.close()
        
        # Save classification report
        with open(save_dir / 'classification_report.txt', 'w') as f:
            f.write("CNN Classification Report:\n")
            f.write(classification_report(y_true, cnn_preds))
            
            f.write("\n\nTraditional ML Results:\n")
            for name, acc in ml_results.items():
                f.write(f"{name.upper()}: {acc:.4f}\n")
                
    except Exception as e:
        logger.error(f"Visualization failed: {str(e)}")
        raise