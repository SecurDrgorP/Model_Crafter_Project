# evaluate.py
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from config import RESULTS_DIR, logger

def evaluate_model(model, test_gen, model_type='cnn'):
    """Evaluate CNN model and save results"""
    try:
        logger.info(f"Evaluating {model_type.upper()} model...")
        y_pred = np.argmax(model.predict(test_gen), axis=1)
        y_true = test_gen.classes
        class_names = list(test_gen.class_indices.keys())

        # Classification report
        report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
        with open(RESULTS_DIR / f'{model_type}_report.txt', 'w') as f:
            f.write(classification_report(y_true, y_pred, target_names=class_names))

        # Confusion matrix
        plt.figure(figsize=(15,12))
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names)
        plt.title(f'{model_type.upper()} Confusion Matrix')
        plt.savefig(RESULTS_DIR / f'{model_type}_confusion_matrix.png')
        plt.close()

        return report

    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        raise

def compare_results(cnn_report, ml_reports):
    """Compare CNN and traditional ML results"""
    try:
        # Accuracy comparison
        accuracies = {
            'CNN': cnn_report['accuracy'],
            'SVM': ml_reports['SVM']['accuracy'],
            'KNN': ml_reports['KNN']['accuracy'],
            'RandomForest': ml_reports['RandomForest']['accuracy']
        }

        plt.figure(figsize=(10,6))
        plt.bar(accuracies.keys(), accuracies.values())
        plt.title('Model Accuracy Comparison')
        plt.ylabel('Accuracy')
        plt.ylim(0, 1)
        plt.savefig(RESULTS_DIR / 'accuracy_comparison.png')
        plt.close()

        # F1-score comparison
        f1_scores = {
            'CNN': cnn_report['macro avg']['f1-score'],
            'SVM': ml_reports['SVM']['macro avg']['f1-score'],
            'KNN': ml_reports['KNN']['macro avg']['f1-score'],
            'RandomForest': ml_reports['RandomForest']['macro avg']['f1-score']
        }

        plt.figure(figsize=(10,6))
        plt.bar(f1_scores.keys(), f1_scores.values())
        plt.title('Model F1-Score Comparison')
        plt.ylabel('F1 Score')
        plt.ylim(0, 1)
        plt.savefig(RESULTS_DIR / 'f1_comparison.png')
        plt.close()

    except Exception as e:
        logger.error(f"Comparison failed: {str(e)}")
        raise