import matplotlib.pyplot as plt
from utils.visualize import plot_confusion_matrix

# Load CNN predictions
cnn_preds = model.predict(test_ds)
cnn_acc = np.mean(np.argmax(cnn_preds, axis=1) == y_test)

# Compare accuracies
models = ['CNN', 'SVM', 'KNN', 'Random Forest']
accuracies = [cnn_acc, svm_acc, knn_acc, rf_acc]

plt.bar(models, accuracies)
plt.title('Model Comparison')
plt.ylabel('Accuracy')
plt.savefig('results/comparison.png')