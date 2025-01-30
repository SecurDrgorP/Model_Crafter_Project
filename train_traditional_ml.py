import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from utils.data_loader import load_data
from config import *

# Load data and flatten images
def extract_features(dataset):
    images, labels = [], []
    for img, label in dataset.unbatch():
        images.append(img.numpy().flatten())
        labels.append(np.argmax(label.numpy()))
    return np.array(images), np.array(labels)

train_ds, _, test_ds = load_data(DATA_DIR, IMG_SIZE, BATCH_SIZE)
X_train, y_train = extract_features(train_ds)
X_test, y_test = extract_features(test_ds)

# Train SVM
svm = SVC()
svm.fit(X_train, y_train)
print("SVM Accuracy:", svm.score(X_test, y_test))

# Train KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
print("KNN Accuracy:", knn.score(X_test, y_test))

# Train Random Forest
rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_train)
print("RF Accuracy:", rf.score(X_test, y_test))