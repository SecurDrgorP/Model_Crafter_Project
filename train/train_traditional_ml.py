# train/train_traditional_ml.py
import joblib
import numpy as np
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from config import MODEL_DIR, logger

def extract_features(generator):
    """Extract features from image generator with progress tracking"""
    X, y = [], []
    total_batches = len(generator)
    
    for batch_idx in range(total_batches):
        images, labels = next(generator)
        X.append(images.reshape(images.shape[0], -1))
        y.append(np.argmax(labels, axis=1))
        
        # Show progress every 10%
        if (batch_idx + 1) % (total_batches // 10) == 0:
            logger.info(f"Processed {batch_idx+1}/{total_batches} batches")
    
    logger.info("Feature extraction completed")
    return np.vstack(X), np.concatenate(y)

def train_traditional_models(train_gen, test_gen):
    """Train and save traditional ML models"""
    try:
        logger.info("Extracting features...")
        X_train, y_train = extract_features(train_gen)
        X_test, y_test = extract_features(test_gen)

        # Feature preprocessing pipeline
        preprocessor = make_pipeline(
            StandardScaler(),
            PCA(n_components=0.95, random_state=42)
        )
        
        logger.info("Preprocessing features...")
        X_train = preprocessor.fit_transform(X_train)
        X_test = preprocessor.transform(X_test)

        # Initialize models
        models = {
            'SVM': SVC(kernel='rbf', probability=True),
            'KNN': KNeighborsClassifier(n_neighbors=5),
            'RandomForest': RandomForestClassifier(n_estimators=100)
        }

        # Train and save models
        trained_models = {}
        for name, model in models.items():
            logger.info(f"Training {name}...")
            model.fit(X_train, y_train)
            joblib.dump(model, MODEL_DIR / f"{name.lower()}_model.pkl")
            trained_models[name] = model

        joblib.dump(preprocessor, MODEL_DIR / "preprocessor.pkl")
        return trained_models, (X_test, y_test)

    except Exception as e:
        logger.error(f"Traditional ML training failed: {str(e)}")
        raise