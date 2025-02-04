import joblib
import numpy as np
from sklearn import logger
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
import tensorflow as tf

from config import IMG_SIZE, MODEL_DIR, SEED

def extract_features(generator):
    """Extract deep features using MobileNetV2"""
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(*IMG_SIZE, 3), # *IMG_SIZE unpacks tuple
        include_top=False,
        pooling='avg',
        weights='imagenet'
    )
    
    features, labels = [], []
    for _ in range(len(generator)):
        x, y = next(generator)
        features.append(base_model.predict(x, verbose=0))
        labels.append(np.argmax(y, axis=1))
        
    return np.vstack(features), np.concatenate(labels)

def train_traditional_models(train_gen, test_gen):
    """Train ML models with optimized parameters"""
    try:
        logger.info("Extracting CNN features...")
        X_train, y_train = extract_features(train_gen)
        X_test, y_test = extract_features(test_gen)

        # Reduced dimensionality while preserving class structure
        preprocessor = make_pipeline(
            StandardScaler(),
            PCA(n_components=50, random_state=SEED)  # Optimal for 28 classes
        )
        
        X_train = preprocessor.fit_transform(X_train)
        X_test = preprocessor.transform(X_test)

        # Optimized models for produce classification
        models = {
            'SVM': SVC(kernel='rbf', C=1.0, gamma='scale', probability=True),
            'KNN': KNeighborsClassifier(n_neighbors=3, weights='distance'),
            'RandomForest': RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                class_weight='balanced_subsample'
            )
        }

        # Cross-validate during training
        trained_models = {}
        for name, model in models.items():
            logger.info(f"Training {name} with 5-fold CV...")
            scores = cross_val_score(model, X_train, y_train, cv=5)
            logger.info(f"{name} CV Accuracy: {np.mean(scores):.2f} (±{np.std(scores):.2f})")
            
            model.fit(X_train, y_train)
            joblib.dump(model, MODEL_DIR / f"{name.lower()}_model.pkl")
            trained_models[name] = model

        joblib.dump(preprocessor, MODEL_DIR / "ml_preprocessor.pkl")
        return trained_models, (X_test, y_test)

    except Exception as e:
        logger.error(f"Traditional ML failed: {str(e)}")
        raise