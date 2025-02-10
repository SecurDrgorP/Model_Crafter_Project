import joblib
import pandas as pd
import numpy as np
import os
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import IncrementalPCA
from config import DATA_DIR, MODEL_DIR, logger
from utils.feature_extraction import extract_features
from utils.save_data import save_large_csv



def train_traditional_models(train_gen, test_gen):
    """
    Train and save traditional ML models on preprocessed features.
    
    The pipeline:
      1. Extract (or load) raw features.
      2. Fit a StandardScaler incrementally.
      3. Fit an IncrementalPCA (reducing to 100 features) incrementally.
      4. Transform both training and test data using the scaler and IPCA.
      5. Train models on the transformed (100-dimensional) features.
      6. Save both the preprocessor components and the trained models.
    """
    try:
        # Define paths for CSV files
        X_train_path = DATA_DIR / "X_train.csv"
        X_test_path = DATA_DIR / "X_test.csv"
        y_train_path = DATA_DIR / "y_train.csv"
        y_test_path = DATA_DIR / "y_test.csv"

        # Either load previously saved features or extract them
        if all(os.path.exists(path) for path in [X_train_path, X_test_path, y_train_path, y_test_path]):
            logger.info("Loading saved features...")
            X_train = pd.read_csv(X_train_path).values
            X_test = pd.read_csv(X_test_path).values
            y_train = pd.read_csv(y_train_path).values.ravel()
            y_test = pd.read_csv(y_test_path).values.ravel()
        else:
            logger.info("Extracting features...")
            X_train, y_train = extract_features(train_gen)
            X_test, y_test = extract_features(test_gen)

            # Save features to CSV in chunks
            logger.info("Saving features to CSV in chunks...")
            save_large_csv(pd.DataFrame(X_train), X_train_path)
            save_large_csv(pd.DataFrame(X_test), X_test_path)
            pd.DataFrame(y_train).to_csv(y_train_path, index=False)
            pd.DataFrame(y_test).to_csv(y_test_path, index=False)
            logger.info("Saving features completed")

        # Initialize the preprocessor components
        scaler = StandardScaler()
        ipca = IncrementalPCA(n_components=100, batch_size=1000)

        logger.info("Preprocessing features...")

        # Process training data in chunks for incremental fitting
        chunk_size = 10000  # Adjust based on system memory
        num_chunks = int(np.ceil(X_train.shape[0] / chunk_size))

        # Incrementally fit the StandardScaler on raw training data
        for i in range(num_chunks):
            start = i * chunk_size
            end = (i + 1) * chunk_size
            scaler.partial_fit(X_train[start:end])

        # Transform the whole training data with the fitted scaler
        X_train_scaled = scaler.transform(X_train)

        # Incrementally fit the IncrementalPCA on the scaled training data
        for i in range(num_chunks):
            start = i * chunk_size
            end = (i + 1) * chunk_size
            ipca.partial_fit(X_train_scaled[start:end])
        
        # Transform the entire training data into the reduced feature space
        X_train_final = ipca.transform(X_train_scaled)

        # Process the test data using the already fitted scaler and IPCA
        X_test_scaled = scaler.transform(X_test)
        X_test_final = ipca.transform(X_test_scaled)

        # Save the preprocessor components for use during inference
        joblib.dump(scaler, MODEL_DIR / "scaler.pkl")
        joblib.dump(ipca, MODEL_DIR / "ipca.pkl")

        # Define the models to be trained on the preprocessed (100-dim) data
        models = {
            'SVM': SVC(kernel='rbf', probability=True),
            'KNN': KNeighborsClassifier(n_neighbors=5),
            'RandomForest': RandomForestClassifier(n_estimators=100)
        }

        trained_models = {}
        for model_name, model in models.items():
            logger.info(f"Training {model_name.upper()}...")
            model.fit(X_train_final, y_train)
            joblib.dump(model, MODEL_DIR / f"{model_name}_model.pkl")
            trained_models[model_name] = model

        return trained_models, (X_test_final, y_test)

    except Exception as e:
        logger.error(f"Traditional ML training failed: {str(e)}")
        raise

