import joblib
import numpy as np
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from typing import Dict, Any
from config import MODEL_DIR, logger
from utils.feature_extraction import FeatureExtractor

class TraditionalMLTrainer:
    def __init__(self):
        self.models = {
            'svm': SVC(probability=True),
            'knn': KNeighborsClassifier(),
            'random_forest': RandomForestClassifier()
        }
        
        self.param_grid = {
            'svm': {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']},
            'knn': {'n_neighbors': [3, 5, 7]},
            'random_forest': {'n_estimators': [50, 100, 200]}
        }

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, Any]:
        try:
            trained_models = {}
            for name, model in self.models.items():
                logger.info(f"Training {name.upper()}...")
                
                # Hyperparameter tuning
                grid_search = GridSearchCV(
                    estimator=model,
                    param_grid=self.param_grid[name],
                    cv=3,
                    n_jobs=-1
                )
                
                grid_search.fit(X_train, y_train)
                best_model = grid_search.best_estimator_
                
                # Save model
                model_path = MODEL_DIR / f'{name}_model.pkl'
                joblib.dump(best_model, model_path)
                
                trained_models[name] = {
                    'model': best_model,
                    'best_params': grid_search.best_params_,
                    'best_score': grid_search.best_score_
                }
                logger.info(f"{name.upper()} training completed. Best params: {grid_search.best_params_}")
            
            return trained_models
            
        except Exception as e:
            logger.error(f"Traditional ML training failed: {str(e)}")
            raise