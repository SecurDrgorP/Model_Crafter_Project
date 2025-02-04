import joblib
import keras
import numpy as np
from PIL import Image
from pathlib import Path
from config import MODEL_DIR, IMG_SIZE, logger

class FruitDiseasePredictor:
    def __init__(self, model_type='cnn'):
        self.model_type = model_type
        self.class_names = None
        self.preprocessor = None
        self.model = None
        self.load_model()

    def load_model(self):
        """Load appropriate model based on type"""
        try:
            if self.model_type == 'cnn':
                self.model = keras.models.load_model(MODEL_DIR / 'cnn_model.h5')
                self.class_names = list(self.model.train_gen.class_indices.keys())
            else:
                self.preprocessor = joblib.load(MODEL_DIR / 'ml_preprocessor.pkl')
                self.model = joblib.load(MODEL_DIR / f'{self.model_type}_model.pkl')
                self.class_names = joblib.load(MODEL_DIR / 'class_names.pkl')
            
            logger.info(f"Loaded {self.model_type.upper()} model successfully")
        except Exception as e:
            logger.error(f"Model loading failed: {str(e)}")
            raise

    def preprocess_image(self, image_path):
        """Universal image preprocessing"""
        img = Image.open(image_path).convert('RGB')
        img = img.resize(IMG_SIZE)
        img_array = np.array(img) / 255.0
        
        if self.model_type != 'cnn':
            img_array = img_array.reshape(-1, *IMG_SIZE, 3)
            if self.preprocessor:
                img_array = self.preprocessor.transform(img_array.flatten().reshape(1, -1))
        
        return img_array

    def predict(self, image_path):
        """Make prediction on single image"""
        try:
            processed_img = self.preprocess_image(image_path)
            
            if self.model_type == 'cnn':
                probs = self.model.predict(processed_img[np.newaxis, ...])[0]
            else:
                probs = self.model.predict_proba(processed_img)[0]
            
            return {
                'predictions': sorted(
                    [{'class': name, 'probability': float(prob)} 
                     for name, prob in zip(self.class_names, probs)],
                    key=lambda x: x['probability'],
                    reverse=True
                )[:3]  # Return top 3 predictions
            }
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            return {'error': str(e)}