# main.py
import sys
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
from sklearn.metrics import classification_report

# Local imports
from config import (
    SEED, RAW_DATA_DIR, SPLIT_DATA_DIR, MODEL_DIR, RESULTS_DIR, logger
)
from preprocess.clean_images import ImageCleaner
from preprocess.split_data import split_dataset
from utils.data_loader import create_data_generators, get_class_weights
from utils.evaluate import evaluate_model, compare_results, plot_confusion_matrix
from train.model import create_model
from train.train_cnn import train_model
from train.train_traditional_ml import train_traditional_models
from services.prediction_service import FruitDiseasePredictor
from preprocess.class_balancer import ClassBalancer

def initialize_environment():
    """Set up directories and random seeds"""
    Path(MODEL_DIR).mkdir(parents=True, exist_ok=True)
    Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)
    
    # Set all random seeds
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    logger.info(f"Environment initialized with seed {SEED}")

def prepare_data():
    """Full data preparation pipeline"""
    try:
        # Clean raw data
        logger.info("Cleaning raw images...")
        cleaner = ImageCleaner(RAW_DATA_DIR)
        cleaner.clean_dataset()
        cleaner.convert_cmyk_to_rgb()

        # Split dataset
        if not any(SPLIT_DATA_DIR.iterdir()):
            logger.info("Splitting dataset...")
            split_dataset()

    except Exception as e:
        logger.error(f"Data preparation failed: {str(e)}")
        sys.exit(1)

def train_and_evaluate_cnn(train_gen, val_gen, test_gen):
    """CNN training pipeline"""
    try:
        logger.info("\n=== CNN Training ===")
        model = create_model(train_gen.num_classes)
        model.summary(print_fn=lambda x: logger.info(x))

        logger.info("Calculating class weights...")
        class_weights = get_class_weights(train_gen)

        history = train_model(
            model=model,
            train_gen=train_gen,
            val_gen=val_gen,
            class_weights=class_weights
        )

        # Save training history
        pd.DataFrame(history.history).to_csv(RESULTS_DIR / 'training_history.csv')
        logger.info("CNN training completed")

        # Evaluate
        logger.info("Evaluating CNN...")
        cnn_report = evaluate_model(model, test_gen, 'cnn')
        
        # Plot confusion matrix
        y_true = test_gen.classes
        y_pred = np.argmax(model.predict(test_gen), axis=1)
        plot_confusion_matrix(y_true, y_pred, test_gen.class_indices.keys())
        
        return model, cnn_report

    except Exception as e:
        logger.error(f"CNN training failed: {str(e)}")
        raise

def train_and_evaluate_ml(train_gen, test_gen):
    """Traditional ML pipeline"""
    try:
        logger.info("\n=== Traditional ML Training ===")
        ml_models, (X_test, y_test) = train_traditional_models(train_gen, test_gen)
        
        # Evaluate ML models
        ml_reports = {}
        for name, model in ml_models.items():
            logger.info(f"Evaluating {name}...")
            y_pred = model.predict(X_test)
            ml_reports[name] = classification_report(y_test, y_pred, output_dict=True)
            
            # Save reports
            with open(RESULTS_DIR / f'{name.lower()}_report.txt', 'w') as f:
                f.write(classification_report(y_test, y_pred))

        return ml_reports

    except Exception as e:
        logger.error(f"ML training failed: {str(e)}")
        raise

def main():
    try:
        # Initialization
        initialize_environment()
        
        # Data preparation
        if not any(SPLIT_DATA_DIR.iterdir()):
            prepare_data()

        # Data loading
        logger.info("Creating data generators...")
        train_gen, val_gen, test_gen = create_data_generators()
        logger.info(f"Detected {train_gen.num_classes} classes")

        # Class balancing
        logger.info("Analyzing class balance...")
        #balancer = ClassBalancer(train_gen)
        #balancer.analyze_balance()
        
        # Get balanced generator
        #balanced_gen = balancer.apply_balance(strategy='augmentation')
        
        # CNN Training with balanced generator
        cnn_model, cnn_report = train_and_evaluate_cnn(train_gen, val_gen, test_gen)

        # Traditional ML Training
        ml_reports = train_and_evaluate_ml(train_gen, test_gen)

        # Performance comparison
        logger.info("Comparing results...")
        comparison = compare_results(cnn_report, ml_reports)
        pd.DataFrame(comparison).to_csv(RESULTS_DIR / 'model_comparison.csv')

        # Save final CNN model
        cnn_model.save(MODEL_DIR / 'cnn_model.h5')
        logger.info("CNN model saved as cnn_model.h5")

        # Save class names
        joblib.dump(train_gen.class_indices, MODEL_DIR / 'class_names.pkl')

        # Final test prediction
        logger.info("Testing prediction service...")
        predictor = FruitDiseasePredictor()
        test_image = next((RAW_DATA_DIR).glob('*/*.jpg'))
        prediction = predictor.predict(test_image)
        logger.info(f"Sample prediction: {prediction}")

        logger.info("Pipeline completed successfully!")

    except Exception as e:
        logger.error(f"Main pipeline failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()