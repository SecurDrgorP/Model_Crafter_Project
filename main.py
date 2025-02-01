# main.py
import sys
from pathlib import Path
import joblib
import tensorflow as tf
from sklearn.metrics import classification_report
from config import SEED, RAW_DATA_DIR, SPLIT_DATA_DIR, MODEL_DIR, RESULTS_DIR, logger
from utils.data_loader import create_data_generators
from train.model import create_model
from train.train_cnn import train_model
from train.train_traditional_ml import train_traditional_models
from utils.evaluate import compare_results, evaluate_model
from preprocess.split_data import split_dataset

def main():
    try:
        # Initialize environment
        logger.info("Initializing pipeline...")
        Path(MODEL_DIR).mkdir(parents=True, exist_ok=True)
        Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)

        # Data preparation
        if not any(RAW_DATA_DIR.iterdir()):
            logger.error(f"Dataset missing in {RAW_DATA_DIR}")
            sys.exit(1)
            
        if not any(SPLIT_DATA_DIR.iterdir()):
            logger.info("Splitting dataset...")
            split_dataset()

        # Data loading
        logger.info("Creating data generators...")
        train_gen, val_gen, test_gen = create_data_generators()
        
        # Model configuration
        num_classes = train_gen.num_classes
        logger.info(f"Detected {num_classes} classes")

        # CNN Training
        logger.info("Building CNN model...")
        model = create_model(num_classes)
        model.summary()
        
        logger.info("Training CNN...")
        tf.random.set_seed(SEED)
        history = train_model(model, train_gen, val_gen)
        
        # CNN Evaluation
        logger.info("Evaluating CNN...")
        cnn_report = evaluate_model(model, test_gen, 'cnn')

        # Traditional ML Training
        logger.info("Training traditional models...")
        ml_models, (X_test, y_test) = train_traditional_models(train_gen, test_gen)

        # Traditional ML Evaluation
        ml_reports = {}
        for name, model in ml_models.items():
            logger.info(f"Evaluating {name}...")
            y_pred = model.predict(X_test)
            ml_reports[name] = classification_report(y_test, y_pred, output_dict=True)
            
            # Save reports
            with open(RESULTS_DIR / f'{name.lower()}_report.txt', 'w') as f:
                f.write(classification_report(y_test, y_pred))

        # Performance comparison
        logger.info("Comparing results...")
        compare_results(cnn_report, ml_reports)

        # Finalize
        logger.info("Saving final models...")
        model.save(MODEL_DIR / 'cnn_model.h5')
        logger.info("Pipeline completed successfully!")

    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()