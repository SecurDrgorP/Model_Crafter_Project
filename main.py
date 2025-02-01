# main.py
import sys
from pathlib import Path
import keras
import tensorflow as tf
from config import BASE_DIR, RAW_DATA_DIR, SEED, SPLIT_DATA_DIR, MODEL_DIR, RESULTS_DIR, logger
from utils.data_loader import create_data_generators
from train.model import create_model
from train.train_cnn import train_model
from utils.evaluate import evaluate_model
from preprocess.split_data import split_dataset

def main():
    try:
        logger.info("Initializing pipeline...")
        
        # Ensure directories exist
        Path(MODEL_DIR).mkdir(parents=True, exist_ok=True)
        Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)

        # Data preparation
        if not any(RAW_DATA_DIR.iterdir()):
            logger.error(f"No data found in {RAW_DATA_DIR}. Please add dataset.")
            sys.exit(1)
            
        if not any(SPLIT_DATA_DIR.iterdir()):
            logger.info("Splitting dataset...")
            split_dataset()

        # Data loading
        logger.info("Creating data generators...")
        train_gen, val_gen, test_gen = create_data_generators()
        
        # Get number of classes from data
        num_classes = train_gen.num_classes
        logger.info(f"Detected {num_classes} classes in the dataset")
        
        # Model creation
        logger.info("Creating model...")
        model = create_model(num_classes)
        model.summary()
        
        # Training
        logger.info("Starting training...")
        logger.info("Training CNN...")
        tf.random.set_seed(SEED | 42)
        history = train_model(model, train_gen, val_gen)
        
        # Evaluation
        logger.info("Evaluating model...")
        evaluate_model(model, test_gen, history)
        
        # Save final model
        logger.info("Saving final model...")
        model.save(str(MODEL_DIR / 'final_model.keras'))
        
        logger.info("Pipeline completed successfully!")

    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        sys.exit(1)
        
    # TODO: Train with traditional ML algorithms (SVM, KNN, Random Forest)
    # TODO: Evaluate the model using classification report and confusion matrix
    # TODO: Save the models to the models directory
    # TODO: Save the classification report and confusion matrix to the results directory
    # TODO: Compare this results with CNN results



if __name__ == "__main__":
    main()