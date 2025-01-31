import sys
from pathlib import Path
from utils.data_loader import DataLoader
from utils.split_data import split_dataset
from train.train_cnn import CNNTrainer
from config import BASE_DIR, RAW_DATA_DIR, SPLIT_DATA_DIR, MODEL_DIR, RESULTS_DIR, logger

def main():
    try:
        logger.info("Initializing pipeline...")
        
        # Data preparation
        if not any(RAW_DATA_DIR.iterdir()):
            logger.error(f"No data found in {RAW_DATA_DIR}. Please add dataset.")
            sys.exit(1)
            
        if not any(SPLIT_DATA_DIR.iterdir()):
            logger.info("Splitting dataset...")
            split_dataset()

        # Data loading
        logger.info("Loading datasets...")
        loader = DataLoader()
        train_ds, val_ds, test_ds = loader.load_datasets()

        # Model training
        logger.info("Starting CNN training...")
        num_classes = len([d for d in (SPLIT_DATA_DIR / "train").iterdir() if d.is_dir()])
        cnn_trainer = CNNTrainer(num_classes)
        cnn_trainer.train(train_ds, val_ds)

        logger.info("Pipeline completed successfully!")

    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()