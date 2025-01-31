from pathlib import Path
import logging

# Base directory
BASE_DIR = Path(__file__).parent

# Data paths
RAW_DATA_DIR = BASE_DIR / "data" / "FAVDD"
SPLIT_DATA_DIR = BASE_DIR / "data" / "split_dataset"  # Renamed from DATA_DIR
MODEL_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

# Model parameters
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 15
SEED = 42

# Create directories if missing
for path in [RAW_DATA_DIR, SPLIT_DATA_DIR, MODEL_DIR, RESULTS_DIR]:
    path.mkdir(parents=True, exist_ok=True)

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)