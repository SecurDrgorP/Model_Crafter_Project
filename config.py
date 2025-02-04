from pathlib import Path
import logging

# Base directory
BASE_DIR = Path(__file__).parent

# Data paths
RAW_DATA_DIR = BASE_DIR / "data" / "FAVDD"
SPLIT_DATA_DIR = BASE_DIR / "data" / "split_dataset"
MODEL_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

# Model parameters
BATCH_SIZE = 16 # Number of images to process at once before updating weights and biases in the model data file ( defaults to 32) 
IMG_SIZE = (128, 128) # Image size for resizing the images to a consistent size ( defaults to (256, 256) )
EPOCHS = 5 # Number of times the model will cycle through the entire dataset ( defaults to 20 )
# Random seed for reproducibility
SEED = 42 # Seed for random number generation ( defaults to 42 ) ; default value is used for reproducibility
COLOR_MODEL = 'rgb' # Color model to use for images ( defaults to 'rgb' )
NUM_CHANNELS = 3 # Number of image channels ( defaults to 3 )

# Automatically determine number of classes
try:
    NUM_CLASSES = len([d for d in (SPLIT_DATA_DIR / "train").iterdir() if d.is_dir()])
except FileNotFoundError:
    NUM_CLASSES = None  # Will be set after data splitting

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)