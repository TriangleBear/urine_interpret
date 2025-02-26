import torch
import os
import time

# Device Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Path Configuration
BASE_PATH = r"/content/urine_interpret"
DATA_ROOT = os.path.join(BASE_PATH, r"Datasets/Final Dataset I think")
TRAIN_IMAGE_FOLDER = os.path.join(DATA_ROOT, "train/images")
TRAIN_MASK_FOLDER = os.path.join(DATA_ROOT, "train/labels")
VALID_IMAGE_FOLDER = os.path.join(DATA_ROOT, "valid/images")
VALID_MASK_FOLDER = os.path.join(DATA_ROOT, "valid/labels")
TEST_IMAGE_FOLDER = os.path.join(DATA_ROOT, "test/images")
TEST_MASK_FOLDER = os.path.join(DATA_ROOT, "test/labels")

# Create directories if they don't exist
for dir_path in [TRAIN_IMAGE_FOLDER, TRAIN_MASK_FOLDER, 
                VALID_IMAGE_FOLDER, VALID_MASK_FOLDER,
                TEST_IMAGE_FOLDER, TEST_MASK_FOLDER]:
    os.makedirs(dir_path, exist_ok=True)

# Training Hyperparameters
BATCH_SIZE = 2  # Reduced from 4 to 2
NUM_EPOCHS = 150
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 1e-5
ACCUMULATION_STEPS = 16  # Increased from 8 to 16
NUM_CLASSES = 11  # Ensure this matches the number of classes in your dataset
PATIENCE = 15
IMAGE_SIZE = (224, 224)  # Reduced from 256x256

# Memory Management
torch.cuda.empty_cache()  # Clear CUDA cache
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'  # Remove expandable_segments
torch.backends.cudnn.benchmark = True

# Model Saving
def get_model_folder():
    models_dir = os.path.join(BASE_PATH, "models")
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    model_folder = os.path.join(models_dir, timestamp)
    os.makedirs(model_folder, exist_ok=True)  # Ensure the directory exists
    return model_folder

def get_model_filename():
    model_folder = get_model_folder()
    return os.path.join(model_folder, "unet_model.pt")

def get_svm_filename():
    model_folder = get_model_folder()
    return os.path.join(model_folder, "svm_model.pkl")
