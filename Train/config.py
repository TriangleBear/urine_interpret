import torch
import os
import time

# Create a flag to track if CUDA info has been printed
_CUDA_INFO_PRINTED = False

def get_device_info():
    """Get device information only once."""
    global _CUDA_INFO_PRINTED
    
    if not _CUDA_INFO_PRINTED:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            # Print CUDA details
            # print(f"CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")
            # print(f"CUDA Device Count: {torch.cuda.device_count()}")
            # print(f"CUDA Device Capability: {torch.cuda.get_device_capability(0)}")
            # print(f"CUDA Memory Allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
            # print(f"CUDA Memory Reserved: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")
            
            # Set safe CUDA optimization flags for RTX 4050
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.enabled = True
            
            # Only enable TF32 if architecture supports it (Ampere or newer)
            if torch.cuda.get_device_capability(0)[0] >= 8:
                # print("Enabling TensorFloat-32 for faster computation")
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
        else:
            print("CUDA is not available. Using CPU.")
        
        _CUDA_INFO_PRINTED = True
    
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Get device once during module import
device = get_device_info()

# Path Configuration
BASE_PATH = r"D:/Programming/urine_interpret/"
DATA_ROOT = os.path.join(BASE_PATH, r"Datasets/Split_70_20_10")
# IMPORTANT: Make sure we're using the modified label directory
# which should contain the missing Strip and Background classes
TRAIN_IMAGE_FOLDER = os.path.join(DATA_ROOT, "train/images")
TRAIN_MASK_FOLDER = os.path.join(DATA_ROOT, "train/labels")  # Use modified labels with all classes
VALID_IMAGE_FOLDER = os.path.join(DATA_ROOT, "valid/images")
VALID_MASK_FOLDER = os.path.join(DATA_ROOT, "valid/labels")  # Also modify validation labels
TEST_IMAGE_FOLDER = os.path.join(DATA_ROOT, "test/images")
TEST_MASK_FOLDER = os.path.join(DATA_ROOT, "test/labels")

# Create directories if they don't exist
for dir_path in [TRAIN_IMAGE_FOLDER, TRAIN_MASK_FOLDER, 
                VALID_IMAGE_FOLDER, VALID_MASK_FOLDER,
                TEST_IMAGE_FOLDER, TEST_MASK_FOLDER]:
    os.makedirs(dir_path, exist_ok=True)

# Training Hyperparameters
# Adjust batch size based on GPU memory
BATCH_SIZE = 2  # Further reduced batch size for better memory management

# For NVIDIA T4, use optimized settings
ACCUMULATION_STEPS = 4  # Adjust based on batch size
NUM_CLASSES = 12  # Updated number of classes to include strip and background
NUM_EPOCHS = 100
PATIENCE = 15
IMAGE_SIZE = (512, 512)  # Change size to 512x512

# New Learning Rate
LEARNING_RATE = 1e-4  # Adjusted learning rate

# Learning Rate Scheduler
LR_SCHEDULER_STEP_SIZE = 10
LR_SCHEDULER_GAMMA = 0.1

# Memory Management - safer approach
torch.cuda.empty_cache()

# Set safer CUDA memory allocation limits without expandable_segments
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:64'

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
    return os.path.join(model_folder, "svm_rbf_model.pkl")
