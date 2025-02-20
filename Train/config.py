import torch
import os
import time

# Device Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Path Configuration
BASE_PATH = r"D:\Programming\urine_interpret"
IMAGE_FOLDER = os.path.join(BASE_PATH, r"Datasets\Final Dataset I think\images")
MASK_FOLDER = os.path.join(BASE_PATH, r"Datasets\Final Dataset I think\labels")

# Training Hyperparameters
BATCH_SIZE = 4  # Increased batch size since T4 has more VRAM
NUM_EPOCHS = 100
LEARNING_RATE = 1e-4  # Adjusted learning rate
WEIGHT_DECAY = 1e-4  # Adjusted weight decay
ACCUMULATION_STEPS = 2  # Reduced since we increased batch size
NUM_CLASSES = 11
PATIENCE = 10
IMAGE_SIZE = (256, 256)  # Slightly larger for better feature extraction

# # Data Augmentation Parameters
# MEAN = [0.485, 0.456, 0.406]  # Default ImageNet mean
# STD = [0.229, 0.224, 0.225]   # Default ImageNet std

# Model Saving
def get_model_filename():
    models_dir = os.path.join(BASE_PATH, "models")
    os.makedirs(models_dir, exist_ok=True)  # Ensure the directory exists
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    return os.path.join(models_dir, f"unet_model_{timestamp}.pth")

def get_svm_filename():
    models_dir = os.path.join(BASE_PATH, "models")
    os.makedirs(models_dir, exist_ok=True)  # Ensure the directory exists
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    return os.path.join(models_dir, f"svm_model_{timestamp}.pkl")