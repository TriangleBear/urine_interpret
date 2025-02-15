import torch
import os
import time

# Device Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Path Configuration
BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
print(BASE_PATH)
IMAGE_FOLDER = os.path.join(BASE_PATH, r"Datasets\Test test\images")
MASK_FOLDER = os.path.join(BASE_PATH, r"Datasets\Test test\labels")

# Training Hyperparameters
BATCH_SIZE = 16  # Reduced batch size
NUM_EPOCHS = 100
LEARNING_RATE = 1e-4  # Reduced learning rate
WEIGHT_DECAY = 1e-4
ACCUMULATION_STEPS = 8  # Increased accumulation steps
NUM_CLASSES = 11
PATIENCE = 5
IMAGE_SIZE = (256, 256)  # Reduced image size

# # Data Augmentation Parameters
# MEAN = [0.485, 0.456, 0.406]  # Default ImageNet mean
# STD = [0.229, 0.224, 0.225]   # Default ImageNet std

# Model Saving
def get_model_filename():
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    return f"unet_model_{timestamp}.pth"

def get_svm_filename():
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    return f"svm_model_{timestamp}.pkl"