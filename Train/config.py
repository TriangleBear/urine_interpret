import torch
import os

# Device Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Path Configuration
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
IMAGE_FOLDER = os.path.join(BASE_PATH, r"Datasets\Test test\images")
MASK_FOLDER = os.path.join(BASE_PATH, r"Datasets\Test test\labels")

# Training Hyperparameters
BATCH_SIZE = 8
NUM_EPOCHS = 100
LEARNING_RATE = 3e-4
NUM_CLASSES = 11
PATIENCE = 10

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