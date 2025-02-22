import torch
import os
import time

# Device Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Path Configuration
BASE_PATH = r"D:/Programming/urine_interpret"
IMAGE_FOLDER = os.path.join(BASE_PATH, r"Datasets/Final Dataset I think/images")
MASK_FOLDER = os.path.join(BASE_PATH, r"Datasets/Final Dataset I think/labels")

# Training Hyperparameters
BATCH_SIZE = 8  # Increased batch size for better gradient estimation
NUM_EPOCHS = 150  # Increased number of epochs for better convergence
LEARNING_RATE = 5e-5  # Adjusted learning rate for more stable training
WEIGHT_DECAY = 1e-5  # Adjusted weight decay to prevent overfitting
ACCUMULATION_STEPS = 4  # Increased accumulation steps to simulate larger batch size
NUM_CLASSES = 11
PATIENCE = 15  # Increased patience for early stopping
IMAGE_SIZE = (256, 256)  # Slightly larger for better feature extraction

# Model Saving
def get_model_filename():
    models_dir = os.path.join(BASE_PATH, "models")
    os.makedirs(models_dir, exist_ok=True)  # Ensure the directory exists
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    return os.path.join(models_dir, f"unet_model_{timestamp}.pt")

def get_svm_filename():
    models_dir = os.path.join(BASE_PATH, "models")
    os.makedirs(models_dir, exist_ok=True)  # Ensure the directory exists
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    return os.path.join(models_dir, f"svm_model_{timestamp}.pkl")