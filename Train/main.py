import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from torch.utils.data import DataLoader
from ultralytics.nn.tasks import DetectionModel  # Import the required class
from datasets import UrineStripDataset
from models import UNetYOLO
from config import TRAIN_IMAGE_FOLDER, TRAIN_MASK_FOLDER, NUM_CLASSES, IMAGE_SIZE  # Import more constants
from train_unet_yolo import train_model  # Import train_model function
from train_svm import train_svm_classifier  # Import train_svm_classifier function

# Add DetectionModel to safe globals for model loading
torch.serialization.add_safe_globals([DetectionModel])

# Load dataset
def load_data(batch_size=16):
    dataset = UrineStripDataset(TRAIN_IMAGE_FOLDER, TRAIN_MASK_FOLDER)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Evaluate the model
def evaluate_model(model, dataloader):
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for images, targets, _ in dataloader:
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    return np.array(all_targets), np.array(all_preds)

# Calculate metrics
def calculate_metrics(targets, preds):
    precision = precision_score(targets, preds, average='weighted', zero_division=0)
    recall = recall_score(targets, preds, average='weighted', zero_division=0)
    f1 = f1_score(targets, preds, average='weighted', zero_division=0)
    return precision, recall, f1

# Main function
if __name__ == "__main__":
    # Load model
    model = UNetYOLO(in_channels=3, out_channels=NUM_CLASSES)
    
    # Fix path using raw string or forward slashes to avoid escape sequence warning
    weights_path = r'models/weights.pt'  # Use raw string with forward slashes
    
    try:
        # Load the model with the security fix
        model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
        print("Successfully loaded model weights!")
    except Exception as e:
        print(f"Error loading model weights: {e}")
        print("Trying alternative loading method...")
        try:
            # Alternative: explicitly set weights_only to False (less secure but may work)
            model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu'), weights_only=False))
            print("Successfully loaded model weights with weights_only=False!")
        except Exception as e2:
            print(f"Failed to load model: {e2}")
            print("Continuing with uninitialized model...")

    # Load data
    dataloader = load_data()

    # Evaluate model
    targets, preds = evaluate_model(model, dataloader)

    # Calculate metrics
    precision, recall, f1 = calculate_metrics(targets, preds)

    # Print metrics
    print(f'Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}')
    
    # Generate and save confusion matrix
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(targets, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap='Blues', values_format='d')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    plt.close()
    
    print("Evaluation complete! Confusion matrix saved to confusion_matrix.png")
    
    # Train UNet-YOLO model
    train_model(num_epochs=10, batch_size=16, learning_rate=0.001)
    
    # Train SVM classifier
    unet_model_path = r'models/weights.pt'
    train_svm_classifier(unet_model_path)
