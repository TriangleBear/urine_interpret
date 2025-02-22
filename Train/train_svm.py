import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from utils import extract_features_and_labels, save_svm_model
from datasets import UrineStripDataset
from config import IMAGE_FOLDER, MASK_FOLDER, device, get_svm_filename, NUM_CLASSES
from models import UNetYOLO
from ultralytics.nn.tasks import DetectionModel  # Import the required module
from icecream import ic  # Import icecream for debugging
from tqdm import tqdm  # Import tqdm for progress bar

def train_svm_rbf(unet_model_path, svm_model_path=None):
    ic("Loading dataset...")
    # Load the dataset
    dataset = UrineStripDataset(IMAGE_FOLDER, MASK_FOLDER)
    
    ic("Loading trained UNet model...")
    # Load the trained UNet model
    unet_model = UNetYOLO(3, NUM_CLASSES).to(device)
    torch.serialization.add_safe_globals([DetectionModel])  # Add DetectionModel to safe globals
    unet_model.load_state_dict(torch.load(unet_model_path, map_location=device, weights_only=False), strict=False)  # Set weights_only to False
    unet_model.eval()
    
    ic("Extracting features and labels...")
    # Extract features and labels
    features, labels = [], []
    for i in tqdm(range(len(dataset)), desc="Extracting features"):  # Add tqdm progress bar
        image, mask = dataset[i]
        image = image.to(device).unsqueeze(0)
        with torch.no_grad():
            output = unet_model(image)
        features.append(output.cpu().numpy().flatten())
        labels.append(mask.numpy().flatten())
    features = np.array(features)
    labels = np.array(labels).flatten().astype(int)  # Flatten and convert labels to integers
    labels = labels[:len(features)]  # Ensure labels have the same length as features
    ic(f"Extracted {len(features)} features and {len(labels)} labels.")
    ic(f"Unique labels: {np.unique(labels)}")
    ic(f"Label distribution: {np.bincount(labels, minlength=NUM_CLASSES)}")
    
    # Ensure all classes are represented in the labels
    if len(np.unique(labels)) == NUM_CLASSES and np.all(np.bincount(labels, minlength=NUM_CLASSES) > 0):
        ic("Splitting data into training and testing sets...")
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
        
        ic("Training SVM RBF classifier...")
        # Train the SVM RBF classifier
        svm_model = SVC(kernel='rbf', C=1, gamma='scale')
        svm_model.fit(X_train, y_train)
        
        ic("Evaluating SVM model...")
        # Evaluate the SVM model
        y_pred = svm_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred) * 100
        ic(f"SVM RBF Accuracy: {accuracy:.2f}%")
        
        # Save the SVM model
        if svm_model_path is None:
            svm_model_path = get_svm_filename()
        save_svm_model(svm_model, svm_model_path)
        ic(f"SVM model saved to {svm_model_path}")
    else:
        ic("Not all classes are represented in the dataset or some classes have zero samples. Skipping SVM training and evaluation.")

if __name__ == "__main__":
    unet_model_path = r"D:\Programming\urine_interpret\models\weights.pt"  # Path to the trained UNet model
    train_svm_rbf(unet_model_path)
