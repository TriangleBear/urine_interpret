import torch
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from utils import extract_features_and_labels, save_svm_model
from datasets import UrineStripDataset
from config import (
    TRAIN_IMAGE_FOLDER,
    TRAIN_MASK_FOLDER,
    VALID_IMAGE_FOLDER,
    VALID_MASK_FOLDER,
    TEST_IMAGE_FOLDER,
    TEST_MASK_FOLDER,
    device,
    NUM_CLASSES,
    get_model_folder
)
from models import UNetYOLO
from ultralytics.nn.tasks import DetectionModel
from icecream import ic
import os
import cv2

# Define class names
CLASS_NAMES = {
    0: 'Bilirubin',
    1: 'Blood',
    2: 'Glucose',
    3: 'Ketone',
    4: 'Leukocytes',
    5: 'Nitrite',
    6: 'Protein',
    7: 'SpGravity',
    8: 'Urobilinogen',
    9: 'pH',
    10: 'strip',
    11: 'background'  # Add background class
}

def extract_polygon_features(image, mask, class_id):
    """Extract features for polygon-shaped bounding boxes."""
    binary_mask = (mask == class_id).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    features = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        region = image[y:y+h, x:x+w]
        features.append(region.mean(axis=(0, 1)))  # Mean color in the region
    if features:
        return np.mean(features, axis=0)  # Average features across all polygons
    else:
        return np.zeros(3)  # Return zeros if no polygons are found

def extract_features_from_segmentation(image, segmentation_map):
    """Extract features from segmentation map for SVM classification."""
    features = []
    
    # Get unique classes in the segmentation map
    unique_classes = np.unique(segmentation_map)
    
    # For each class, extract region properties
    for class_id in range(NUM_CLASSES):
        # Create binary mask for this class
        class_mask = (segmentation_map == class_id)
        
        # Skip if class not present
        if not class_mask.any():
            # Add zeros for absent classes
            features.extend([0, 0, 0, 0, 0, 0])
            continue
            
        # Calculate region properties
        # 1. Area ratio (area of class / total area)
        area_ratio = np.sum(class_mask) / segmentation_map.size
        features.append(area_ratio)
        
        # 2. Mean color within the class region
        for channel in range(image.shape[2]):
            mean_color = np.mean(image[:, :, channel][class_mask])
            features.append(mean_color)
            
        # 3. Mean position (center of mass)
        y_indices, x_indices = np.where(class_mask)
        if len(y_indices) > 0 and len(x_indices) > 0:
            mean_y = np.mean(y_indices) / segmentation_map.shape[0]
            mean_x = np.mean(x_indices) / segmentation_map.shape[1]
            features.extend([mean_y, mean_x])
        else:
            features.extend([0, 0])
    
    return np.array(features, dtype=np.float32)

def train_svm_classifier(unet_model_path, model_path=None):
    ic("Loading training, validation, and test datasets...")
    train_dataset = UrineStripDataset(TRAIN_IMAGE_FOLDER, TRAIN_MASK_FOLDER)
    valid_dataset = UrineStripDataset(VALID_IMAGE_FOLDER, VALID_MASK_FOLDER)
    test_dataset = UrineStripDataset(TEST_IMAGE_FOLDER, TEST_MASK_FOLDER)
    
    ic("Loading trained UNet model...")
    unet_model = UNetYOLO(3, NUM_CLASSES).to(device)
    torch.serialization.add_safe_globals([DetectionModel])
    unet_model.load_state_dict(torch.load(unet_model_path, map_location=device), strict=False)
    unet_model.eval()
    
    ic("Extracting features and labels...")
    train_features, train_labels = extract_features_and_labels(train_dataset, unet_model)
    valid_features, valid_labels = extract_features_and_labels(valid_dataset, unet_model)
    test_features, test_labels = extract_features_and_labels(test_dataset, unet_model)
    
    ic(f"Training set: {len(train_features)} samples")
    ic(f"Validation set: {len(valid_features)} samples")
    ic(f"Test set: {len(test_features)} samples")
    
    # Print class distribution
    ic("Training set distribution:")
    for class_id, count in enumerate(np.bincount(train_labels, minlength=NUM_CLASSES)):
        ic(f"{CLASS_NAMES[class_id]}: {count} samples")
    
    ic("Training SVM classifier with RBF kernel...")
    
    # Skip background class (11) when training SVM
    svm_data = [(features, label) for features, label in zip(train_features, train_labels) if label != NUM_CLASSES]
    if svm_data:
        svm_features, svm_labels = zip(*svm_data)
        svm_features = np.array(svm_features)
        svm_labels = np.array(svm_labels)
    else:
        print("Warning: No valid samples for SVM training after filtering background class")
        return
    
    # Scale features
    scaler = StandardScaler()
    train_features_scaled = scaler.fit_transform(svm_features)
    valid_features_scaled = scaler.transform(valid_features)
    test_features_scaled = scaler.transform(test_features)
    
    # Train SVM with RBF kernel
    svm_model = SVC(
        kernel='rbf',
        C=1.0,
        gamma='scale',
        probability=True
    )
    
    svm_model.fit(train_features_scaled, svm_labels)
    
    # Evaluate on validation set
    valid_pred = svm_model.predict(valid_features_scaled)
    valid_accuracy = accuracy_score(valid_labels, valid_pred) * 100
    ic(f"Validation Accuracy: {valid_accuracy:.2f}%")
    
    # Print detailed classification report for validation set
    ic("Classification Report on Validation Set:")
    valid_report = classification_report(
        valid_labels, valid_pred,
        target_names=[CLASS_NAMES[i] for i in range(NUM_CLASSES)],
        zero_division=0
    )
    ic(valid_report)
    
    # Evaluate on test set
    test_pred = svm_model.predict(test_features_scaled)
    test_accuracy = accuracy_score(test_labels, test_pred) * 100
    ic(f"Test Accuracy: {test_accuracy:.2f}%")
    
    # Print detailed classification report for test set
    ic("Classification Report on Test Set:")
    test_report = classification_report(
        test_labels, test_pred,
        target_names=[CLASS_NAMES[i] for i in range(NUM_CLASSES)],
        zero_division=0
    )
    ic(test_report)
    
    # Save the model and preprocessing objects
    if model_path is None:
        model_folder = get_model_folder()
        model_path = os.path.join(model_folder, "svm_rbf_model.pkl")
    
    model_data = {
        'model': svm_model,
        'scaler': scaler,
        'class_names': CLASS_NAMES
    }
    save_svm_model(model_data, model_path)
    ic(f"Model saved to {model_path}")

if __name__ == "__main__":
    unet_model_path = r"D:\Programming\urine_interpret\models\weights.pt"
    train_svm_classifier(unet_model_path)
