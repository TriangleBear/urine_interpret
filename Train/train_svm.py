import torch
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from utils import extract_features_and_labels, save_svm_model
from datasets import UrineStripDataset
from config import (
    TRAIN_IMAGE_FOLDER,
    TRAIN_MASK_FOLDER,
    VALID_IMAGE_FOLDER,
    VALID_MASK_FOLDER,
    device,
    get_svm_filename,
    NUM_CLASSES,
    get_model_folder
)
from models import UNetYOLO
from ultralytics.nn.tasks import DetectionModel
from icecream import ic
import os
import xgboost as xgb
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler

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
    10: 'strip'
}

def train_classifier(unet_model_path, model_path=None):
    ic("Loading training and validation datasets...")
    train_dataset = UrineStripDataset(TRAIN_IMAGE_FOLDER, TRAIN_MASK_FOLDER)
    valid_dataset = UrineStripDataset(VALID_IMAGE_FOLDER, VALID_MASK_FOLDER)
    
    ic("Loading trained UNet model...")
    unet_model = UNetYOLO(3, NUM_CLASSES).to(device)
    torch.serialization.add_safe_globals([DetectionModel])
    unet_model.load_state_dict(torch.load(unet_model_path, map_location=device), strict=False)
    unet_model.eval()
    
    ic("Extracting features and labels...")
    train_features, train_labels = extract_features_and_labels(train_dataset, unet_model)
    valid_features, valid_labels = extract_features_and_labels(valid_dataset, unet_model)
    
    ic(f"Training set: {len(train_features)} samples")
    ic(f"Validation set: {len(valid_features)} samples")
    
    # Print class distribution
    ic("Training set distribution:")
    for class_id, count in enumerate(np.bincount(train_labels, minlength=NUM_CLASSES)):
        if class_id < NUM_CLASSES:  # Add this check
            ic(f"{CLASS_NAMES[class_id]}: {count} samples")
    
    ic("Training XGBoost classifier...")
    
    # Scale features
    scaler = StandardScaler()
    train_features_scaled = scaler.fit_transform(train_features)
    valid_features_scaled = scaler.transform(valid_features)
    
    # Feature selection
    xgb_selector = xgb.XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5
    )
    selector = SelectFromModel(xgb_selector, prefit=False)
    train_features_selected = selector.fit_transform(train_features_scaled, train_labels)
    valid_features_selected = selector.transform(valid_features_scaled)
    
    # Train XGBoost with optimized parameters
    xgb_model = xgb.XGBClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=6,
        min_child_weight=1,
        gamma=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='multi:softprob',
        num_class=NUM_CLASSES,
        tree_method='gpu_hist',  # Use GPU acceleration if available
        eval_metric=['mlogloss', 'merror'],
        early_stopping_rounds=20
    )
    
    # Train with early stopping
    xgb_model.fit(
        train_features_selected, train_labels,
        eval_set=[(train_features_selected, train_labels),
                 (valid_features_selected, valid_labels)],
        verbose=True
    )
    
    # Evaluate
    valid_pred = xgb_model.predict(valid_features_selected)
    accuracy = accuracy_score(valid_labels, valid_pred) * 100
    ic(f"Validation Accuracy: {accuracy:.2f}%")
    
    # Print detailed classification report
    ic("Classification Report on Validation Set:")
    report = classification_report(
        valid_labels, valid_pred,
        target_names=[CLASS_NAMES[i] for i in range(NUM_CLASSES)],
        zero_division=0
    )
    ic(report)
    
    # Feature importance analysis
    feature_importance = xgb_model.feature_importances_
    ic("Top 10 most important features:")
    top_indices = np.argsort(feature_importance)[-10:]
    for idx in top_indices:
        ic(f"Feature {idx}: {feature_importance[idx]:.4f}")
    
    # Save the model and preprocessing objects
    if model_path is None:
        model_folder = get_model_folder()
        model_path = os.path.join(model_folder, "xgboost_model.pkl")
    
    model_data = {
        'model': xgb_model,
        'scaler': scaler,
        'selector': selector,
        'class_names': CLASS_NAMES,
        'feature_importance': feature_importance
    }
    save_svm_model(model_data, model_path)
    ic(f"Model saved to {model_path}")

if __name__ == "__main__":
    unet_model_path = r"D:\Programming\urine_interpret\models\weights.pt"
    train_classifier(unet_model_path)
