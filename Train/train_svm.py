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
from sklearn.preprocessing import StandardScaler  # Import StandardScaler
from sklearn.pipeline import Pipeline  # Import Pipeline
from sklearn.model_selection import GridSearchCV  # Import GridSearchCV
from imblearn.over_sampling import SMOTE  # Import SMOTE for oversampling

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
    features, labels = extract_features_and_labels(dataset, unet_model)
    ic(f"Extracted {len(features)} features and {len(labels)} labels.")
    ic(f"Unique labels: {np.unique(labels)}")
    ic(f"Label distribution: {np.bincount(labels, minlength=NUM_CLASSES)}")
    
    # Ensure all classes are represented in the labels
    if len(np.unique(labels)) == NUM_CLASSES and np.all(np.bincount(labels, minlength=NUM_CLASSES) > 0):
        ic("Balancing the dataset using SMOTE...")
        # Balance the dataset using SMOTE
        smote = SMOTE()
        features, labels = smote.fit_resample(features, labels)
        ic(f"Balanced label distribution: {np.bincount(labels, minlength=NUM_CLASSES)}")
        
        ic("Splitting data into training and testing sets...")
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
        
        ic("Training SVM RBF classifier with feature scaling and hyperparameter search...")
        # Create a pipeline with feature scaling and SVM
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('svm', SVC(kernel='rbf'))
        ])
        
        # Define hyperparameter grid
        param_grid = {
            'svm__C': [0.1, 1, 10, 100],
            'svm__gamma': [0.001, 0.01, 0.1, 1]
        }
        
        # Perform grid search with cross-validation
        grid_search = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        # Get the best model
        svm_model = grid_search.best_estimator_
        
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
