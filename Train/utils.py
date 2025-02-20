import torch
import numpy as np
import cv2
from skimage import color
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
import joblib
from config import device, NUM_CLASSES
import torchvision.transforms as T  # Add this import

def compute_mean_std(dataset):
    loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False)
    mean = 0.
    std = 0.
    for images, _ in loader:
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
    mean /= len(loader.dataset)
    std /= len(loader.dataset)
    return mean.tolist(), std.tolist()

def dynamic_normalization(tensor_image):
    if not isinstance(tensor_image, torch.Tensor):
        tensor_image = T.ToTensor()(tensor_image)
    mean = torch.mean(tensor_image, dim=[1, 2], keepdim=True)
    std = torch.std(tensor_image, dim=[1, 2], keepdim=True)
    std = torch.clamp(std, min=1e-6)
    normalize = T.Normalize(mean.flatten().tolist(), std.flatten().tolist())
    return normalize(tensor_image)

def extract_features_and_labels(dataset, model):
    features = []
    labels = []
    model.eval()
    for img, mask in dataset:
        with torch.no_grad():
            output = model(img.unsqueeze(0).to(device))
            pred_mask = output.argmax(1).squeeze().cpu().numpy()
        
        # Feature extraction logic
        lab_image = color.rgb2lab(img.permute(1, 2, 0).numpy())
        for class_id in range(NUM_CLASSES):
            if np.any(pred_mask == class_id):
                region = lab_image[pred_mask == class_id]
                features.append(region.mean(axis=0))
                labels.append(class_id)
    return np.array(features), np.array(labels)

def post_process_mask(mask):
    # Apply morphological operations to remove noise and fill gaps
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask

def train_svm_classifier(features, labels):
    param_grid = {'C': [0.1, 1, 10], 'gamma': [0.001, 0.01, 0.1]}
    grid_search = GridSearchCV(SVC(kernel='rbf'), param_grid, cv=3, error_score='raise')
    grid_search.fit(features, labels)
    return grid_search.best_estimator_

def save_svm_model(svm_model, filename):
    joblib.dump(svm_model, filename)