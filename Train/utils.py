import torch
import numpy as np
import cv2
from skimage import color, feature
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
import joblib
from config import device, NUM_CLASSES
import torchvision.transforms as T
from scipy import stats

def compute_mean_std(dataset):
    loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False)
    mean = 0.
    std = 0.
    total_images = len(loader.dataset)
    if total_images == 0:
        raise ValueError("Dataset is empty.")
    for images, _ in loader:
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
    mean /= total_images
    std /= total_images
    return mean.tolist(), std.tolist()

def dynamic_normalization(tensor_images):
    if len(tensor_images.shape) == 4:  # Batch of images
        mean = torch.mean(tensor_images, dim=[0, 2, 3], keepdim=True)
        std = torch.std(tensor_images, dim=[0, 2, 3], keepdim=True)
    else:  # Single image
        mean = torch.mean(tensor_images, dim=[1, 2], keepdim=True)
        std = torch.std(tensor_images, dim=[1, 2], keepdim=True)
    std = torch.clamp(std, min=1e-6)
    normalize = T.Normalize(mean.flatten().tolist(), std.flatten().tolist())
    return normalize(tensor_images)

def extract_features_and_labels(dataset, unet_model):
    features_list = []
    labels_list = []
    
    for image, label in dataset:
        # Ensure label is a single integer
        if isinstance(label, (list, tuple, np.ndarray, torch.Tensor)):
            label = int(label[0]) if len(label) > 0 else 0
        else:
            label = int(label)
        
        # Convert tensor to numpy and change format from (C,H,W) to (H,W,C)
        if isinstance(image, torch.Tensor):
            image_np = image.permute(1, 2, 0).numpy()
        else:
            image_np = np.array(image)
        
        # Ensure the image is in the correct format
        if image_np.shape[2] != 3:
            image_np = np.transpose(image_np, (1, 2, 0))
        
        # Convert to different color spaces
        lab_image = color.rgb2lab(image_np)
        hsv_image = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)
        gray_image = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        
        # Get UNet predictions
        with torch.no_grad():
            if isinstance(image, torch.Tensor):
                image_tensor = image.unsqueeze(0)
            else:
                image_tensor = T.ToTensor()(image).unsqueeze(0)
            image_tensor = image_tensor.to(device)
            
            pred = unet_model(image_tensor)
            pred = torch.softmax(pred, dim=1)
            pred_np = pred.squeeze().cpu().numpy()
        
        # Calculate features
        features = []
        
        # Color statistics in multiple color spaces
        for img in [lab_image, hsv_image]:
            for channel in range(img.shape[2]):
                channel_data = img[:, :, channel]
                features.extend([
                    np.mean(channel_data),
                    np.std(channel_data),
                    stats.skew(channel_data.flatten()),
                    stats.kurtosis(channel_data.flatten()),
                    np.percentile(channel_data, 25),
                    np.percentile(channel_data, 75)
                ])
        
        # Texture features using GLCM
        glcm = feature.graycomatrix(
            (gray_image * 255).astype(np.uint8), 
            distances=[1, 2], 
            angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], 
            levels=256,
            symmetric=True, 
            normed=True
        )
        
        # Calculate GLCM properties
        glcm_props = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']
        for prop in glcm_props:
            features.extend(feature.graycoprops(glcm, prop).flatten())
        
        # Add UNet prediction distribution statistics
        for class_idx in range(NUM_CLASSES):
            class_pred = pred_np[class_idx]
            features.extend([
                np.mean(class_pred),
                np.std(class_pred),
                np.max(class_pred),
                np.sum(class_pred > 0.5)  # Area of high confidence predictions
            ])
        
        features_list.append(features)
        labels_list.append(label)
    
    return np.array(features_list, dtype=np.float32), np.array(labels_list, dtype=np.int32)

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

def compute_class_weights(dataset):
    """Compute class weights for the dataset."""
    labels_list = []
    for _, label in dataset:
        labels_list.append(label)
    labels = np.array(labels_list)
    class_counts = np.bincount(labels, minlength=NUM_CLASSES)
    total_samples = len(dataset)
    if total_samples == 0:
        raise ValueError("Dataset is empty.")
    # Avoid division by zero
    class_weights = total_samples / (NUM_CLASSES * np.where(class_counts == 0, 1, class_counts))
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)
    if class_weights_tensor.dim() == 0:
        class_weights_tensor = class_weights_tensor.unsqueeze(0)
    return class_weights_tensor