import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import torch.nn.functional as F
import cv2
import pickle
from tqdm import tqdm
from config import device, NUM_CLASSES

def log_class_distribution(class_distribution):
    """Log the class distribution."""
    classes = list(class_distribution.keys())
    counts = list(class_distribution.values())
    
    plt.bar(classes, counts)
    plt.xlabel('Classes')
    plt.ylabel('Counts')
    plt.title('Class Distribution')
    plt.show()

def dynamic_sampling(dataset, target_classes, sampling_factor=2):
    """Implement dynamic sampling for underrepresented classes."""
    sampled_indices = []
    for class_id in target_classes:
        class_indices = [i for i, (_, label) in enumerate(dataset) if label == class_id]
        sampled_indices.extend(class_indices * sampling_factor)  # Oversample underrepresented classes
    return sampled_indices

def compute_mean_std(dataset, batch_size=16):
    """Compute the mean and standard deviation of a dataset."""
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    
    mean = torch.zeros(3)
    std = torch.zeros(3)
    num_samples = 0
    
    for batch in tqdm(loader, desc="Computing dataset statistics"):
        # Correctly unpack the batch - the first element is the image
        images = batch[0]  # Extract just the image tensors
        
        batch_size = images.size(0)
        images = images.view(batch_size, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        num_samples += batch_size
    
    mean /= num_samples
    std /= num_samples
    
    return mean.numpy(), std.numpy()

def dynamic_normalization(images, epsilon=1e-5):
    """Normalize images dynamically based on their own statistics."""
    batch_size = images.size(0)
    mean = images.view(batch_size, images.size(1), -1).mean(dim=2).view(batch_size, images.size(1), 1, 1)
    std = images.view(batch_size, images.size(1), -1).std(dim=2).view(batch_size, images.size(1), 1, 1) + epsilon
    return (images - mean) / std

def compute_class_weights(dataset, max_weight=50.0, min_weight=0.5):
    """
    Compute class weights for handling imbalanced datasets with improved 
    handling of missing classes and variable class distributions.
    """
    class_counts = {}
    total_samples = 0
    
    # Initialize all classes with zero count
    for i in range(NUM_CLASSES):
        class_counts[i] = 0
    
    # Count actual occurrences
    for _, label, _ in dataset:
        label_val = label.item() if isinstance(label, torch.Tensor) else label
        if 0 <= label_val < NUM_CLASSES:  # Ensure label is within valid range
            class_counts[label_val] = class_counts.get(label_val, 0) + 1
            total_samples += 1
        else:
            print(f"Warning: Encountered invalid class label: {label_val}")
    
    # Print class distribution summary - more compact
    print("\nClass distribution in dataset:")
    valid_classes = []
    missing_classes = []
    
    for cls in range(NUM_CLASSES):
        count = class_counts.get(cls, 0)
        if count > 0:
            valid_classes.append(cls)
            print(f"Class {cls}: {count} samples ({count/total_samples*100:.1f}%)")
        else:
            missing_classes.append(cls)
    
    if missing_classes:
        print(f"Missing classes: {missing_classes}")
    
    # Calculate class weights with a minimum count to avoid division by zero
    weights = torch.ones(NUM_CLASSES, device=device)
    
    for i in range(NUM_CLASSES):
        if class_counts[i] > 0:
            # Inverse frequency weighting with smoothing
            weights[i] = 1.0 / (class_counts[i] / total_samples)
        else:
            # For missing classes, use a sensible default weight
            # Use the average weight of present classes, or a default value
            if valid_classes:
                avg_weight = sum(1.0 / (class_counts[c] / total_samples) for c in valid_classes) / len(valid_classes)
                weights[i] = avg_weight
            else:
                weights[i] = 1.0  # Default weight if no classes have samples
    
    # Normalize weights to have mean=1
    if weights.sum() > 0:
        avg_weight = weights.sum() / NUM_CLASSES
        weights = weights / avg_weight
    
    # Cap minimum and maximum weights
    weights = torch.clamp(weights, min_weight, max_weight)
    
    return weights

def post_process_mask(mask, kernel_size=3):
    """Apply post-processing to the predicted mask to remove noise and smooth boundaries."""
    # Convert to uint8 for OpenCV operations
    mask_uint8 = mask.astype(np.uint8)
    
    # Apply morphological operations
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    mask_cleaned = cv2.morphologyEx(mask_uint8, cv2.MORPH_OPEN, kernel)
    mask_cleaned = cv2.morphologyEx(mask_cleaned, cv2.MORPH_CLOSE, kernel)
    
    return mask_cleaned

def extract_features_and_labels(dataset, model):
    """Extract features from images using a trained model."""
    features = []
    labels = []
    
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        num_workers=0
    )
    
    model.eval()
    with torch.no_grad():
        for images, batch_labels, _ in tqdm(loader, desc="Extracting features"):
            images = images.to(device)
            images = dynamic_normalization(images)
            outputs = model(images)
            
            # Global average pooling for features
            pooled_features = F.adaptive_avg_pool2d(outputs, 1).squeeze(-1).squeeze(-1)
            
            features.append(pooled_features.cpu().numpy())
            labels.append(batch_labels.cpu().numpy())
    
    features = np.vstack(features)
    labels = np.concatenate(labels)
    
    return features, labels

def save_svm_model(model_data, filepath):
    """Save SVM model and related data to disk."""
    with open(filepath, 'wb') as f:
        pickle.dump(model_data, f)
