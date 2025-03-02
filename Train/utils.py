import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import torch.nn.functional as F
import cv2
import pickle
from tqdm import tqdm
from config import device, NUM_CLASSES

# Standard utility functions
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
        images = batch[0]  # Extract just the image tensors
        
        batch_size = images.size(0)
        images = images.view(batch_size, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        num_samples += batch_size
    
    mean /= num_samples
    std /= num_samples
    
    return mean.numpy(), std.numpy()

def dynamic_normalization(images, epsilon=1e-5, use_channels=True):
    """Optimized normalization that can work per-image or per-batch."""
    if use_channels:
        # Per-channel normalization (original)
        batch_size = images.size(0)
        mean = images.view(batch_size, images.size(1), -1).mean(dim=2).view(batch_size, images.size(1), 1, 1)
        std = images.view(batch_size, images.size(1), -1).std(dim=2).view(batch_size, images.size(1), 1, 1) + epsilon
    else:
        # Faster global normalization
        mean = images.mean(dim=(2, 3), keepdim=True)
        std = images.std(dim=(2, 3), keepdim=True) + epsilon
    return (images - mean) / std

def post_process_mask(mask, kernel_size=3):
    """Apply post-processing to the predicted mask to remove noise and smooth boundaries."""
    # Convert to uint8 for OpenCV operations
    mask_uint8 = mask.astype(np.uint8)
    
    # Apply morphological operations
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    mask_cleaned = cv2.morphologyEx(mask_uint8, cv2.MORPH_OPEN, kernel)
    mask_cleaned = cv2.morphologyEx(mask_cleaned, cv2.MORPH_CLOSE, kernel)
    
    return mask_cleaned

def post_process_segmentation(logits, apply_layering=True):
    """Optimized post-processing with better memory usage."""
    if not apply_layering:
        # Standard argmax approach (highest probability wins)
        return torch.argmax(F.softmax(logits, dim=1), dim=1)
    
    # Get device to avoid unnecessary transfers
    device = logits.device
    batch_size, num_classes, height, width = logits.shape
    
    # Process in smaller chunks if needed for memory efficiency
    masks = torch.zeros((batch_size, height, width), device=device, dtype=torch.long)
    
    # More efficient softmax calculation
    with torch.no_grad():
        # Use log_softmax which is more numerically stable
        log_probs = F.log_softmax(logits, dim=1)
        probs = torch.exp(log_probs)  # Convert to probabilities
        
        # Get strip probabilities (class 11)
        strip_prob = probs[:, 11]  # Class 11 (strip)
        
        # Apply strip where it beats background
        masks[strip_prob > 0.5] = 11  # Threshold for strip
        
        # Apply reagent pads (0-8, 10) where they beat both strip and background
        for class_id in list(range(9)) + [10]:  # 0-8, 10 = reagent pads 
            pad_prob = probs[:, class_id]
            pad_wins = pad_prob > strip_prob
            masks[pad_wins] = class_id
        
        # Apply background (class 9) where it beats all other classes
        background_prob = probs[:, 9]  # Class 9 (background)
        background_wins = background_prob > strip_prob
        for class_id in list(range(9)) + [10]:
            background_wins &= background_prob > probs[:, class_id]
        masks[background_wins] = 9
            
    return masks

def compute_class_weights(dataset, max_weight=50.0, min_weight=0.5):
    """
    Compute class weights for handling imbalanced datasets.
    """
    class_counts = {}
    empty_label_count = 0
    total_samples = 0
    
    # Initialize all classes with zero count
    for i in range(NUM_CLASSES + 1):  # +1 to include the special empty label class
        class_counts[i] = 0
    
    # Count actual occurrences
    for _, label, _ in dataset:
        label_val = label.item() if isinstance(label, torch.Tensor) else label
        
        # Count empty labels separately from class 0
        class_counts[label_val] = class_counts.get(label_val, 0) + 1
        total_samples += 1
            
        # Track empty labels for reporting
        if label_val == NUM_CLASSES:  # NUM_CLASSES (12) indicates an empty label
            empty_label_count += 1
    
    # Print class distribution summary
    print("\nClass distribution in dataset:")
    valid_classes = []
    missing_classes = []
    
    for cls in range(NUM_CLASSES):  # Only iterate through valid classes (0-11)
        count = class_counts.get(cls, 0)
        if count > 0:
            valid_classes.append(cls)
            class_percentage = count/total_samples*100 if total_samples > 0 else 0
            print(f"Class {cls}: {count} samples ({class_percentage:.1f}%)")
        else:
            missing_classes.append(cls)
    
    # Report empty labels separately
    if empty_label_count > 0:
        empty_percentage = empty_label_count/total_samples*100
        print(f"Empty labels: {empty_label_count} samples ({empty_percentage:.1f}%)")
    
    if missing_classes:
        print(f"Missing classes: {missing_classes}")
    
    # Calculate class weights for actual classes
    weights = torch.ones(NUM_CLASSES + 1, device=device)  # +1 for empty label class
    
    for i in range(NUM_CLASSES + 1):  # Include the special empty label class
        if class_counts[i] > 0:
            # Inverse frequency weighting with smoothing
            weights[i] = 1.0 / (class_counts[i] / total_samples)
            
            # Special handling for empty labels (NUM_CLASSES)
            if i == NUM_CLASSES:  # Empty label class
                # Almost ignore empty labels by giving them very low weight
                weights[i] *= 0.1  # Minimal emphasis on empty labels
        else:
            # For missing classes, use average weight of present classes
            if valid_classes:
                avg_weight = sum(1.0 / (class_counts[c] / total_samples) for c in valid_classes) / len(valid_classes)
                weights[i] = avg_weight
            else:
                weights[i] = 1.0
    
    # Normalize weights
    if weights[:NUM_CLASSES].sum() > 0:  # Only normalize actual classes (0-11)
        avg_weight = weights[:NUM_CLASSES].sum() / NUM_CLASSES
        weights[:NUM_CLASSES] = weights[:NUM_CLASSES] / avg_weight
    
    # Cap minimum and maximum weights for actual classes
    weights[:NUM_CLASSES] = torch.clamp(weights[:NUM_CLASSES], min_weight, max_weight)
    
    # Debug: Print the final weights to make sure they're correct
    print(f"Final class weights shape: {weights.shape}")
    print(f"Class weights: {weights}")
    
    return weights
