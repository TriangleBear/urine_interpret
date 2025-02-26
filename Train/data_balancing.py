import os
import random
import shutil
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as T
from config import (
    TRAIN_IMAGE_FOLDER, TRAIN_MASK_FOLDER,
    VALID_IMAGE_FOLDER, VALID_MASK_FOLDER,
    NUM_CLASSES
)

# Define class names for better logging
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

def get_class_from_filename(filename):
    """Extract class ID from filename."""
    try:
        parts = os.path.splitext(filename)[0].split('_')
        if len(parts) >= 2:
            class_id = int(parts[1])
            return class_id
        return 10  # Default to strip class
    except:
        return 10  # Default to strip class

def analyze_dataset(folder, label_folder):
    """Analyze class distribution in a dataset."""
    image_files = [f for f in os.listdir(folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
    class_counts = {i: 0 for i in range(NUM_CLASSES)}
    
    for image_file in image_files:
        class_id = get_class_from_filename(image_file)
        class_counts[class_id] = class_counts.get(class_id, 0) + 1
    
    total_images = len(image_files)
    print(f"\nDataset analysis for {folder}:")
    print(f"Total images: {total_images}")
    
    for class_id in range(NUM_CLASSES):
        count = class_counts.get(class_id, 0)
        percentage = (count / total_images * 100) if total_images > 0 else 0
        print(f"Class {class_id} ({CLASS_NAMES[class_id]}): {count} images ({percentage:.2f}%)")
    
    return class_counts

def oversample_minority_classes(src_image_folder, src_label_folder, target_min_samples=10):
    """Oversample minority classes to ensure minimum number of samples."""
    image_files = [f for f in os.listdir(src_image_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    # Group images by class
    class_images = {i: [] for i in range(NUM_CLASSES)}
    for img_file in image_files:
        class_id = get_class_from_filename(img_file)
        class_images[class_id].append(img_file)
    
    augmented_count = 0
    
    # For each under-represented class, create augmented versions
    for class_id in range(NUM_CLASSES):
        images = class_images[class_id]
        if len(images) == 0:
            print(f"Warning: No images for class {class_id} ({CLASS_NAMES[class_id]})")
            continue
            
        if len(images) < target_min_samples:
            needed = target_min_samples - len(images)
            print(f"Class {class_id} ({CLASS_NAMES[class_id]}) has only {len(images)} samples. Creating {needed} augmented copies.")
            
            # Basic augmentation transformations
            augment = T.Compose([
                T.RandomHorizontalFlip(p=0.5),
                T.RandomVerticalFlip(p=0.5),
                T.RandomRotation(15),
                T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
            ])
            
            for i in range(needed):
                # Select random image from this class
                source_img = random.choice(images)
                source_path = os.path.join(src_image_folder, source_img)
                source_label_path = os.path.join(src_label_folder, os.path.splitext(source_img)[0] + '.txt')
                
                # Create new filename for augmented image
                name, ext = os.path.splitext(source_img)
                new_name = f"{name}_aug{i}{ext}"
                target_path = os.path.join(src_image_folder, new_name)
                target_label_path = os.path.join(src_label_folder, os.path.splitext(new_name)[0] + '.txt')
                
                # Apply augmentation to image
                img = Image.open(source_path).convert('RGB')
                augmented_img = augment(img)
                augmented_img.save(target_path)
                
                # Copy the label file
                if os.path.exists(source_label_path):
                    shutil.copy(source_label_path, target_label_path)
                
                augmented_count += 1
    
    print(f"Created {augmented_count} augmented images to balance the dataset")

def balance_training_data(target_min_samples=10):
    """Analyze and balance the training and validation datasets."""
    print("Analyzing dataset distribution...")
    
    train_class_counts = analyze_dataset(TRAIN_IMAGE_FOLDER, TRAIN_MASK_FOLDER)
    valid_class_counts = analyze_dataset(VALID_IMAGE_FOLDER, VALID_MASK_FOLDER)
    
    print("\nBalancing training dataset...")
    oversample_minority_classes(TRAIN_IMAGE_FOLDER, TRAIN_MASK_FOLDER, target_min_samples)
    
    print("\nAfter balancing:")
    train_class_counts = analyze_dataset(TRAIN_IMAGE_FOLDER, TRAIN_MASK_FOLDER)
    
    return train_class_counts, valid_class_counts

if __name__ == "__main__":
    balance_training_data(target_min_samples=20)
