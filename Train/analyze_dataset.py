import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import Counter
import pandas as pd

# Define constants
DATA_ROOT = "D:/Programming/urine_interpret/Datasets/Final"
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
    9: 'Background',
    10: 'pH',
    11: 'Strip'
}

def parse_yolo_annotation(txt_path, image_size=(256, 256)):
    """Parse YOLO annotation file and return class IDs present"""
    if not os.path.exists(txt_path) or os.path.getsize(txt_path) == 0:
        return []
        
    classes = set()
    with open(txt_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            try:
                class_id = int(parts[0])
                classes.add(class_id)
            except:
                pass
    return list(classes)

def analyze_dataset_split(split_name):
    """Analyze a dataset split (train/valid/test)"""
    print(f"\nAnalyzing {split_name} dataset...")
    
    labels_dir = os.path.join(DATA_ROOT, split_name, "labels")
    if not os.path.exists(labels_dir):
        print(f"Directory not found: {labels_dir}")
        return
        
    label_files = [f for f in os.listdir(labels_dir) if f.endswith('.txt')]
    
    # Count class occurrences
    class_counts = Counter()
    files_with_class = {i: [] for i in range(12)}
    missing_classes = set(range(12))
    
    for label_file in tqdm(label_files, desc=f"Processing {split_name}"):
        txt_path = os.path.join(labels_dir, label_file)
        classes = parse_yolo_annotation(txt_path)
        
        for cls in classes:
            class_counts[cls] += 1
            files_with_class[cls].append(label_file)
            if cls in missing_classes:
                missing_classes.remove(cls)
    
    # Create pretty table for display
    data = []
    for class_id in range(12):
        class_name = CLASS_NAMES.get(class_id, f"Class {class_id}")
        count = class_counts.get(class_id, 0)
        percentage = count / len(label_files) * 100 if label_files else 0
        data.append([class_id, class_name, count, f"{percentage:.1f}%"])
    
    # Print results using pandas DataFrame for nice formatting
    df = pd.DataFrame(data, columns=["Class ID", "Class Name", "Count", "Percentage"])
    print(df.to_string(index=False))
    
    # Report missing classes
    if missing_classes:
        print(f"\nMissing classes in {split_name}: {', '.join(CLASS_NAMES[cls] for cls in missing_classes)}")
    
    # Create bar chart
    plt.figure(figsize=(12, 6))
    plt.bar(
        [CLASS_NAMES.get(i, f"Class {i}") for i in range(12)],
        [class_counts.get(i, 0) for i in range(12)]
    )
    plt.xticks(rotation=45, ha='right')
    plt.title(f"Class Distribution in {split_name} Dataset")
    plt.ylabel("Number of Images")
    plt.tight_layout()
    plt.savefig(f"{split_name}_class_distribution.png", dpi=300)

if __name__ == "__main__":
    # Analyze each dataset split
    for split in ["train", "valid", "test"]:
        analyze_dataset_split(split)
    
    print("\nAnalysis complete. Check the generated charts for class distribution.")
