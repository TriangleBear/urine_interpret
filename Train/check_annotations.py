import os
import numpy as np
import matplotlib.pyplot as plt
from config import TRAIN_MASK_FOLDER, VALID_MASK_FOLDER, TEST_MASK_FOLDER

# Define class names for better reporting
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
    11: 'background'
}

def check_annotation_files(folder_path, max_files=50):
    """
    Check YOLO annotation files in a folder and report class distribution.
    """
    print(f"\nChecking annotation files in: {folder_path}")
    
    # Get all txt files
    files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]
    
    if not files:
        print("No annotation files found!")
        return None
    
    # Limit the number of files to check
    files = files[:max_files]
    
    # Track class distribution
    class_counts = {i: 0 for i in range(12)}
    file_with_classes = {i: [] for i in range(12)}
    
    # Check each file
    for filename in files:
        file_path = os.path.join(folder_path, filename)
        
        with open(file_path, 'r') as f:
            lines = f.readlines()
            
            # Count annotations by class
            file_classes = set()
            for line in lines:
                parts = line.strip().split()
                if parts:
                    try:
                        class_id = int(parts[0])
                        class_counts[class_id] += 1
                        file_classes.add(class_id)
                        file_with_classes[class_id].append(filename)
                    except ValueError:
                        print(f"Invalid class ID in {filename}: {line.strip()}")
    
    # Report results
    print(f"\nAnnotation class distribution across {len(files)} files:")
    for cls in range(12):
        if class_counts[cls] > 0:
            print(f"Class {cls} ({CLASS_NAMES[cls]}): {class_counts[cls]} annotations in {len(set(file_with_classes[cls]))} files")
    
    # Report missing classes
    missing_classes = [cls for cls in range(11) if class_counts[cls] == 0]  # Skip background (11)
    if missing_classes:
        print(f"\nMissing classes: {missing_classes}")
        print("These classes have no annotations in the dataset.")
    
    # Plot distribution
    plt.figure(figsize=(12, 6))
    classes = list(range(11))  # Skip background
    counts = [class_counts[cls] for cls in classes]
    
    plt.bar(classes, counts)
    plt.xticks(classes, [CLASS_NAMES[cls] for cls in classes], rotation=45, ha='right')
    plt.xlabel('Classes')
    plt.ylabel('Number of Annotations')
    plt.title(f'Class Distribution in {os.path.basename(folder_path)}')
    plt.tight_layout()
    plt.show()
    
    return class_counts

def examine_specific_files(mask_folder, filenames):
    """
    Check the content of specific annotation files.
    """
    print(f"\nExamining specific files:")
    
    for filename in filenames:
        file_path = os.path.join(mask_folder, filename)
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            continue
            
        print(f"\nFile: {filename}")
        
        with open(file_path, 'r') as f:
            content = f.read()
            print("Content:")
            print(content)
            
            # Count classes
            classes = {}
            for line in content.strip().split('\n'):
                if line.strip():
                    parts = line.strip().split()
                    if parts:
                        try:
                            class_id = int(parts[0])
                            classes[class_id] = classes.get(class_id, 0) + 1
                        except ValueError:
                            pass
                            
            print("Classes found:", sorted(classes.keys()))

if __name__ == "__main__":
    # Check train, validation, and test sets
    train_counts = check_annotation_files(TRAIN_MASK_FOLDER, max_files=None)
    valid_counts = check_annotation_files(VALID_MASK_FOLDER, max_files=None)
    test_counts = check_annotation_files(TEST_MASK_FOLDER, max_files=None)
    
    # Compare distributions
    if train_counts and valid_counts and test_counts:
        print("\nClass distribution comparison:")
        for cls in range(11):  # Skip background
            print(f"Class {cls} ({CLASS_NAMES[cls]}):")
            print(f"  Train: {train_counts[cls]} annotations")
            print(f"  Valid: {valid_counts[cls]} annotations")
            print(f"  Test:  {test_counts[cls]} annotations")
    
    # Check specific files from debug output
    specific_files = [
        "10_jpg.rf.06fb047ad59b246fe8f9e9a817ca99b9.txt",
        "10_jpg.rf.29a403680135a3e61650c13539075099.txt",
        "10_jpg.rf.4615bd7fa5bf269532c0e4e7724926ab.txt"
    ]
    examine_specific_files(TRAIN_MASK_FOLDER, specific_files)
    
    # Check 5 random files to see if they have multiple classes
    import random
    all_files = [f for f in os.listdir(TRAIN_MASK_FOLDER) if f.endswith('.txt')]
    random_files = random.sample(all_files, 5)
    examine_specific_files(TRAIN_MASK_FOLDER, random_files)
