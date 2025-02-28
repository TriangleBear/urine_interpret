import numpy as np
import matplotlib.pyplot as plt

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
