import os
import numpy as np
import torch
import torch.nn.functional as F
from datasets import UrineStripDataset
from models import UNetYOLO
from config import TRAIN_IMAGE_FOLDER, TRAIN_MASK_FOLDER, VALID_IMAGE_FOLDER, VALID_MASK_FOLDER, device, NUM_CLASSES
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
import cv2

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

def load_model(weights_path):
    """Load the trained model from weights path"""
    print(f"Loading model weights from {weights_path}")
    model = UNetYOLO(3, NUM_CLASSES).to(device)
    
    try:
        state_dict = torch.load(weights_path, map_location=device)
        model.load_state_dict(state_dict)
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model weights: {e}")
        print("Continuing with uninitialized model")
    
    model.eval()
    return model

def analyze_dataset(dataset_name, image_folder, mask_folder, model=None):
    print(f"\n=== Analyzing {dataset_name} dataset ===")
    
    # Check if folders exist
    if not os.path.exists(image_folder):
        print(f"Error: Image folder {image_folder} does not exist")
        return
    if not os.path.exists(mask_folder):
        print(f"Error: Mask folder {mask_folder} does not exist")
        return
    
    # Count files
    image_files = [f for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
    mask_files = [f for f in os.listdir(mask_folder) if f.endswith('.txt')]
    
    print(f"Number of images: {len(image_files)}")
    print(f"Number of mask files: {len(mask_files)}")
    
    # Create dataset
    dataset = UrineStripDataset(image_folder, mask_folder)
    
    # Analyze labels and model predictions
    labels = []
    predictions = []
    label_errors = 0
    
    for i in range(len(dataset)):
        try:
            image, label = dataset[i]
            
            # Record label
            if isinstance(label, torch.Tensor):
                label_value = label.item() if label.numel() == 1 else label.numpy().tolist()
            else:
                label_value = label
            
            labels.append(label_value)
            
            # Get model prediction if model is provided
            if model is not None:
                with torch.no_grad():
                    image_tensor = image.unsqueeze(0).to(device)
                    outputs = model(image_tensor)
                    pooled_outputs = F.adaptive_avg_pool2d(outputs, 1).squeeze(-1).squeeze(-1)
                    _, predicted = torch.max(pooled_outputs, 1)
                    prediction = predicted.item()
                    predictions.append(prediction)
                    
            
        except Exception as e:
            print(f"Error processing item {i}: {e}")
            label_errors += 1
    
    # Analyze label distribution
    unique_labels = np.unique(labels)
    label_counts = {label: labels.count(label) for label in unique_labels}
    
    print(f"\nLabel distribution:")
    for label, count in sorted(label_counts.items()):
        class_name = CLASS_NAMES.get(label, f"Unknown-{label}")
        print(f"Class {label} ({class_name}): {count} samples ({count/len(labels)*100:.2f}%)")
    
    # If model predictions are available, analyze accuracy
    if model is not None and predictions:
        overall_accuracy = sum(1 for l, p in zip(labels, predictions) if l == p) / len(labels)
        print(f"\nOverall model accuracy: {overall_accuracy*100:.2f}%")
        
        # Per-class accuracy
        print("\nPer-class accuracy:")
        for class_id in unique_labels:
            indices = [i for i, label in enumerate(labels) if label == class_id]
            if indices:
                class_predictions = [predictions[i] for i in indices]
                correct = sum(1 for l, p in zip([labels[i] for i in indices], class_predictions) if l == p)
                print(f"  Class {class_id} ({CLASS_NAMES.get(class_id, 'Unknown')}): {correct/len(indices)*100:.2f}% ({correct}/{len(indices)})")
            else:
                print(f"  Class {class_id} ({CLASS_NAMES.get(class_id, 'Unknown')}): No samples")
        
        # Confusion matrix
        confusion = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)
        for l, p in zip(labels, predictions):
            confusion[l][p] += 1
        
        print("\nConfusion Matrix:")
        print("    " + "".join([f"{i:4d}" for i in range(NUM_CLASSES)]))
        print("    " + "----" * NUM_CLASSES)
        for i in range(NUM_CLASSES):
            print(f"{i:2d} |" + "".join([f"{confusion[i][j]:4d}" for j in range(NUM_CLASSES)]))
    
    # Visualize samples with predictions
    if model is not None:
        print("\nVisualizing samples with predictions...")
        visualize_samples_with_predictions(dataset, model, 5)
    else:
        print("\nVisualizing random samples...")
        visualize_samples(dataset, 5)

def visualize_samples(dataset, num_samples=5):
    indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
    
    for idx in indices:
        image, label = dataset[idx]
        
        # Convert to numpy for visualization
        if isinstance(image, torch.Tensor):
            # Convert from CxHxW to HxWxC
            image_np = image.permute(1, 2, 0).numpy()
            # Normalize to [0, 1] range if needed
            if image_np.max() > 1.0:
                image_np = image_np / 255.0
        else:
            image_np = np.array(image) / 255.0
        
        plt.figure(figsize=(8, 6))
        plt.imshow(image_np)
        class_name = CLASS_NAMES.get(label, f"Unknown-{label}")
        plt.title(f"Sample {idx}: Class {label} ({class_name})")
        plt.axis('off')
        plt.show()

def visualize_samples_with_predictions(dataset, model, num_samples=5):
    indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
    
    for idx in indices:
        image, label = dataset[idx]
        
        # Get model prediction
        with torch.no_grad():
            image_tensor = image.unsqueeze(0).to(device)
            outputs = model(image_tensor)
            
            # Get class probabilities
            pooled_outputs = F.adaptive_avg_pool2d(outputs, 1).squeeze(-1).squeeze(-1)
            probabilities = F.softmax(pooled_outputs, dim=1).squeeze().cpu().numpy()
            
            # Get predicted class
            _, predicted = torch.max(pooled_outputs, 1)
            prediction = predicted.item()
            
            # Get segmentation map
            seg_map = torch.argmax(outputs, dim=1).squeeze().cpu().numpy()
        
        # Convert image to numpy for visualization
        if isinstance(image, torch.Tensor):
            image_np = image.permute(1, 2, 0).cpu().numpy()
            if image_np.max() > 1.0:
                image_np = image_np / 255.0
        else:
            image_np = np.array(image) / 255.0
        
        # Create colored segmentation mask
        color_mask = np.zeros((seg_map.shape[0], seg_map.shape[1], 3), dtype=np.uint8)
        for class_id in range(NUM_CLASSES):
            mask = (seg_map == class_id)
            if mask.any():
                color = np.array([class_id * 25 % 255, 
                                 (class_id * 50) % 255, 
                                 (class_id * 100) % 255], dtype=np.uint8)
                color_mask[mask] = color
        
        # Create figure with multiple plots
        plt.figure(figsize=(15, 8))
        
        # Original image with true label
        plt.subplot(1, 3, 1)
        plt.imshow(image_np)
        true_class = CLASS_NAMES.get(label, f"Unknown-{label}")
        plt.title(f"True: {true_class} (Class {label})")
        plt.axis('off')
        
        # Segmentation mask
        plt.subplot(1, 3, 2)
        plt.imshow(color_mask)
        pred_class = CLASS_NAMES.get(prediction, f"Unknown-{prediction}")
        plt.title(f"Prediction: {pred_class} (Class {prediction})")
        plt.axis('off')
        
        # Class probabilities
        plt.subplot(1, 3, 3)
        bars = plt.bar(range(NUM_CLASSES), probabilities)
        plt.xticks(range(NUM_CLASSES), [f"{i}" for i in range(NUM_CLASSES)], rotation=45)
        plt.title('Class Probabilities')
        plt.ylim(0, 1)
        
        # Highlight true and predicted class
        bars[label].set_color('green')
        if prediction != label:
            bars[prediction].set_color('red')
        
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    # Look for model weights in the models directory
    models_dir = r"D:\Programming\urine_interpret\models"
    weights_files = [f for f in os.listdir(models_dir) if f.endswith(('.pt', '.pth'))]
    
    if weights_files:
        # Use the most recent weights file
        weights_files.sort(key=lambda f: os.path.getmtime(os.path.join(models_dir, f)), reverse=True)
        weights_path = os.path.join(models_dir, weights_files[0])
        print(f"Found model weights: {weights_path}")
        model = load_model(weights_path)
    else:
        print("No model weights found. Proceeding with dataset analysis only.")
        model = None
    
    analyze_dataset("Training", TRAIN_IMAGE_FOLDER, TRAIN_MASK_FOLDER, model)
    analyze_dataset("Validation", VALID_IMAGE_FOLDER, VALID_MASK_FOLDER, model)
