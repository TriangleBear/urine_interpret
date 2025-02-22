import matplotlib.pyplot as plt
from config import *
import numpy as np
from train_unet_yolo import train_unet_yolo
from utils import compute_mean_std, dynamic_normalization, post_process_mask  # Import post_process_mask
from datasets import UrineStripDataset, visualize_dataset

if __name__ == "__main__":
    # Compute dataset statistics
    dataset = UrineStripDataset(IMAGE_FOLDER, MASK_FOLDER)
    mean, std = compute_mean_std(dataset)
    print(f"Dataset mean: {mean}, std: {std}")
    
    # Visualize the dataset
    # visualize_dataset(dataset)
    
    # Train UNetYOLO
    unet_model, train_losses, val_losses, val_accuracies = train_unet_yolo()
    
    # Plot training results
    epochs_range = range(1, len(train_losses) + 1)
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.plot(epochs_range, train_losses, label='Train Loss')
    plt.plot(epochs_range, val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.subplot(1, 3, 2)
    plt.plot(epochs_range, val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.title('Validation Accuracy')
    plt.tight_layout()
    plt.show()

    # Apply post-processing to the predicted masks
    for i, (images, masks) in enumerate(dataset):
        with torch.no_grad():
            images = images.unsqueeze(0).to(device)
            images = dynamic_normalization(images)  # Normalize the input images
            outputs = unet_model(images)
            predicted_mask = torch.argmax(outputs, dim=1).squeeze(0).cpu().numpy()
            refined_mask = post_process_mask(predicted_mask)
            cv2.imwrite(f"refined_mask_{i}.png", refined_mask)