import matplotlib.pyplot as plt
from config import *  # This will now print CUDA info only once
import numpy as np
from train_unet_yolo import train_unet_yolo
from utils import compute_mean_std, dynamic_normalization, post_process_mask, post_process_segmentation
from datasets import UrineStripDataset, visualize_dataset

if __name__ == "__main__":  
    # Compute dataset statistics using training dataset
    train_dataset = UrineStripDataset(TRAIN_IMAGE_FOLDER, TRAIN_MASK_FOLDER)
    mean, std = compute_mean_std(train_dataset)
    print(f"Dataset mean: {mean}, std: {std}")
    
    # Visualize the training dataset
    
    # Path to pre-trained weights
    pre_trained_weights = r"D:\Programming\urine_interpret\models\weights.pt"
    
    # Train UNetYOLO
    result = train_unet_yolo(pre_trained_weights=pre_trained_weights)
    if result is None:
        print("Training was not completed due to an error.")
    else:
        unet_model, train_losses, val_losses, val_accuracies, test_accuracy = result
    
        # Plot training results
        epochs_range = range(1, len(train_losses) + 1)
        plt.figure(figsize=(12, 4))
        
        # Plot losses
        plt.subplot(1, 3, 1)
        plt.plot(epochs_range, train_losses, label='Train Loss')
        plt.plot(epochs_range, val_losses, label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training and Validation Loss')
        
        # Plot validation accuracy
        plt.subplot(1, 3, 2)
        plt.plot(epochs_range, val_accuracies, label='Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy (%)')
        plt.title('Validation Accuracy')
        
        # Add final test accuracy
        plt.subplot(1, 3, 3)
        plt.text(0.5, 0.5, f'Test Accuracy:\n{test_accuracy:.2f}%', 
                 horizontalalignment='center', verticalalignment='center')
        plt.axis('off')
        plt.title('Test Results')
        
        plt.tight_layout()
        plt.show()

        # Apply post-processing to the predicted masks using test dataset
        test_dataset = UrineStripDataset(TEST_IMAGE_FOLDER, TEST_MASK_FOLDER)
        for i, (images, masks) in enumerate(test_dataset):
            with torch.no_grad():
                images = images.unsqueeze(0).to(device)
                images = dynamic_normalization(images)
                outputs = unet_model(images)
                
                # Use layered post-processing for better visualization
                processed_outputs = post_process_segmentation(outputs, apply_layering=True)
                predicted_mask = processed_outputs.squeeze(0).cpu().numpy()
                
                refined_mask = post_process_mask(predicted_mask)
                cv2.imwrite(f"refined_mask_{i}.png", refined_mask)