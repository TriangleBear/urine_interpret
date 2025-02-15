# main.py
import matplotlib.pyplot as plt
from config import *
from train_unet import train_unet
from utils import compute_mean_std, extract_features, train_svm

if __name__ == "__main__":
    # Compute dataset statistics
    dataset = UrineStripDataset(IMAGE_FOLDER, MASK_FOLDER)
    mean, std = compute_mean_std(dataset)
    print(f"Dataset mean: {mean}, std: {std}")
    
    # Train UNet
    unet_model = train_unet()
    
    # Train SVM
    features, labels = extract_features(unet_model, dataset)
    svm_model = train_svm(features, labels)
    joblib.dump(svm_model, get_svm_filename())
    

    # Plot training results
    epochs_range = range(1, len(train_losses) + 1)
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, train_losses, label='Train Loss')
    plt.plot(epochs_range, val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.title('Validation Accuracy')
    plt.tight_layout()
    plt.show()