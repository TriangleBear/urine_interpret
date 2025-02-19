import matplotlib.pyplot as plt
from config import *
from train_unet import train_unet
from utils import compute_mean_std, extract_features_and_labels, train_svm_classifier, save_svm_model
from datasets import UrineStripDataset, visualize_dataset
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    # Compute dataset statistics
    dataset = UrineStripDataset(IMAGE_FOLDER, MASK_FOLDER)
    mean, std = compute_mean_std(dataset)
    print(f"Dataset mean: {mean}, std: {std}")
    
    # Visualize the dataset
    visualize_dataset(dataset)
    
    # Train UNet
    unet_model, train_losses, val_losses, val_accuracies = train_unet()
    
    # Extract features and labels for SVM
    features, labels = extract_features_and_labels(dataset, unet_model)
    
    # Ensure there are multiple classes in the labels
    if len(np.unique(labels)) > 1:
        # Train SVM
        svm_model = train_svm_classifier(features, labels)
        save_svm_model(svm_model, get_svm_filename())

        # Evaluate SVM
        X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2)
        svm_accuracy = svm_model.score(X_test, y_test) * 100
        print(f"SVM RBF Accuracy: {svm_accuracy:.2f}%")
        
        # Plot SVM accuracy
        plt.subplot(1, 3, 3)
        plt.bar(['SVM RBF'], [svm_accuracy])
        plt.ylabel('Accuracy (%)')
        plt.title('SVM RBF Accuracy')
    else:
        print("Not enough classes to train SVM. Skipping SVM training and evaluation.")
    
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
            outputs = unet_model(images)
            predicted_mask = torch.argmax(outputs, dim=1).squeeze(0).cpu().numpy()
            refined_mask = post_process_mask(predicted_mask)
            cv2.imwrite(f"refined_mask_{i}.png", refined_mask)