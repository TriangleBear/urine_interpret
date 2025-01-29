import torch
from torch.utils.data import DataLoader, Dataset, random_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from torchvision import transforms
import cv2
import os
import time
import numpy as np
import torch.optim as optim
import torch.nn as nn
from torch.amp import autocast, GradScaler
import torch.nn.functional as F
from tqdm import tqdm
from PIL import Image
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

# Define the paths to your image and mask folders
image_folder = r'D:\Programming\urine_interpret\New Urine Datasets\train\images'  # Replace with the actual path to the images
mask_folder = r'D:\Programming\urine_interpret\New Urine Datasets\train\labels'   # Replace with the actual path to the masks

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels=21):
        super(UNet, self).__init__()

        # Encoder path
        self.enc1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.enc2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.enc3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.enc4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)  # Add this layer

        # Bottleneck layer
        self.bottleneck = nn.Conv2d(256, 512, kernel_size=3, padding=1)  # Adjusted channels
        self.dropout = nn.Dropout(p=0.2)  # Add dropout

        # Decoder path
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)  # Adjusted to match enc4 channels
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)

        # Additional Conv2d layers after concatenation to reduce channels
        self.conv_up3 = nn.Conv2d(512, 256, kernel_size=3, padding=1)  # Adjusted to match enc4 channels after concat
        self.conv_up2 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.conv_up1 = nn.Conv2d(128, 64, kernel_size=3, padding=1)

        # Output layer
        self.output = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder path
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)  # Add this layer

        # Bottleneck layer
        bottleneck = self.bottleneck(enc4)
        bottleneck = self.dropout(bottleneck)  # Apply dropout

        # Decoder path
        up3 = self.upconv3(bottleneck)
        up3 = F.interpolate(up3, size=enc4.size()[2:], mode='bilinear', align_corners=True)
        up3 = torch.cat([up3, enc4], dim=1)  # Now both have 512 channels
        up3 = self.conv_up3(up3)  # Output 256 channels

        up2 = self.upconv2(up3)
        up2 = F.interpolate(up2, size=enc3.size()[2:], mode='bilinear', align_corners=True)
        up2 = torch.cat([up2, enc3], dim=1)  # Now both have 256 channels
        up2 = self.conv_up2(up2)  # Output 128 channels

        up1 = self.upconv1(up2)
        up1 = F.interpolate(up1, size=enc2.size()[2:], mode='bilinear', align_corners=True)
        up1 = torch.cat([up1, enc2], dim=1)  # Now both have 128 channels
        up1 = self.conv_up1(up1)  # Output 64 channels

        # Output layer
        output = self.output(up1)

        return output

class UNetWithFeatures(UNet):
    def extract_features(self, x):
        """
        Extract bottleneck features from the U-Net for SVM classification.
        """
        # Encoder path
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        bottleneck = self.bottleneck(enc4)

        # Flatten bottleneck features for classification
        return bottleneck.view(bottleneck.size(0), -1)

num_classes = 21  # Adjust based on actual output classes

def create_mask_from_yolo(txt_path, image_size=(256, 256)):
    """
    Parse a YOLOv8 annotation file and create a multi-class mask based on the `data.yaml` file.
    """
    mask = np.zeros(image_size, dtype=np.uint8)  # Initialize a blank mask
    mask_height, mask_width = image_size
    
    try:
        with open(txt_path, 'r') as file:
            lines = file.readlines()
        
        if not lines:
            print(f"Warning: Empty annotation file {txt_path}. Skipping.")
            return mask  # Return empty mask to keep dataset size consistent

        for line in lines:
            parts = line.strip().split()

            if len(parts) < 5:
                print(f"Warning: Invalid line in {txt_path}: '{line.strip()}'. Expected at least 5 values.")
                continue  # Skip but log the issue

            try:
                class_id, center_x, center_y, bbox_width, bbox_height = map(float, parts[:5])
            except ValueError:
                print(f"Warning: Could not parse line in {txt_path}: '{line.strip()}'. Skipping.")
                continue

            # Ensure class_id is within valid range
            if not (0 <= int(class_id) < num_classes):
                print(f"Warning: Invalid class ID {class_id} in {txt_path}. Skipping.")
                continue

            # Convert normalized YOLOv8 coordinates to absolute pixel coordinates
            center_x = int(center_x * mask_width)
            center_y = int(center_y * mask_height)
            bbox_width = max(1, int(bbox_width * mask_width))  # Ensure min size 1 pixel
            bbox_height = max(1, int(bbox_height * mask_height))

            # Calculate bounding box coordinates
            xmin = max(0, center_x - bbox_width // 2)
            ymin = max(0, center_y - bbox_height // 2)
            xmax = min(mask_width, center_x + bbox_width // 2)
            ymax = min(mask_height, center_y + bbox_height // 2)

            # Check if valid bbox
            if xmax <= xmin or ymax <= ymin:
                print(f"Warning: Zero-size bounding box in {txt_path}. Skipping.")
                continue

            # Use np.maximum to preserve overlapping regions instead of overwriting
            mask[ymin:ymax, xmin:xmax] = np.maximum(mask[ymin:ymax, xmin:xmax], int(class_id))

    except Exception as e:
        print(f"Error processing {txt_path}: {e}")
        return mask  # Return empty mask to prevent dataset issues

    return mask

# Define data augmentation transformations
data_augmentation = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomPerspective(distortion_scale=0.2),  # Add perspective distortion
    transforms.ToTensor()
])

class UrineStripDataset(Dataset):
    def __init__(self, image_folder, mask_folder, transform=None):
        # Initialize the dataset with paths and transformations
        self.image_folder = image_folder
        self.mask_folder = mask_folder
        self.transform = transform

        # List all image files and corresponding txt files
        self.image_files = sorted(os.listdir(image_folder))
        self.txt_files = sorted(os.listdir(mask_folder))

        # Check that the lengths of both lists are the same
        if len(self.image_files) != len(self.txt_files):
            raise ValueError(f"Mismatch between number of images ({len(self.image_files)}) and masks ({len(self.txt_files)})")

    def __len__(self):
        # Return the number of images in the dataset
        return len(self.image_files)

    def __getitem__(self, index):
        # Ensure the index is within bounds
        if index >= len(self.image_files) or index >= len(self.txt_files):
            raise IndexError(f"Index {index} out of range: image_files = {len(self.image_files)}, txt_files = {len(self.txt_files)}")

        # Load image and mask
        image_file = self.image_files[index]
        txt_file = self.txt_files[index]

        image_path = os.path.join(self.image_folder, image_file)
        txt_path = os.path.join(self.mask_folder, txt_file)

        # Read image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Image at {image_path} could not be loaded")

        # Create multi-class mask
        mask = create_mask_from_yolo(txt_path, image_size=(256, 256))

        # Preprocess image
        image = cv2.resize(image, (256, 256))
        image = image.astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))

        image = Image.fromarray((image * 255).astype(np.uint8).transpose(1, 2, 0))  

        if self.transform:
            image = self.transform(image)

        # Ensure the mask is a torch tensor with the correct shape
        mask = torch.tensor(mask, dtype=torch.long)  # Change to torch.long for multi-class

        return image, mask

# Split the dataset into training and validation sets
dataset = UrineStripDataset(image_folder, mask_folder, transform=data_augmentation)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

# Initialize the U-Net model                                                                   
unet_model = UNetWithFeatures(in_channels=3, out_channels=num_classes)

# Set the device (GPU or CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
unet_model.to(device)
print(f"Using device: {device}")

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()  # Multi-class loss
optimizer = optim.Adam(unet_model.parameters(), lr=1e-4, weight_decay=0.01)

# Mixed Precision Training Setup
scaler = GradScaler("cuda")

# Early stopping parameters
patience = 20  # Number of epochs to wait before stopping
best_val_loss = float('inf')  # Initialize with a large value
counter = 0  # Counter for epochs without improvement

# Training loop
num_epochs = 100  # Set a max number of epoch
torch.autograd.set_detect_anomaly(True)

for epoch in range(num_epochs):
    unet_model.train()
    running_loss = 0.0

    for images, masks in tqdm(train_loader):
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()

        # Ensure mask is valid
        masks = masks.squeeze(1)  # Ensure shape is (batch, H, W)
        masks = masks.long()       # Ensure dtype is correct

        # Validate mask values
        assert masks.min() >= 0, f"Invalid mask value: {masks.min()}"
        assert masks.max() < num_classes, f"Invalid mask value: {masks.max()}"

        # Mixed precision training
        with autocast('cuda'):
            outputs = unet_model(images)  # Model output is raw logits

            loss = criterion(outputs, masks)  # No sigmoid needed for CE Loss

        if torch.isnan(loss) or torch.isinf(loss):  # Debugging check
            print("NaN or Inf detected in loss! Exiting training.")
            break  # Stop training to debug

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()

    # Validation phase
    unet_model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for images, masks in tqdm(val_loader):
            images = images.to(device)
            masks = masks.squeeze(1)  # Remove channel dimension if present
            masks = masks.long()      # Ensure correct dtype
            masks = masks.to(device)

            with autocast('cuda'):
                outputs = unet_model(images)  # raw logits
                loss = criterion(outputs, masks)  # no sigmoid

            val_loss += loss.item()

    avg_train_loss = running_loss / len(train_loader)
    avg_val_loss = val_loss / len(val_loader)

    print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

    # Generate a timestamp-based filename
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    model_filename = f"unet_model_{timestamp}.pth"

    # Early stopping logic
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        counter = 0
        torch.save(unet_model.state_dict(), model_filename)
    else:
        counter += 1
        if counter >= patience:
            print("Early stopping")
            break

# Save the final model
torch.save(unet_model.state_dict(), model_filename)

# # Extract bottleneck features for SVM classification
# def extract_features_and_labels(model, dataloader, device):
#     model.eval()
#     features = []
#     labels = []

#     with torch.no_grad():
#         for images, masks in tqdm(dataloader, desc="Extracting Features"):
#             images = images.to(device)

#             # Extract bottleneck features
#             bottleneck_features = model.extract_features(images)
#             features.append(bottleneck_features.cpu().numpy())

#             # Flatten masks and append corresponding labels incrementally
#             for mask in masks:
#                 labels.append(mask.cpu().numpy().flatten())  # Ensure alignment

#     return np.array(features), np.array(labels)

# # Extract features from the training dataset
# train_features, train_labels = extract_features_and_labels(unet_model, train_loader, device)

# # Standardize features for SVM
# scaler = StandardScaler()
# train_features_scaled = scaler.fit_transform(train_features)

# # Train an SVM classifier
# svm_classifier = SVC(kernel='rbf', random_state=42)
# svm_classifier.fit(train_features_scaled, train_labels.flatten())

# # Save the scaler and SVM model
# joblib.dump(scaler, 'scaler.pkl')
# joblib.dump(svm_classifier, 'svm_classifier.pkl')

# print("SVM model and scaler saved.")

# # Now you can use the `svm_classifier` for prediction on new samples.