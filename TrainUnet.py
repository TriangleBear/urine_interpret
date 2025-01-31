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
import torch.utils.checkpoint as checkpoint
from torch.amp import autocast, GradScaler
import torch.nn.functional as F
from tqdm import tqdm
from PIL import Image
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import random

torch.backends.cudnn.benchmark = True # Enable cuDNN benchmark for performance
torch.backends.cudnn.enabled = True # Enable cuDNN for performance

# Define the paths to your image and mask folders
image_folder = r'D:\Programming\urine_interpret\urine\train\images'  # Replace with the actual path to the images
mask_folder = r'D:\Programming\urine_interpret\urine\train\labels'   # Replace with the actual path to the masks

# Define the model filename with a timestamp
timestamp = time.strftime("%Y%m%d-%H%M%S")
model_filename = f"unet_model_{timestamp}.pth"

# Define the U-Net model class
class UNet(nn.Module):
    def __init__(self, in_channels, out_channels=10):
        super(UNet, self).__init__()

        # Encoder path
        self.enc1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.enc2 = nn.Conv2d(32, 64, kernel_size=3, padding=1) 
        self.enc3 = nn.Conv2d(64, 128, kernel_size=3, padding=1) 
        self.enc4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)

        # Bottleneck layer
        self.bottleneck = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(p=0.5)

        # Decoder path
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)

        # Additional Conv2d layers
        self.conv_up3 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.conv_up2 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.conv_up1 = nn.Conv2d(128, 64, kernel_size=3, padding=1)

        # Output layer
        self.output = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        enc1 = checkpoint.checkpoint(self.enc1, x, use_reentrant=False)
        enc2 = checkpoint.checkpoint(self.enc2, enc1, use_reentrant=False)
        enc3 = checkpoint.checkpoint(self.enc3, enc2, use_reentrant=False)
        enc4 = checkpoint.checkpoint(self.enc4, enc3, use_reentrant=False)

        bottleneck = checkpoint.checkpoint(self.bottleneck, enc4, use_reentrant=False)
        bottleneck = self.dropout(bottleneck)

        up3 = checkpoint.checkpoint(self.upconv3, bottleneck, use_reentrant=False)
        up3 = F.interpolate(up3, size=enc4.size()[2:], mode='bilinear', align_corners=True)
        up3 = torch.cat([up3, enc4], dim=1)
        up3 = self.conv_up3(up3)

        up2 = checkpoint.checkpoint(self.upconv2, up3, use_reentrant=False)
        up2 = F.interpolate(up2, size=enc3.size()[2:], mode='bilinear', align_corners=True)
        up2 = torch.cat([up2, enc3], dim=1)
        up2 = self.conv_up2(up2)

        up1 = checkpoint.checkpoint(self.upconv1, up2, use_reentrant=False)
        up1 = F.interpolate(up1, size=enc2.size()[2:], mode='bilinear', align_corners=True)
        up1 = torch.cat([up1, enc2], dim=1)
        up1 = self.conv_up1(up1)

        return self.output(up1)

class RandomFlip:
    def __init__(self, horizontal=True, vertical=False):
        self.horizontal = horizontal
        self.vertical = vertical

    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']
        
        if self.horizontal and random.random() > 0.5:
            image = transforms.functional.hflip(image)
            mask = transforms.functional.hflip(mask)
        
        if self.vertical and random.random() > 0.5:
            image = transforms.functional.vflip(image)
            mask = transforms.functional.vflip(mask)
        
        return {'image': image, 'mask': mask}


class RandomRotation:
    def __init__(self, degrees):
        self.degrees = degrees

    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']
        angle = random.uniform(-self.degrees, self.degrees)
        image = transforms.functional.rotate(image, angle)
        mask = transforms.functional.rotate(mask, angle)
        return {'image': image, 'mask': mask}


class RandomAffine:
    def __init__(self, translate=(0.1, 0.1)):
        self.translate = translate

    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']
        params = transforms.RandomAffine.get_params(
            degrees=0, translate=self.translate, scale=None, shear=None, img_size=image.size
        )
        image = transforms.functional.affine(image, angle=0, translate=params[0], scale=1, shear=0)
        mask = transforms.functional.affine(mask, angle=0, translate=params[0], scale=1, shear=0)
        return {'image': image, 'mask': mask}

# Compose all the transformations
class RandomTransformations:
    def __init__(self):
        self.transform = transforms.Compose([
            RandomFlip(horizontal=True, vertical=True),
            RandomRotation(degrees=10),  # Rotate by up to 10 degrees
            RandomAffine(translate=(0.1, 0.1)),  # Small translations
        ])

    def __call__(self, sample):
        return self.transform(sample)

# Dataset class
class UrineStripDataset(Dataset):
    def __init__(self, image_folder, mask_folder, transform=None):
        self.image_folder = image_folder
        self.mask_folder = mask_folder
        self.image_files = sorted(os.listdir(image_folder))
        self.txt_files = sorted(os.listdir(mask_folder))
        self.transform = transform  # Optional transform to apply to both image and mask

        if len(self.image_files) != len(self.txt_files):
            raise ValueError("Mismatch between number of images and masks")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        image_file = self.image_files[index]
        txt_file = self.txt_files[index]
        image_path = os.path.join(self.image_folder, image_file)
        txt_path = os.path.join(self.mask_folder, txt_file)

        # Load image using OpenCV (BGR format) and convert to RGB
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Image at {image_path} could not be loaded")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize image (if needed)
        image = cv2.resize(image, (128, 128))

        # Convert to tensor
        image = transforms.ToTensor()(image)

        # Load mask and create from YOLO format
        mask = self.create_mask_from_yolo(txt_path)
        mask = cv2.resize(mask, (128, 128), interpolation=cv2.INTER_NEAREST)
        mask = torch.tensor(mask, dtype=torch.long)  # Convert to tensor

        if self.transform:
            # Apply any augmentations (ensure mask also gets transformed)
            sample = {'image': image, 'mask': mask}
            sample = self.transform(sample)
            image, mask = sample['image'], sample['mask']

        return image, mask

    def create_mask_from_yolo(self, txt_path, image_size=(256, 256)):
        mask = np.zeros(image_size, dtype=np.uint8)
        with open(txt_path, 'r') as file:
            lines = file.readlines()
        for line in lines:
            parts = line.strip().split()
            class_id, center_x, center_y, bbox_width, bbox_height = map(float, parts[:5])
            center_x = int(center_x * image_size[1])
            center_y = int(center_y * image_size[0])
            bbox_width = int(bbox_width * image_size[1])
            bbox_height = int(bbox_height * image_size[0])

            xmin = max(0, center_x - bbox_width // 2)
            ymin = max(0, center_y - bbox_height // 2)
            xmax = min(image_size[1], center_x + bbox_width // 2)
            ymax = min(image_size[0], center_y + bbox_height // 2)

            mask[ymin:ymax, xmin:xmax] = int(class_id)

        return mask

    def create_mask_from_yolo(self, txt_path, image_size=(256, 256)):
        mask = np.zeros(image_size, dtype=np.uint8)
        with open(txt_path, 'r') as file:
            lines = file.readlines()
        for line in lines:
            parts = line.strip().split()
            class_id, center_x, center_y, bbox_width, bbox_height = map(float, parts[:5])
            center_x = int(center_x * image_size[1])
            center_y = int(center_y * image_size[0])
            bbox_width = int(bbox_width * image_size[1])
            bbox_height = int(bbox_height * image_size[0])

            xmin = max(0, center_x - bbox_width // 2)
            ymin = max(0, center_y - bbox_height // 2)
            xmax = min(image_size[1], center_x + bbox_width // 2)
            ymax = min(image_size[0], center_y + bbox_height // 2)

            mask[ymin:ymax, xmin:xmax] = int(class_id)

        return mask

# Split the dataset
dataset = UrineStripDataset(image_folder, mask_folder)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)  # Reduce batch size for stability
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    
# Initialize the model
num_classes = 10  # Adjust based on your use case
unet_model = UNet(in_channels=3, out_channels=num_classes)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
unet_model.to(device)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(unet_model.parameters(), lr=0.001, weight_decay=1e-4)

# Mixed precision setup
scaler = GradScaler()

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(unet_model.parameters(), lr=0.001, weight_decay=1e-4)

# Define Early Stopping Parameters
patience = 20  # Stop if validation loss doesn't improve for 10 epochs
best_val_loss = float('inf')  # Initialize best validation loss
early_stop_counter = 0  # Track epochs without improvement

# Training Loop with Early Stopping
num_epochs = 50
torch.cuda.empty_cache()  # Clear cache to prevent memory issues

for epoch in range(num_epochs):
    unet_model.train()
    running_loss = 0.0
    
    for images, masks in tqdm(train_loader):
        images, masks = images.to(device), masks.to(device)
        images.requires_grad_(True)
        optimizer.zero_grad()

        # Training without mixed precision
        outputs = unet_model(images)
        loss = criterion(outputs, masks)

        if torch.isnan(loss) or torch.isinf(loss):
            print("NaN or Inf detected in loss! Exiting training.")
            break

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # Validation
    unet_model.eval()
    val_loss = 0.0
    correct_predictions = 0
    total_pixels = 0

    with torch.no_grad():
        for images, masks in tqdm(val_loader):
            images, masks = images.to(device), masks.to(device)
            outputs = unet_model(images)
            loss = criterion(outputs, masks)
            
            # Calculate accuracy
            predicted = torch.argmax(outputs, dim=1)
            correct_predictions += (predicted == masks).sum().item()
            total_pixels += masks.numel()  # Total number of pixels in the mask

            val_loss += loss.item()

    avg_train_loss = running_loss / len(train_loader)
    avg_val_loss = val_loss / len(val_loader)
    accuracy = 100 * correct_predictions / total_pixels  # Percentage of correct predictions

    print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Accuracy: {accuracy:.2f}%")

    # Early Stopping Logic
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        early_stop_counter = 0  # Reset counter
        torch.save(unet_model.state_dict(), model_filename)  # Save best model
        print("✅ Model improved and saved!")
    else:
        early_stop_counter += 1
        print(f"⚠️ No improvement in validation loss for {early_stop_counter}/{patience} epochs")

    # Stop training if no improvement for 'patience' epochs
    if early_stop_counter >= patience:
        print("⛔ Early stopping triggered! Training stopped.")
        break

# Save the model
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
