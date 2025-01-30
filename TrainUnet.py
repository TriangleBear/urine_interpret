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

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

# Define the paths to your image and mask folders
image_folder = r'D:\Programming\urine_interpret\Own\train\images'  # Replace with the actual path to the images
mask_folder = r'D:\Programming\urine_interpret\Own\train\labels'   # Replace with the actual path to the masks

timestamp = time.strftime("%Y%m%d-%H%M%S")
model_filename = f"unet_model_{timestamp}.pth"

# Define the U-Net model class
class UNet(nn.Module):
    def __init__(self, in_channels, out_channels=21):
        super(UNet, self).__init__()

        # Encoder path
        self.enc1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.enc2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.enc3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.enc4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)

        # Bottleneck layer
        self.bottleneck = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(p=0.2)

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
        enc1 = checkpoint.checkpoint(self.enc1, x)
        enc2 = checkpoint.checkpoint(self.enc2, enc1)
        enc3 = checkpoint.checkpoint(self.enc3, enc2)
        enc4 = checkpoint.checkpoint(self.enc4, enc3)

        bottleneck = checkpoint.checkpoint(self.bottleneck, enc4)
        bottleneck = self.dropout(bottleneck)

        up3 = checkpoint.checkpoint(self.upconv3, bottleneck)
        up3 = F.interpolate(up3, size=enc4.size()[2:], mode='bilinear', align_corners=True)
        up3 = torch.cat([up3, enc4], dim=1)
        up3 = self.conv_up3(up3)

        up2 = checkpoint.checkpoint(self.upconv2, up3)
        up2 = F.interpolate(up2, size=enc3.size()[2:], mode='bilinear', align_corners=True)
        up2 = torch.cat([up2, enc3], dim=1)
        up2 = self.conv_up2(up2)

        up1 = checkpoint.checkpoint(self.upconv1, up2)
        up1 = F.interpolate(up1, size=enc2.size()[2:], mode='bilinear', align_corners=True)
        up1 = torch.cat([up1, enc2], dim=1)
        up1 = self.conv_up1(up1)

        return self.output(up1)

# Dataset class
class UrineStripDataset(Dataset):
    def __init__(self, image_folder, mask_folder):
        self.image_folder = image_folder
        self.mask_folder = mask_folder
        self.image_files = sorted(os.listdir(image_folder))
        self.txt_files = sorted(os.listdir(mask_folder))

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

        image = cv2.resize(image, (128, 128))  # Try (128,128) or (256,256)

        # Convert to tensor (no augmentation)
        image = transforms.ToTensor()(image)

        # Load mask and convert to tensor
        mask = self.create_mask_from_yolo(txt_path)
        mask = cv2.resize(mask, (128, 128), interpolation=cv2.INTER_NEAREST)  # Resize mask
        mask = torch.tensor(mask, dtype=torch.long)  # Convert to tensor

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

# Split the dataset
dataset = UrineStripDataset(image_folder, mask_folder)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)  # Reduce batch size for stability
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

# Initialize the model
num_classes = 10  # Adjust based on your use case
unet_model = UNet(in_channels=3, out_channels=num_classes)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
unet_model.to(device)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(unet_model.parameters(), lr=1e-4)

# Mixed precision setup
scaler = GradScaler()

# Training loop
num_epochs = 100
torch.cuda.empty_cache()  # Clear cache to prevent memory issues
for epoch in range(num_epochs):
    unet_model.train()
    running_loss = 0.0
    for images, masks in tqdm(train_loader):
        images, masks = images.to(device), masks.to(device)

        optimizer.zero_grad()

        # Mixed precision
        with autocast(device_type='cuda', dtype=torch.float16):  # Use mixed precision
            outputs = unet_model(images)
            loss = criterion(outputs, masks)

        if torch.isnan(loss) or torch.isinf(loss):
            print("NaN or Inf detected in loss! Exiting training.")
            break

        optimizer.zero_grad()  # Reset gradients
        scaler.scale(loss).backward()  # Backpropagate loss
        scaler.step(optimizer)  # Update model weights
        scaler.update()  # Adjust scale for next iteration

        running_loss += loss.item()

    # Validation
    unet_model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, masks in tqdm(val_loader):
            images, masks = images.to(device), masks.to(device)
            with autocast('cuda'):
                outputs = unet_model(images)
                loss = criterion(outputs, masks)
            val_loss += loss.item()

    avg_train_loss = running_loss / len(train_loader)
    avg_val_loss = val_loss / len(val_loader)
    print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

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
