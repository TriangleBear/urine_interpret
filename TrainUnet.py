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
import random

# Ensure cuDNN settings for performance
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

# Define paths
image_folder = r"almost 1k dataset/train/images"
mask_folder = r"almost 1k dataset/train/labels"
timestamp = time.strftime("%Y%m%d-%H%M%S")
model_filename = f"unet_model_{timestamp}.pth"

# Define U-Net model with proper initialization
class UNet(nn.Module):
    def __init__(self, in_channels, out_channels=10):
        super(UNet, self).__init__()

        # Encoder path
        self.enc1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False)
        self.enc2 = nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False)
        self.enc3 = nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False)
        self.enc4 = nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False)

        # Bottleneck layer
        self.bottleneck = nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=False)
        self.dropout = nn.Dropout(p=0.5)

        # Decoder path
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)

        # Output layer
        self.output = nn.Conv2d(64, out_channels, kernel_size=1, bias=False)

        # Initialize weights
        self.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)

        bottleneck = self.bottleneck(enc4)
        bottleneck = self.dropout(bottleneck)

        up3 = self.upconv3(bottleneck)
        up3 = F.interpolate(up3, size=enc4.size()[2:], mode='bilinear', align_corners=True)
        up3 = torch.cat([up3, enc4], dim=1)

        up2 = self.upconv2(up3)
        up2 = F.interpolate(up2, size=enc3.size()[2:], mode='bilinear', align_corners=True)
        up2 = torch.cat([up2, enc3], dim=1)

        up1 = self.upconv1(up2)
        up1 = F.interpolate(up1, size=enc2.size()[2:], mode='bilinear', align_corners=True)
        up1 = torch.cat([up1, enc2], dim=1)

        return self.output(up1)

# Dataset class with NaN and Inf checks
class UrineStripDataset(Dataset):
    def __init__(self, image_folder, mask_folder, transform=None):
        self.image_folder = image_folder
        self.mask_folder = mask_folder
        self.image_files = sorted(os.listdir(image_folder))
        self.txt_files = sorted(os.listdir(mask_folder))
        self.transform = transform
        if len(self.image_files) != len(self.txt_files):
            raise ValueError("Mismatch between number of images and masks")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        image_file = self.image_files[index]
        txt_file = self.txt_files[index]
        image_path = os.path.join(self.image_folder, image_file)
        txt_path = os.path.join(self.mask_folder, txt_file)

        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Image at {image_path} could not be loaded")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (128, 128))
        image = transforms.ToTensor()(image)

        # Check for NaN or inf in image
        if torch.isnan(image).any() or torch.isinf(image).any():
            raise ValueError(f"Image at {image_path} contains NaN or inf values")

        # Load mask
        mask = self.create_mask_from_yolo(txt_path)
        mask = cv2.resize(mask, (128, 128), interpolation=cv2.INTER_NEAREST)
        mask = torch.tensor(mask, dtype=torch.long).squeeze(0)

        # Check for NaN or inf in mask
        if torch.isnan(mask).any() or torch.isinf(mask).any():
            raise ValueError(f"Mask at {txt_path} contains NaN or inf values")

        if self.transform:
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

# Main training loop
def main():
    # Split the dataset
    dataset = UrineStripDataset(image_folder, mask_folder)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)

    # Initialize model, loss, and optimizer
    unet_model = UNet(in_channels=3, out_channels=10)
    unet_model = torch.compile(unet_model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    unet_model.to(device)

    # Mixed precision setup
    scaler = GradScaler()

    # Loss function and optimizer with reduced learning rate
    optimizer = torch.optim.Adam(unet_model.parameters(), lr=0.0001, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    for epoch in range(10):  # Example number of epochs
        running_loss = 0.0
        for images, masks in tqdm(train_loader):
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()

            # Forward pass
            with autocast(device_type="cuda"):
                outputs = unet_model(images)
                
                # Check for NaN or inf in outputs
                if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                    raise ValueError("Model outputs contain NaN or inf values")

                loss = criterion(outputs, masks)
                
                # Check for NaN or inf in loss
                if torch.isnan(loss).any() or torch.isinf(loss).any():
                    raise ValueError("Loss contains NaN or inf values")

            # Backward pass with gradient clipping
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)  # Unscale gradients before clipping
            torch.nn.utils.clip_grad_norm_(unet_model.parameters(), max_norm=1.0)  # Clip gradients
            scaler.step(optimizer)
            scaler.update()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}")

    # Save the trained model
    torch.save(unet_model.state_dict(), model_filename)
    print(f"Model saved to {model_filename}")

# Run the training loop
if __name__ == "__main__":
    main()
