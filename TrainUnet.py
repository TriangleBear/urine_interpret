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
import torch._dynamo
from torch.amp import autocast, GradScaler
import torch.nn.functional as F
from tqdm import tqdm
from PIL import Image
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import random

torch._dynamo.config.suppress_errors = True

# Check if CUDA is available and set the device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

torch.backends.cudnn.benchmark = True  # Enable cuDNN benchmark for performance
torch.backends.cudnn.enabled = True  # Enable cuDNN for performance

# Define the paths to your image and mask folders
image_folder = r"almost 1k dataset/train/images"
mask_folder = r"almost 1k dataset/train/labels"

# Define the model filename with a timestamp
timestamp = time.strftime("%Y%m%d-%H%M%S")
model_filename = f"unet_model_{timestamp}.pth"

# Define the U-Net model class
class UNet(nn.Module):
    def __init__(self, in_channels, out_channels=10, dropout_prob=0.5):
        super(UNet, self).__init__()

        # Encoder path
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_prob)
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_prob)
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_prob)
        )
        self.enc4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_prob)
        )

        # Bottleneck layer
        self.bottleneck = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_prob)
        )

        # Decoder path
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)

        # Additional Conv2d layers
        self.conv_up3 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_prob)
        )
        self.conv_up2 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_prob)
        )
        self.conv_up1 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_prob)
        )

        # Output layer
        self.output = nn.Conv2d(64, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        # Encoder
        enc1 = checkpoint.checkpoint(self.enc1, x, use_reentrant=False)
        enc2 = checkpoint.checkpoint(self.enc2, enc1, use_reentrant=False)
        enc3 = checkpoint.checkpoint(self.enc3, enc2, use_reentrant=False)
        enc4 = checkpoint.checkpoint(self.enc4, enc3, use_reentrant=False)

        # Bottleneck
        bottleneck = checkpoint.checkpoint(self.bottleneck, enc4, use_reentrant=False)

        # Decoder
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

        # Output
        return self.output(up1)

# Data augmentation classes
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
        
        # Rotate image
        image = transforms.functional.rotate(image, angle)

        # Add a channel dimension to the mask
        mask = mask.unsqueeze(0)  # Shape: [1, H, W]
        
        # Rotate mask
        mask = transforms.functional.rotate(mask, angle)
        
        # Remove the channel dimension
        mask = mask.squeeze(0)  # Shape: [H, W]

        return {'image': image, 'mask': mask}

class RandomAffine:
    def __init__(self, translate=(0.1, 0.1)):
        self.translate = translate

    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']
        
        # Get parameters for affine transformation
        params = transforms.RandomAffine.get_params(
            degrees=[-10, 10],  # Random rotation between -10 and 10 degrees
            translate=self.translate,  # Random translation
            scale_ranges=None,  # No scaling
            shears=None,  # No shearing
            img_size=image.size()[1:]  # Image size (height, width)
        )
        
        # Apply affine transformation to image
        image = transforms.functional.affine(
            image, 
            angle=params[0],  # Rotation angle
            translate=params[1],  # Translation
            scale=params[2],  # Scale
            shear=params[3]  # Shear
        )

        # Add a channel dimension to the mask
        mask = mask.unsqueeze(0)  # Shape: [1, H, W]
        
        # Apply affine transformation to mask
        mask = transforms.functional.affine(
            mask, 
            angle=params[0],  # Rotation angle
            translate=params[1],  # Translation
            scale=params[2],  # Scale
            shear=params[3]  # Shear
        )
        
        # Remove the channel dimension
        mask = mask.squeeze(0)  # Shape: [H, W]

        return {'image': image, 'mask': mask}

class RandomTransformations:
    def __init__(self):
        self.transform = transforms.Compose([
            RandomFlip(horizontal=True, vertical=True),
            RandomRotation(degrees=10),  # Rotate by up to 10 degrees
            RandomAffine(translate=(0.1, 0.1)),  # Small translations
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),  # Color jitter
            transforms.RandomGrayscale(p=0.1),  # Randomly convert images to grayscale
            transforms.ToTensor(),  # Convert image to tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
        ])

    def __call__(self, sample):
        return self.transform(sample)

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

def main():
    # Split the dataset
    dataset = UrineStripDataset(image_folder, mask_folder, transform=RandomTransformations())
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

    # Initialize the model
    num_classes = 10
    unet_model = UNet(in_channels=3, out_channels=num_classes)
    unet_model = torch.compile(unet_model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    unet_model.to(device)

    # Mixed precision setup
    scaler = GradScaler()

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(unet_model.parameters(), lr=0.0001, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    # Early stopping parameters
    patience = 20
    best_val_loss = float('inf')
    early_stop_counter = 0

    # Lists to store loss and accuracy values
    train_losses = []
    val_losses = []
    val_accuracies = []

    # Training loop
    num_epochs = 100
    torch.cuda.empty_cache()

    torch.autograd.set_detect_anomaly(True)

    for epoch in range(num_epochs):
        torch.cuda.empty_cache()
        unet_model.train()
        running_loss = 0.0

        for images, masks in tqdm(train_loader, desc='Training Epoch', leave=False):
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

            # Backward pass
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)  # Unscale gradients before clipping
            torch.nn.utils.clip_grad_norm_(unet_model.parameters(), max_norm=1.0)  # Clip gradients
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation (keep torch.no_grad() here)
        unet_model.eval()
        val_loss = 0.0
        correct_predictions = 0
        total_pixels = 0

        with torch.no_grad():
            torch.cuda.memory_reserved(device)
            for images, masks in tqdm(val_loader, desc="Validation", leave=False):
                images, masks = images.to(device), masks.to(device)
                outputs = unet_model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()

                # Calculate accuracy
                preds = torch.argmax(outputs, dim=1)
                correct_predictions += (preds == masks).sum().item()
                total_pixels += masks.numel()

        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        accuracy = 100 * correct_predictions / total_pixels
        val_accuracies.append(accuracy)

        print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Accuracy: {accuracy:.2f}%")
        scheduler.step(avg_val_loss)

        # Early stopping logic
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            early_stop_counter = 0
            torch.save(unet_model.state_dict(), model_filename)
            print("✅ Model improved and saved!")
        else:
            early_stop_counter += 1
            print(f"⚠️ No improvement in validation loss for {early_stop_counter}/{patience} epochs")

        if early_stop_counter >= patience:
            print("⛔ Early stopping triggered! Training stopped.")
            break

    # Save the final model
    torch.save(unet_model.state_dict(), model_filename)

    # Plot the training and validation loss and accuracy
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')

    plt.subplot(1, 2, 2)
    plt.plot(epochs, val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.title('Validation Accuracy')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()