import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
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
import random
import matplotlib.pyplot as plt  # For plotting

torch._dynamo.config.suppress_errors = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.backends.cudnn.benchmark = True  # Enable cuDNN benchmark for performance
torch.backends.cudnn.enabled = True  # Enable cuDNN for performance

# Define the model filename with a timestamp
timestamp = time.strftime("%Y%m%d-%H%M%S")
model_filename = f"unet_model_{timestamp}.pth"

# -------------------------------
# Custom helper function for masks
# -------------------------------
def mask_to_tensor(mask):
    mask = np.array(mask, dtype=np.int64)
    return torch.from_numpy(mask)

# -------------------------------
# Define the U-Net model class
# -------------------------------
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
        enc1 = checkpoint.checkpoint(self.enc1, x, use_reentrant=False)
        enc2 = checkpoint.checkpoint(self.enc2, enc1, use_reentrant=False)
        enc3 = checkpoint.checkpoint(self.enc3, enc2, use_reentrant=False)
        enc4 = checkpoint.checkpoint(self.enc4, enc3, use_reentrant=False)
        bottleneck = checkpoint.checkpoint(self.bottleneck, enc4, use_reentrant=False)
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

# -------------------------------
# Data augmentation classes
# -------------------------------
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
            degrees=[-10, 10],
            translate=self.translate,
            scale_ranges=None,
            shears=None,
            img_size=image.size
        )
        image = transforms.functional.affine(image, angle=params[0], translate=params[1], scale=params[2], shear=params[3])
        mask = transforms.functional.affine(mask, angle=params[0], translate=params[1], scale=params[2], shear=params[3])
        return {'image': image, 'mask': mask}

# -------------------------------
# Define separate transformations for training and validation.
# -------------------------------
# Training uses heavy augmentation:
class RandomTrainTransformations:
    def __init__(self):
        self.joint_transform = transforms.Compose([
            RandomFlip(horizontal=True, vertical=True),
            RandomRotation(degrees=10),
            RandomAffine(translate=(0.1, 0.1))
        ])
        self.image_transform = transforms.Compose([
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
            transforms.RandomGrayscale(p=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225])
        ])
        self.mask_transform = transforms.Compose([mask_to_tensor])

    def __call__(self, sample):
        sample = self.joint_transform(sample)
        image = self.image_transform(sample['image'])
        mask = self.mask_transform(sample['mask'])
        return {'image': image, 'mask': mask}

# Validation uses only basic transforms:
class SimpleValTransformations:
    def __init__(self):
        self.image_transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225])
        ])
        self.mask_transform = transforms.Compose([
            transforms.Resize((128, 128), interpolation=Image.NEAREST),
            mask_to_tensor
        ])

    def __call__(self, sample):
        # For validation, we typically don't need joint spatial augmentation.
        image = self.image_transform(sample['image'])
        mask = self.mask_transform(sample['mask'])
        return {'image': image, 'mask': mask}

# -------------------------------
# Define your dataset.
# -------------------------------
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
        image = Image.open(image_path).convert("RGB")
        image = image.resize((128, 128))
        mask = self.create_mask_from_yolo(txt_path)
        mask = Image.fromarray(mask).resize((128, 128), Image.NEAREST)
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

# -------------------------------
# Main training loop
# -------------------------------
def main():
    image_folder = r"almost 1k dataset/train/images"
    mask_folder = r"almost 1k dataset/train/labels"

    # Create two datasets with different transforms:
    full_dataset = UrineStripDataset(image_folder, mask_folder, transform=None)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_indices, val_indices = list(range(train_size)), list(range(train_size, len(full_dataset)))
    
    train_dataset = torch.utils.data.Subset(
        UrineStripDataset(image_folder, mask_folder, transform=RandomTrainTransformations()),
        train_indices
    )
    val_dataset = torch.utils.data.Subset(
        UrineStripDataset(image_folder, mask_folder, transform=SimpleValTransformations()),
        val_indices
    )

    # DataLoaders with persistent workers:
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True,
                              num_workers=8, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False,
                            num_workers=8, pin_memory=True, persistent_workers=True)

    # Initialize the model and send to device
    num_classes = 10
    unet_model = UNet(in_channels=3, out_channels=num_classes)
    unet_model.to(device)

    scaler = GradScaler()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(unet_model.parameters(), lr=0.0002, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    patience = 10
    best_val_loss = float('inf')
    early_stop_counter = 0

    train_losses = []
    val_losses = []
    val_accuracies = []

    num_epochs = 100
    validation_interval = 2  # Validate every 2 epochs

    # Disable anomaly detection for production runs (uncomment if debugging)
    # torch.autograd.set_detect_anomaly(True)

    for epoch in range(num_epochs):
        # (Optional) Remove or reduce frequent calls to empty_cache:
        # torch.cuda.empty_cache()

        unet_model.train()
        running_loss = 0.0

        for images, masks in tqdm(train_loader, desc='Training Epoch', leave=False):
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            with autocast(device_type="cuda"):
                outputs = unet_model(images)
                loss = criterion(outputs, masks)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(unet_model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}")

        # Validation: runs every epoch in this example
        unet_model.eval()
        val_loss = 0.0
        correct_predictions = 0
        total_pixels = 0

        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc="Validation", leave=False):
                images, masks = images.to(device), masks.to(device)
                outputs = unet_model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()
                preds = torch.argmax(outputs, dim=1)
                correct_predictions += (preds == masks).sum().item()
                total_pixels += masks.numel()

        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        accuracy = 100 * correct_predictions / total_pixels
        val_accuracies.append(accuracy)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Val Loss: {avg_val_loss:.4f}, Val Accuracy: {accuracy:.2f}%")
        scheduler.step(avg_val_loss)

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

    torch.save(unet_model.state_dict(), model_filename)
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

if __name__ == '__main__':
    print(f"Using device: {device}")
    main()
