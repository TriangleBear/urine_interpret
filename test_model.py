import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import os
import cv2
import matplotlib.pyplot as plt
import random

# -------------------------------
# Model Definition (UNet)
# -------------------------------
class UNet(nn.Module):
    def __init__(self, in_channels, out_channels=10):
        super(UNet, self).__init__()

        # Encoder path
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.enc4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        # Bottleneck layer
        self.bottleneck = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.dropout = nn.Dropout(p=0.2)

        # Decoder path
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)

        # Additional Conv2d layers after concatenation
        self.conv_up3 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.conv_up2 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.conv_up1 = nn.Conv2d(128, 64, kernel_size=3, padding=1)

        # Output layer
        self.output = nn.Conv2d(64, out_channels, kernel_size=1)

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
        up3 = self.conv_up3(up3)

        up2 = self.upconv2(up3)
        up2 = F.interpolate(up2, size=enc3.size()[2:], mode='bilinear', align_corners=True)
        up2 = torch.cat([up2, enc3], dim=1)
        up2 = self.conv_up2(up2)

        up1 = self.upconv1(up2)
        up1 = F.interpolate(up1, size=enc2.size()[2:], mode='bilinear', align_corners=True)
        up1 = torch.cat([up1, enc2], dim=1)
        up1 = self.conv_up1(up1)

        output = self.output(up1)
        return output   

# -------------------------------
# Load the Model
# -------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = UNet(in_channels=3, out_channels=10)

checkpoint = torch.load('unet_model_20250209-223510.pth', map_location=device)

# If the checkpoint contains a 'state_dict' key, use it; otherwise, use the checkpoint directly.
if 'state_dict' in checkpoint:
    checkpoint = checkpoint['state_dict']

# Optionally remove mismatched keys (for example, those in the output layer)
checkpoint = {k: v for k, v in checkpoint.items() if 'output' not in k}
model.load_state_dict(checkpoint, strict=False)
model = model.to(device)
model.eval()

# -------------------------------
# Test Data Preparation
# -------------------------------
def load_image(image_path):
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    image = transform(image)
    image = image.unsqueeze(0)  # add batch dimension
    return image

def load_images_from_folder(folder_path):
    images = []
    filenames = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(folder_path, filename)
            images.append(load_image(image_path))
            filenames.append(image_path)
    if images:
        images = torch.cat(images)
    return images, filenames

# -------------------------------
# Bounding Box Creation Functions
# -------------------------------
def dynamic_threshold(pred):
    pred_scaled = (pred * 255).astype(np.uint8)  # Convert to 0-255 scale
    _, binary_image = cv2.threshold(pred_scaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    threshold = np.mean(binary_image) / 255  # Get mean value instead of using an array
    return max(threshold, 0.4)  # Ensure threshold is not too low

def create_bounding_boxes(predictions, dynamic=True):
    bounding_boxes = []
    class_ids = []
    urine_strip_classes = 9
    for i in range(predictions.shape[0]):  # Iterate over batch
        pred = predictions[i].cpu().numpy()

        # 🔍 DEBUG: Show the predicted segmentation mask
        plt.figure(figsize=(10, 5))
        for c in range(pred.shape[0]):  # Loop through each class
            plt.subplot(1, pred.shape[0], c+1)
            plt.imshow(pred[c], cmap='gray')
            plt.title(f"Class {c}")
            plt.colorbar()
        plt.show()

        confidence_map = pred[urine_strip_classes]  # Highest confidence per pixel
        threshold = dynamic_threshold(confidence_map) if dynamic else 0.6  # Use fixed threshold
        binary_mask = (confidence_map > threshold).astype(np.uint8)

        # 🔍 DEBUG: Show binary mask before contour detection
        plt.figure(figsize=(6, 6))
        plt.imshow(binary_mask, cmap='gray')
        plt.title("Binary Mask Before Contour Detection")
        plt.colorbar()
        plt.show()

        # Find contours
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        classes = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w * h < (binary_mask.shape[0] * binary_mask.shape[1] * 0.8):  # Ignore huge detections
                class_map = np.argmax(pred, axis=0)  # Most frequent class per pixel
                mask_region = class_map[y:y+h, x:x+w]
                class_id = np.argmax(np.bincount(mask_region.flatten()))  # Most common class
                boxes.append((x, y, w, h))
                classes.append(class_id)
        
        bounding_boxes.append(boxes)
        class_ids.append(classes)
    return bounding_boxes, class_ids

def draw_and_show_bounding_boxes(image_path, boxes, class_ids, min_area_ratio=0.001):
    """
    Loads the original image using cv2, rescales the bounding boxes from the model input size (256x256)
    to the original image size, and draws the boxes.
    """
    image = cv2.imread(image_path)
    orig_h, orig_w = image.shape[:2]
    # Our model input size is 256x256.
    h, w = image_tensor.shape[2:]  # Get input size
    scale_x = orig_w / w
    scale_y = orig_h / h

    for box_list, class_list in zip(boxes, class_ids):
        for (x, y, w, h), cls in zip(box_list, class_list):
            if w * h > (min_area_ratio * orig_w * orig_h) / 2:  # Reduce minimum size filter
                x1 = int(x * scale_x)
                y1 = int(y * scale_y)
                x2 = int((x + w) * scale_x)
                y2 = int((y + h) * scale_y)
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.imshow("Bounding Boxes", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# -------------------------------
# Test Model and Draw Boxes
# -------------------------------
test_folder_path = r'D:\Programming\urine_interpret\urine\test\images'
test_data, filenames = load_images_from_folder(test_folder_path)

# Process each test image individually
for image_path in filenames:
    image_tensor = load_image(image_path).to(device)
    with torch.no_grad():
        # Get model output and apply sigmoid (if using multi-label segmentation)
        prediction = torch.sigmoid(model(image_tensor))
    # For this example, use channel 0 for bounding boxes; change as needed.
    boxes, cls_ids = create_bounding_boxes(prediction, dynamic=True)
    draw_and_show_bounding_boxes(image_path, boxes, cls_ids)

print("Bounding boxes created and images displayed.")
