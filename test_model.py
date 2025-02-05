import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import os
import cv2
import matplotlib.pyplot as plt

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

# Load the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = UNet(in_channels=3, out_channels=10)

checkpoint = torch.load('models/unet_model_20250201-132802.pth', map_location=device)

# Check if it's a full model or just the state_dict
if 'state_dict' in checkpoint:
    checkpoint = checkpoint['state_dict']

# Remove the output layer weights from the checkpoint to prevent mismatch
checkpoint = {k: v for k, v in checkpoint.items() if 'output' not in k}

# Load the weights into the model
model.load_state_dict(checkpoint, strict=False)
model = model.to(device)
model.eval()

# Prepare test data
def load_image(image_path):
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    image = transform(image)
    image = image.unsqueeze(0)
    return image

# Load all images from the specified folder
def load_images_from_folder(folder_path):
    images = []
    for filename in os.listdir(folder_path):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(folder_path, filename)
            images.append(load_image(image_path))
    return images

# Dynamic threshold function
def dynamic_threshold(pred, percentile=95):
    return np.percentile(pred, percentile)

# Updated bounding box scaling function
def scale_and_clip_boxes(boxes, orig_shape, pred_shape):
    orig_h, orig_w = orig_shape
    pred_h, pred_w = pred_shape
    scale_x = orig_w / pred_w
    scale_y = orig_h / pred_h

    scaled_boxes = []
    for x, y, w, h in boxes:
        x = max(0, int(x * scale_x))
        y = max(0, int(y * scale_y))
        w = min(orig_w - x, int(w * scale_x))
        h = min(orig_h - y, int(h * scale_y))
        scaled_boxes.append((x, y, w, h))
    return scaled_boxes

# Updated bounding box creation with dynamic threshold
def create_bounding_boxes(predictions, dynamic=True, default_threshold=0.5, percentile=95):
    bounding_boxes = []
    class_ids = []
    for i in range(predictions.shape[0]):
        pred = predictions[i, 0].cpu().numpy()

        if dynamic:
            threshold = dynamic_threshold(pred, percentile)
        else:
            threshold = default_threshold

        pred = (pred > threshold).astype(np.uint8)
        contours, _ = cv2.findContours(pred, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = [cv2.boundingRect(contour) for contour in contours]
        bounding_boxes.append(boxes)
        class_ids.append([i] * len(boxes))  # Assuming class ID is the same for all boxes in this prediction
    return bounding_boxes, class_ids

# Draw and show bounding boxes without class names
def draw_and_show_bounding_boxes(image_path, boxes, class_ids, min_area_ratio=0.001, threshold=0.3):
    image = cv2.imread(image_path)
    orig_h, orig_w = image.shape[:2]
    scale_x = orig_w / 256
    scale_y = orig_h / 256

    for box_list, class_list in zip(boxes, class_ids):
        for (x, y, w, h), class_id in zip(box_list, class_list):
            if w * h > min_area_ratio * orig_w * orig_h:
                x1, y1, x2, y2 = int(x * scale_x), int(y * scale_y), int((x + w) * scale_x), int((y + h) * scale_y)
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imshow("Bounding Boxes", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Replace this with your actual test folder path
test_folder_path = r'D:\Programming\urine_interpret\urine\test\images'

test_data = load_images_from_folder(test_folder_path)
test_data = torch.cat(test_data).to(device)

# Run the model on the test data
with torch.no_grad():
    predictions = torch.sigmoid(model(test_data))
    print(f"Model Output Shape: {predictions.shape}")
    print(f"Sample Prediction Output: {predictions[0, 0].cpu().numpy()}")

# Create bounding boxes
bounding_boxes, class_ids = create_bounding_boxes(predictions)

# Draw bounding boxes and display the images
for image_path in os.listdir(test_folder_path):
    if image_path.endswith(('.png', '.jpg', '.jpeg')):
        full_image_path = os.path.join(test_folder_path, image_path)
        image = load_image(full_image_path).to(device)

        with torch.no_grad():
            prediction = torch.sigmoid(model(image))
        
        bounding_boxes, class_ids = create_bounding_boxes(prediction)
        draw_and_show_bounding_boxes(full_image_path, bounding_boxes, class_ids)

print("Bounding boxes created and images displayed.")
