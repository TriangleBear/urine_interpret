import torch
import torchvision.transforms as T
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch.serialization
from PIL import Image, ImageTk
from ultralytics.nn.tasks import DetectionModel  # Import the required module
import tkinter as tk
from tkinter import ttk

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CONFIDENCE = 0.5
IOU_THRESHOLD = 0.5

# Define the UNet model
class UNet(nn.Module):
    def __init__(self, in_channels, out_channels=11, dropout_prob=0.3):
        super(UNet, self).__init__()
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Dropout(p=dropout_prob)
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Dropout(p=dropout_prob)
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Dropout(p=dropout_prob)
        )
        self.enc4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Dropout(p=dropout_prob)
        )
        self.bottleneck = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Dropout(p=dropout_prob)
        )
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv_up3 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Dropout(p=dropout_prob)
        )
        self.conv_up2 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Dropout(p=dropout_prob)
        )
        self.conv_up1 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Dropout(p=dropout_prob)
        )
        self.output = nn.Conv2d(64, out_channels, kernel_size=1, bias=False)
        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='leaky_relu')

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        bottleneck = self.bottleneck(enc4)
        
        up3 = self.upconv3(bottleneck)
        up3 = F.interpolate(up3, size=enc4.size()[2:], mode='bilinear', align_corners=False)
        up3 = torch.cat([up3, enc4], dim=1)
        up3 = self.conv_up3(up3)

        up2 = self.upconv2(up3)
        up2 = F.interpolate(up2, size=enc3.size()[2:], mode='bilinear', align_corners=False)
        up2 = torch.cat([up2, enc3], dim=1)
        up2 = self.conv_up2(up2)

        up1 = self.upconv1(up2)
        up1 = F.interpolate(up1, size=enc2.size()[2:], mode='bilinear', align_corners=False)
        up1 = torch.cat([up1, enc2], dim=1)
        up1 = self.conv_up1(up1)

        output = self.output(up1)
        return output

# Dynamic normalization: computes per-image mean and std.
def dynamic_normalization(image):
    image = image.resize((256, 256))
    tensor_image = T.ToTensor()(image)
    mean = torch.mean(tensor_image, dim=[1, 2], keepdim=True)
    std = torch.std(tensor_image, dim=[1, 2], keepdim=True)
    std = torch.clamp(std, min=1e-6)
    normalize = T.Normalize(mean.squeeze().tolist(), std.squeeze().tolist())
    return normalize(tensor_image)

# Fixed normalization: using fixed mean/std (e.g., ImageNet values)
def fixed_normalization(image):
    image = image.resize((256, 256))
    tensor_image = T.ToTensor()(image)
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], 
                            std=[0.229, 0.224, 0.225])
    return normalize(tensor_image)

def load_model(model_path):
    model = UNet(in_channels=3, out_channels=11)
    torch.serialization.add_safe_globals([DetectionModel])  # Add DetectionModel to safe globals
    state_dict = torch.load(model_path, map_location=device, weights_only=False)  # Set weights_only to False
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    return model

def draw_bounding_boxes(image_np, mask, confidence_map, unique_classes, confidence_threshold):
    for class_id in unique_classes:
        # Create a binary mask for the current class
        binary_mask = (mask == class_id).astype(np.uint8) * 255
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        print(f"Class {class_id}: Found {len(contours)} contours")  # Debugging output
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            # Calculate average confidence within this bounding box
            region_confidence = confidence_map[y:y+h, x:x+w]
            avg_conf = np.mean(region_confidence)
            if avg_conf >= confidence_threshold:
                # Label accordingly: reagent pads vs test strip
                if class_id == 20:
                    label = f"Test Strip ({avg_conf:.2f})"
                    color = (0, 255, 0)  # Green for test strip
                else:
                    label = f"Pad {class_id} ({avg_conf:.2f})"
                    color = (255, 0, 255)  # Magenta for reagent pads

                print(f"{label}: x={x}, y={y}, w={w}, h={h}")  # Debugging output
                cv2.rectangle(image_np, (x, y), (x+w, y+h), color, 2)
                cv2.putText(image_np, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

def update_image_on_canvas(image_np):
    image_pil = Image.fromarray(cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB))
    image_tk = ImageTk.PhotoImage(image_pil)
    canvas.itemconfig(image_on_canvas, image=image_tk)
    canvas.image = image_tk

def display_image_with_bboxes(image_np):
    plt.figure(figsize=(8, 8))
    plt.imshow(cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB))
    plt.title("Bounding Box Predictions")
    plt.axis("off")
    plt.show()

def predict_and_visualize(model, image_path, norm_method='dynamic', confidence_threshold=0.5):
    image = Image.open(image_path).convert("RGB")
    
    # Choose normalization method
    if norm_method == 'dynamic':
        print("Using dynamic normalization...")
        image_tensor = dynamic_normalization(image).unsqueeze(0).to(device)
    else:
        print("Using fixed normalization...")
        image_tensor = fixed_normalization(image).unsqueeze(0).to(device)

    with torch.no_grad():
        prediction = model(image_tensor)
        prediction = F.softmax(prediction, dim=1)  # Softmax along channel dimension
        prediction_np = prediction.squeeze().cpu().numpy()  # shape: (11, H, W)
        print("Prediction tensor:", prediction_np)

    # Create segmentation mask (pixel-wise predicted class)
    mask = np.argmax(prediction_np, axis=0)  # shape: (H, W)
    unique_classes = np.unique(mask)
    print("Unique classes in mask:", unique_classes)

    # Create confidence map (max probability per pixel)
    confidence_map = np.max(prediction_np, axis=0)
    mean_confidence = np.mean(confidence_map)
    print(f"Mean confidence: {mean_confidence:.4f}")

    # Draw bounding boxes for each class.
    # Classes 0-9 are reagent pads, and class 20 is the whole test strip.
    image_np = np.array(image.resize((256, 256)))
    image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    draw_bounding_boxes(image_np, mask, confidence_map, unique_classes, confidence_threshold)
    update_image_on_canvas(image_np)

def update_confidence_threshold(val):
    confidence_threshold = float(val)
    predict_and_visualize(model, image_path, norm_method='fixed', confidence_threshold=confidence_threshold)

if __name__ == "__main__":
    model_path = r'D:\Programming\urine_interpret\models\weights.pt'  # Updated path to weight.pt
    image_path = r"D:\Programming\urine_interpret\Datasets\outputGab\IMG_2983.png"
    
    model = load_model(model_path)
    
    # Create a tkinter window
    root = tk.Tk()
    root.title("Confidence Threshold Adjuster")

    # Create a canvas to display the image
    canvas = tk.Canvas(root, width=256, height=256)
    canvas.pack()

    # Initialize the image on the canvas
    image_np = np.zeros((256, 256, 3), dtype=np.uint8)
    image_pil = Image.fromarray(image_np)
    image_tk = ImageTk.PhotoImage(image_pil)
    image_on_canvas = canvas.create_image(0, 0, anchor=tk.NW, image=image_tk)

    # Create a scale widget with higher resolution
    scale = tk.Scale(root, from_=0.0, to=1.0, orient='horizontal', command=update_confidence_threshold, length=300, resolution=0.01)
    scale.set(0.5)  # Set initial value to 0.5
    scale.pack()

    # Run the tkinter main loop
    root.mainloop()
