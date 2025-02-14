import torch
import torchvision.transforms as T
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from icecream import ic

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels=11, dropout_prob=0.5):
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
        enc1 = checkpoint.checkpoint(self.enc1, x, use_reentrant=False)
        enc2 = checkpoint.checkpoint(self.enc2, enc1, use_reentrant=False)
        enc3 = checkpoint.checkpoint(self.enc3, enc2, use_reentrant=False)
        enc4 = checkpoint.checkpoint(self.enc4, enc3, use_reentrant=False)
        bottleneck = checkpoint.checkpoint(self.bottleneck, enc4, use_reentrant=False)

        up3 = checkpoint.checkpoint(self.upconv3, bottleneck, use_reentrant=False)
        up3 = torch.cat([up3, enc4], dim=1)
        up3 = self.conv_up3(up3)

        up2 = checkpoint.checkpoint(self.upconv2, up3, use_reentrant=False)
        up2 = torch.cat([up2, enc3], dim=1)
        up2 = self.conv_up2(up2)

        up1 = checkpoint.checkpoint(self.upconv1, up2, use_reentrant=False)
        up1 = torch.cat([up1, enc2], dim=1)
        up1 = self.conv_up1(up1)

        output = self.output(up1)
        return output

# Load model
model = UNet(in_channels=3, out_channels=11)
state_dict = torch.load(r'unet_model_20250213-014930.pth', map_location=torch.device('cuda'))
model.load_state_dict(state_dict, strict=False)
model.to(torch.device('cuda'))
model.eval()
ic("model loaded")

# Load image
image_path = r'Datasets\Test test\test\IMG_2986_png.rf.5882528ac456a04ff79284960ca91129.jpg'
image = Image.open(image_path).convert("RGB")
ic("image loaded")

def dynamic_normalization(image):
    tensor_image = T.ToTensor()(image)
    mean = torch.mean(tensor_image, dim=[1, 2], keepdim=True)
    std = torch.std(tensor_image, dim=[1, 2], keepdim=True)
    std = torch.clamp(std, min=1e-6)
    normalize = T.Normalize(mean.flatten().tolist(), std.flatten().tolist())
    return normalize(tensor_image)

transform = T.Compose([
    T.Resize((512, 512), interpolation=T.InterpolationMode.BILINEAR),
    dynamic_normalization
])

image_tensor = transform(image).unsqueeze(0).to(torch.device('cuda'))
ic("image transformed")

# Run model
with torch.no_grad():
    prediction = model(image_tensor)

# Convert mask to bounding boxes
if prediction.ndim == 4:
    mask = prediction.squeeze().cpu().numpy()
    mask = np.argmax(mask, axis=0)  # Get class-wise segmentation

    image_np = np.array(image.resize((512, 512)))
    image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    for class_id in range(1, 10):  # Ignore background class (0)
        binary_mask = (mask == class_id).astype(np.uint8) * 255

        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(image_np, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(image_np, f"Class {class_id}", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Show results
    plt.figure(figsize=(8, 8))
    plt.imshow(cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB))
    plt.title("Bounding Box Predictions")
    plt.axis("off")
    plt.show()

else:
    ic("Model output is not a mask. Skipping visualization.")
