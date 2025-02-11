import torch
import torchvision.transforms as T
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from PIL import Image
from icecream import ic
import numpy as np

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels=10, dropout_prob=0.5):
        super(UNet, self).__init__()

        # Encoder path with BatchNorm and LeakyReLU
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),  # 🔥 Added BatchNorm
            nn.LeakyReLU(negative_slope=0.01, inplace=True),  # 🔥 Changed to LeakyReLU
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

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Dropout(p=dropout_prob)
        )

        # Decoder path
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

        # Output layer
        self.output = nn.Conv2d(64, out_channels, kernel_size=1, bias=False)

        # 🚀 Apply Xavier Initialization
        self.initialize_weights()

    def initialize_weights(self):
        """Apply Kaiming Initialization for Conv2d and ConvTranspose2d layers (better for LeakyReLU)."""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                # Use kaiming_uniform_ with 'leaky_relu' nonlinearity
                nn.init.kaiming_uniform_(m.weight, nonlinearity='leaky_relu')


    def forward(self, x):
        enc1 = checkpoint.checkpoint(self.enc1, x, use_reentrant=False)
        #print(f"🔍 Activation after enc1: mean={enc1.abs().mean().item()}, max={enc1.abs().max().item()}")

        enc2 = checkpoint.checkpoint(self.enc2, enc1, use_reentrant=False)
        #print(f"🔍 Activation after enc2: mean={enc2.abs().mean().item()}, max={enc2.abs().max().item()}")

        enc3 = checkpoint.checkpoint(self.enc3, enc2, use_reentrant=False)
        #print(f"🔍 Activation after enc3: mean={enc3.abs().mean().item()}, max={enc3.abs().max().item()}")

        enc4 = checkpoint.checkpoint(self.enc4, enc3, use_reentrant=False)
        #print(f"🔍 Activation after enc4: mean={enc4.abs().mean().item()}, max={enc4.abs().max().item()}")

        bottleneck = checkpoint.checkpoint(self.bottleneck, enc4, use_reentrant=False)
        #print(f"🔍 Activation after bottleneck: mean={bottleneck.abs().mean().item()}, max={bottleneck.abs().max().item()}")

        up3 = checkpoint.checkpoint(self.upconv3, bottleneck, use_reentrant=False)
        up3 = F.interpolate(up3, size=enc4.size()[2:], mode='bilinear', align_corners=True)
        up3 = torch.cat([up3, enc4], dim=1)
        up3 = self.conv_up3(up3)
        #print(f"🔍 Activation after up3: mean={up3.abs().mean().item()}, max={up3.abs().max().item()}")

        up2 = checkpoint.checkpoint(self.upconv2, up3, use_reentrant=False)
        up2 = F.interpolate(up2, size=enc3.size()[2:], mode='bilinear', align_corners=True)
        up2 = torch.cat([up2, enc3], dim=1)
        up2 = self.conv_up2(up2)
        #print(f"🔍 Activation after up2: mean={up2.abs().mean().item()}, max={up2.abs().max().item()}")

        up1 = checkpoint.checkpoint(self.upconv1, up2, use_reentrant=False)
        up1 = F.interpolate(up1, size=enc2.size()[2:], mode='bilinear', align_corners=True)
        up1 = torch.cat([up1, enc2], dim=1)
        up1 = self.conv_up1(up1)
        #print(f"🔍 Activation after up1: mean={up1.abs().mean().item()}, max={up1.abs().max().item()}")

        output = self.output(up1)
        #print(f"🔍 Activation after output: mean={output.abs().mean().item()}, max={output.abs().max().item()}")

        return output

# Define the model first
model = UNet(in_channels=3, out_channels=10)  # Ensure correct class count

# Load the trained weights
state_dict = torch.load(r'models\bestmodel_DONOTDELETEME.pth', map_location=torch.device('cuda'))
model.load_state_dict(state_dict)

# Move model to GPU and set to evaluation mode
model.to(torch.device('cuda'))
model.eval()
ic("model loaded")


# Load and preprocess the test image
image_path = r'Datasets\outputGab\IMG_2979.png'  # Change this to your image
image = Image.open(image_path).convert("RGB")
ic("image loaded")

def dynamic_normalization(image):
    tensor_image = T.ToTensor()(image)  # Convert to tensor first
    mean = torch.mean(tensor_image, dim=[1, 2], keepdim=True)  # Compute per-channel mean
    std = torch.std(tensor_image, dim=[1, 2], keepdim=True)  # Compute per-channel std
    std = torch.clamp(std, min=1e-6)  # Avoid division by zero

    normalize = T.Normalize(mean.flatten().tolist(), std.flatten().tolist())  # Normalize dynamically
    return normalize(tensor_image)

transform = T.Compose([
    T.Resize((512, 512), interpolation=T.InterpolationMode.BILINEAR),  # Resize for consistency
    #T.ToTensor(),
    dynamic_normalization
])

image_tensor = transform(image).unsqueeze(0).to(torch.device('cuda'))  # Add batch dimension & move to GPU
ic("image transformed")

with torch.no_grad():
    prediction = model(image_tensor)

# Show original image and mask side by side
if prediction.ndim == 4:
    mask = prediction.squeeze().cpu().numpy()
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].imshow(image)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    axes[1].imshow(mask[0], cmap="gray")
    axes[1].set_title("Predicted Mask")
    axes[1].axis("off")

    # # Compute accuracy (assuming softmax-like output)
    # predicted_class = np.argmax(mask, axis=0)
    # accuracy = (predicted_class == predicted_class.max()).mean() * 100

    # # Add accuracy text
    # plt.figtext(0.5, 0.01, f"Prediction Accuracy: {accuracy:.2f}%", ha="center", fontsize=12, bbox={"facecolor":"white", "alpha":0.5, "pad":5})

    plt.show()


else:
    ic("Model output is not a mask. Skipping visualization.")
