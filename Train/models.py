import torch
import torch.nn as nn
import torch.nn.functional as F
from config import NUM_CLASSES  # Change to direct import since we're already in the Train directory

class DoubleConv(nn.Module):
    """Standard double convolution block used in UNet."""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, bilinear=False):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if (bilinear):
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=False, dropout_prob=0.5):
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bilinear = bilinear
        
        # Initialize with lower number of features to save memory
        factor = 2 if bilinear else 1
        
        # Encoder path
        self.inc = DoubleConv(in_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024 // factor)
        
        # Dropout for regularization
        self.dropout = nn.Dropout2d(p=dropout_prob)
        
        # Decoder path with skip connections
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        
        # Output layer
        self.outc = nn.Conv2d(64, out_channels, kernel_size=1)
        
        # Initialize weights to avoid exploding losses
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights for stable training."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Store features from encoder path for skip connections
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # Apply dropout at the bottleneck
        x5 = self.dropout(x5)
        
        # Decoder path with skip connections
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        # Final output layer
        x = self.outc(x)
        
        return x

class YOLOHead(nn.Module):
    """YOLO-style detection head to apply on top of UNet features."""
    def __init__(self, in_channels, num_classes):
        super(YOLOHead, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels*2, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels*2),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(in_channels*2, num_classes, kernel_size=1)
        )
        
        # Initialize weights with Kaiming initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu', a=0.2)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        features = self.conv(x)
        return features

class UNetYOLO(nn.Module):  
    def __init__(self, in_channels, out_channels, dropout_prob=0.5):  
        """Combined UNet with YOLO detection head for segmentation.""" 
        super(UNetYOLO, self).__init__()
        self.unet = UNet(in_channels, 64, bilinear=False, dropout_prob=dropout_prob)
        
        # Expose decoder layers
        self.decoder = nn.ModuleList([self.unet.up1, self.unet.up2, self.unet.up3, self.unet.up4])
        
        # Add batch normalization after UNet for better stability
        self.bn = nn.BatchNorm2d(64)
        
        # YOLO head for segmentation
        self.yolo_head = YOLOHead(64, out_channels)  # YOLO head for segmentation

        self._init_weights()

    def _init_weights(self):
        # More careful initialization for better convergence
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Get UNet features
        features = self.unet(x)
        features = self.bn(features)
        
        # Get segmentation map from YOLO head
        segmentation_map = self.yolo_head(features)
        
        return segmentation_map  # Return the final segmentation map
