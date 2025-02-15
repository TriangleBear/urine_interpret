import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels=11, dropout_prob=0.2):
        super(UNet, self).__init__()
        # Encoder blocks
        self.enc1 = self._block(in_channels, 32, dropout_prob)
        self.enc2 = self._block(32, 64, dropout_prob)
        self.enc3 = self._block(64, 128, dropout_prob)
        self.enc4 = self._block(128, 256, dropout_prob)
        
        # Bottleneck
        self.bottleneck = self._block(256, 512, dropout_prob)
        
        # Decoder blocks with upsampling
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        
        # Decoder convolution blocks
        self.conv_up3 = self._block(512, 256, dropout_prob)
        self.conv_up2 = self._block(256, 128, dropout_prob)
        self.conv_up1 = self._block(128, 64, dropout_prob)
        
        # Final output layer
        self.output = nn.Conv2d(64, out_channels, kernel_size=1)
        self.initialize_weights()

    def _block(self, in_channels, features, dropout_prob):
        return nn.Sequential(
            nn.Conv2d(in_channels, features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(features),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Dropout(p=dropout_prob)
        )

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='leaky_relu')

    def forward(self, x):
        # Encoder path
        enc1 = self.enc1(x)
        enc2 = self.enc2(F.max_pool2d(enc1, 2))
        enc3 = self.enc3(F.max_pool2d(enc2, 2))
        enc4 = self.enc4(F.max_pool2d(enc3, 2))
        
        # Bottleneck
        bottleneck = self.bottleneck(F.max_pool2d(enc4, 2))
        
        # Decoder path with skip connections
        dec3 = self.upconv3(bottleneck)
        dec3 = torch.cat([dec3, enc4], dim=1)
        dec3 = self.conv_up3(dec3)
        
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat([dec2, enc3], dim=1)
        dec2 = self.conv_up2(dec2)
        
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat([dec1, enc2], dim=1)
        dec1 = self.conv_up1(dec1)
        
        return self.output(dec1)