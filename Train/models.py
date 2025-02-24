import torch
import torch.nn as nn
import torch.nn.functional as F

class UNetEncoder(nn.Module):
    def __init__(self, in_channels, dropout_prob=0.3):
        super(UNetEncoder, self).__init__()
        self.enc1 = self.conv_block(in_channels, 32, dropout_prob)
        self.enc2 = self.conv_block(32, 64, dropout_prob)
        self.enc3 = self.conv_block(64, 128, dropout_prob)
        self.enc4 = self.conv_block(128, 256, dropout_prob)
        self.bottleneck = self.conv_block(256, 512, dropout_prob)

    def conv_block(self, in_channels, out_channels, dropout_prob):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Dropout(p=dropout_prob)
        )

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        bottleneck = self.bottleneck(enc4)
        return bottleneck, enc1, enc2, enc3, enc4

class YOLODecoder(nn.Module):
    def __init__(self, num_classes):
        super(YOLODecoder, self).__init__()
        
        self.upconv1 = self.upconv_block(512 + 256, 256)  # Skip connection with enc4
        self.upconv2 = self.upconv_block(256 + 128, 128)  # Skip connection with enc3
        self.upconv3 = self.upconv_block(128 + 64, 64)    # Skip connection with enc2
        self.upconv4 = self.upconv_block(64 + 32, 32)     # Skip connection with enc1
        
        self.final_conv = nn.Conv2d(32, num_classes + 5, kernel_size=1)  # (x, y, w, h, conf) + class scores

    def upconv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True)
        )

    def forward(self, x, enc1, enc2, enc3, enc4):
        x = self.upconv1(torch.cat([x, enc4], dim=1))
        x = self.upconv2(torch.cat([x, F.interpolate(enc3, size=x.size()[2:], mode='bilinear', align_corners=True)], dim=1))
        x = self.upconv3(torch.cat([x, F.interpolate(enc2, size=x.size()[2:], mode='bilinear', align_corners=True)], dim=1))
        x = self.upconv4(torch.cat([x, F.interpolate(enc1, size=x.size()[2:], mode='bilinear', align_corners=True)], dim=1))
        x = self.final_conv(x)

        # Apply YOLO-style activations
        x[:, 0:2, :, :] = torch.sigmoid(x[:, 0:2, :, :])  # (x, y) -> range [0,1]
        x[:, 2:4, :, :] = torch.exp(x[:, 2:4, :, :])      # (w, h) -> positive values
        x[:, 4:, :, :] = torch.sigmoid(x[:, 4:, :, :])    # (conf + class scores) -> range [0,1]

        return x

class UNetYOLO(nn.Module):
    def __init__(self, in_channels, num_classes, dropout_prob=0.3):
        super(UNetYOLO, self).__init__()
        self.encoder = UNetEncoder(in_channels, dropout_prob)
        self.decoder = YOLODecoder(num_classes)

    def forward(self, x):
        bottleneck, enc1, enc2, enc3, enc4 = self.encoder(x)
        output = self.decoder(bottleneck, enc1, enc2, enc3, enc4)
        return output  # YOLO-style detection output