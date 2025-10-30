"""
Defines the 2D U-Net model architecture for prostate MRI segmentation.

Implements a fully convolutional encoder–decoder structure with skip connections.
Each block includes convolution, batch normalization, and ReLU activation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Basic building block (Conv → BN → ReLU)
"""
Two convolutional laters each followed by batch and relu 
Eacch block learns increasingly complex features (edges, tissues, intensity gradients)
"""
class DoubleConv(nn.Module):
    """[Conv2d → BatchNorm → ReLU] × 2"""
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

# Down-sampling (Encoder)
"""
Performs max pooling followed by double convolution
Each time we downsample, we the model caputres broader and context
Spatial preciison decreases the eaxact pixel level location is blurred
"""
class Down(nn.Module):
    """Downscaling with MaxPool → DoubleConv"""
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.down = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.down(x)

"""
Decoder path upsamples feature maps to restore spatial resolution
Skip connections concatenate encoder features to retain fine details
"""
class Up(nn.Module):
    """Upscaling → concatenation → DoubleConv"""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # Pad x1 to match x2 size
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        # concatenate along channel dimension
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


# Output layer (1×1 convolution)
"""
A 1x1 convolution to map to desired number of output classes
Each pixel value is a raw logit indicating it belongs to the target class
"""
class OutConv(nn.Module):
    """Final 1×1 convolution to reduce channels"""
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

# Full U-Net architecture
"""
Combines into a U-Shaped encoder-Decoder network with skip connections
"""
class UNet(nn.Module):
    """Full 2D U-Net for binary segmentation"""
    def __init__(self, n_channels=1, n_classes=1, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # Encoder
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)

        # Decoder
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return torch.sigmoid(logits)