"""
Defines the 2D U-Net model architecture for prostate MRI segmentation.

Implements a fully convolutional encoder–decoder structure with skip connections.
Each block includes convolution, batch normalization, and ReLU activation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Basic building block (Conv → BN → ReLU)
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