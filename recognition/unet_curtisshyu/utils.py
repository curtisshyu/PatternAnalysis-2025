"""
Contains utility functions for U-Net architecture
- Dice Coefficient
- Loss Functions
- Testing Param Count
"""

from recognition.unet_curtisshyu.modules import UNet
import torch

def param_check(model, input_shape = (1, 1, 128, 128)):
    # Dummy inputs
    device = next(model.parameters()).device
    x = torch.randn(*input_shape)

    model.eval()

    with torch.no_grad():
        y = model(x)
    params_count = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Trainable parameters: {params_count}")

    return y, params_count # Expect to get x shape = y shape

