"""
Contains utility functions for U-Net architecture
- Dice Coefficient
- Loss Functions
- Testing Param Count
"""

from recognition.unet_curtisshyu.modules import UNet
import torch

"""
Sanity Check
"""
def param_check(model, input_shape=(1, 1, 128, 128)):
    """
    Runs a forward pass with a dummy tensor and prints model parameter count.
    Automatically moves dummy input to the same device as the model.
    """
    # Detect the model's device
    device = next(model.parameters()).device
    model.eval()

    # Create dummy input on the same device
    x = torch.randn(*input_shape, device=device)

    with torch.no_grad():
        y = model(x)

    params_count = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Trainable parameters: {params_count:,}")

    return y, params_count

def dice_coefficient(pred, target, epsilon=1e-6):
    """
    Computes the Dice Coefficient between predicted and ground truth masks.
    Handles logits safely and averages correctly over batch.
    """
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()

    intersection = (pred * target).sum(dim=(1, 2, 3))
    union = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice = (2. * intersection + epsilon) / (union + epsilon)
    return dice.mean()

def dice_loss(pred, target):
    """
    Dice Loss = 1 - Dice Coefficient.
    Lower is better. Encourages spatial overlap.
    """
    return 1 - dice_coefficient(pred, target)