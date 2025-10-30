import matplotlib.pyplot as plt
import pandas as pd
import os
import torch
import torch.nn as nn
import numpy as np
from scipy.ndimage import distance_transform_edt

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

def soft_dice_coefficient(pred, target, epsilon=1e-6):
    """
    Differentiable soft Dice. Works for logits and float masks.
    Clamps to avoid Dice > 1 due to scaling errors.
    """
    pred = torch.sigmoid(pred)
    # Ensure target is in [0, 1]
    target = torch.clamp(target, 0, 1)

    intersection = (pred * target).sum(dim=(1, 2, 3))
    union = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice = (2. * intersection + epsilon) / (union + epsilon)
    return torch.clamp(dice.mean(), 0.0, 1.0)

def dice_loss(pred, target, smooth=1.0):
    pred = torch.sigmoid(pred)
    target = torch.clamp(target, 0, 1)
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    dice = (2. * intersection + smooth) / (union + smooth)
    return 1 - dice

def dice_coefficient(pred, target, threshold=0.4, epsilon=1e-6):
    """
    Dice score on binary masks (thresholded sigmoid outputs).
    """
    pred = torch.sigmoid(pred)
    pred = (pred > threshold).float()
    target = torch.clamp(target, 0, 1)
    intersection = (pred * target).sum(dim=(1, 2, 3))
    union = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice = (2. * intersection + epsilon) / (union + epsilon)
    return dice.mean().item()

def get_class_weights(mask):
    """Higher weight on underrepresented class pixels."""
    c0 = (mask == 0)
    c1 = (mask == 1)
    total = mask.size
    count_0 = c0.sum()
    count_1 = c1.sum()
    if count_1 < 10:
        return np.ones(mask.shape)
    w0 = total / (2.0 * count_0)
    w1 = total / (2.0 * count_1)
    return w0 * c0 + w1 * c1

def weight_map(mask, w0=8, sigma=6):
    """Adds Gaussian boundary weighting for prostate edges."""
    mask = mask.astype(np.uint8)
    dist_bg = distance_transform_edt(mask == 0)
    dist_fg = distance_transform_edt(mask == 1)
    boundary = np.exp(-((dist_fg + dist_bg) ** 2) / (2 * sigma ** 2))
    wc = get_class_weights(mask)
    return wc + w0 * boundary

def calculate_weight_map(masks: np.ndarray):
    weights = []
    for m in np.squeeze(masks):
        weights.append(weight_map(m))
    return np.array(weights)


def dice_coef_loss(pred, target, smooth=1.0):
    """
    Soft Dice loss (expects sigmoid probs)
    """
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return 1.0 - dice

def weighted_bce_dice_loss(pred, target, weights, bce_ratio=0.5):
    """Weighted BCE + Dice combo (tunable ratio)."""
    bce = nn.BCEWithLogitsLoss(reduction='none')(pred, target)
    bce = (bce * weights.unsqueeze(1)).mean()
    dsc = dice_loss(pred, target)
    return bce_ratio * bce + (1 - bce_ratio) * dsc

def plot_training_curves(train_losses, val_dices, save_path="recognition/unet_curtisshyu/checkpoints/training_curve.png"):
    import matplotlib.pyplot as plt, pandas as pd, os

    # Handle unequal lengths safely
    min_len = min(len(train_losses), len(val_dices))
    train_losses, val_dices = train_losses[:min_len], val_dices[:min_len]
    epochs = range(1, min_len + 1)

    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses, label="Train Loss", marker="o")
    plt.plot(epochs, val_dices, label="Val Dice", marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Metric")
    plt.title("Training Curve")
    plt.legend()
    plt.grid(True)

    # Save plot
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()

    pd.DataFrame({
        "epoch": epochs,
        "train_loss": train_losses,
        "val_dice": val_dices
    }).to_csv(os.path.splitext(save_path)[0] + "_log.csv", index=False)

def combined_loss(pred, target, bce_weight=0.5):
    """Simpler combined loss"""
    bce_loss = nn.BCEWithLogitsLoss()(pred, target)
    dice_loss_val = dice_loss(pred, target)
    return bce_weight * bce_loss + (1 - bce_weight) * dice_loss_val

def simple_dice_bce_loss(pred, target, dice_weight=0.5):
    """Simple combined loss - much more stable"""
    bce_loss = nn.BCEWithLogitsLoss()(pred, target)
    dice_loss_val = dice_loss(pred, target)
    return (1 - dice_weight) * bce_loss + dice_weight * dice_loss_val

