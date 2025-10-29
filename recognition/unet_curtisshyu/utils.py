import matplotlib.pyplot as plt
import pandas as pd
import os
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

def dice_loss(pred, target):
    """Dice loss = 1 - soft Dice."""
    return 1 - soft_dice_coefficient(pred, target)



def dice_coefficient(pred, target, epsilon=1e-6):
    """
    Hard Dice for evaluation — thresholded at 0.5.
    Also clamps results to [0, 1].
    """
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()
    target = torch.clamp(target, 0, 1)

    intersection = (pred * target).sum(dim=(1, 2, 3))
    union = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice = (2. * intersection + epsilon) / (union + epsilon)
    return torch.clamp(dice.mean(), 0.0, 1.0)

def plot_training_curves(train_losses, val_dices, save_dir="recognition/unet_curtisshyu/checkpoints"):
    """
    Plots and saves training loss and validation Dice curves.
    Also saves a CSV log of metrics for reproducibility.
    """
    os.makedirs(save_dir, exist_ok=True)
    log_path = os.path.join(save_dir, "training_log.csv")
    plot_path = os.path.join(save_dir, "training_curve.png")

    # Save logs
    pd.DataFrame({
        "epoch": list(range(1, len(train_losses) + 1)),
        "train_loss": train_losses,
        "val_dice": val_dices
    }).to_csv(log_path, index=False)

    # Plot curves
    plt.figure(figsize=(8, 5))
    plt.title("Training Loss and Validation Dice per Epoch")
    plt.plot(range(1, len(train_losses) + 1), train_losses, label="Train Loss", color="blue")
    plt.plot(range(1, len(val_dices) + 1), val_dices, label="Val Dice", color="orange")
    plt.xlabel("Epoch")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()

    print(f"Training curves saved:\n- CSV: {log_path}\n- PNG: {plot_path}")