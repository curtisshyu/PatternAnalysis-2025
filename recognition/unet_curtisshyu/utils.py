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

