from recognition.unet_curtisshyu.modules import UNet
from recognition.unet_curtisshyu.utils import param_check
from recognition.unet_curtisshyu.dataset import get_datasets
from recognition.unet_curtisshyu.utils import plot_training_curves
from recognition.unet_curtisshyu.utils import weighted_bce_dice_loss, calculate_weight_map
from recognition.unet_curtisshyu.utils import soft_dice_coefficient
from recognition.unet_curtisshyu.utils import combined_loss
from recognition.unet_curtisshyu.utils import simple_dice_bce_loss
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from recognition.unet_curtisshyu.utils import param_check, dice_coefficient, dice_loss
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd



"""
Initial Sanity Check
"""
#model = UNet(n_channels=1, n_classes=1)
#_, params = param_check(model)

"""
Loads the Dataset, instantiate the U-Net model, and runs a single forward pass
"""
def sanity_check(device="cuda" if torch.cuda.is_available() else "cpu"):
    print(f"Running on device: {device}")

    # Load datasets
    train_set, val_set, test_set = get_datasets()
    print(f"Train samples: {len(train_set)}, Val: {len(val_set)}, Test: {len(test_set)}")

    # Wrap one set in DataLoader
    train_loader = DataLoader(train_set, batch_size=1, shuffle=True)

    # Instantiate model
    model = UNet(n_channels=1, n_classes=1).to(device)

    # Forward pass on one batch
    img, mask = next(iter(train_loader))
    img, mask = img.to(device), mask.to(device)
    print(f"Input batch shape: {img.shape}")

    with torch.no_grad():
        out = model(img)

    print(f"Output batch shape: {out.shape}")

    # Parameter count
    _, params = param_check(model)
    print(f"Sanity check passed — model has {params:,} parameters")

    return out


"""
Training Skeleton
"""
def train_model(epochs=100, lr=1e-4, batch_size=8):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    train_set, val_set, _ = get_datasets()
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size)

    model = UNet(n_channels=1, n_classes=1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    best_val_dice = 0
    train_losses, val_dices = [], []

    for epoch in range(epochs):
        # Training
        model.train()
        epoch_loss = 0
        for imgs, masks in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = simple_dice_bce_loss(outputs, masks, dice_weight=0.5)
            
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_train_loss = epoch_loss / len(train_loader)

        # Validation
        model.eval()
        val_dice_epoch = []
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                outputs = model(imgs)
                dice_val = soft_dice_coefficient(outputs, masks)
                val_dice_epoch.append(dice_val.item())
        
        avg_val_dice = np.mean(val_dice_epoch)

        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {avg_train_loss:.4f} - Dice: {avg_val_dice:.4f}")

        train_losses.append(avg_train_loss)
        val_dices.append(avg_val_dice)

        if avg_val_dice > best_val_dice:
            best_val_dice = avg_val_dice
            torch.save(model.state_dict(), "best_model.pth")
            print(f"Saved best model with Dice: {best_val_dice:.4f}")

    return model


def test_model(checkpoint_path,
               batch_size):
    """
    Loads the best saved U-Net model and evaluates it on the unseen test set.
    Computes the overall Dice coefficient to measure generalisation.
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Evaluating on device: {device}")

    # Load the test set only
    _, _, test_set = get_datasets()
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size)

    # Load best model weights
    model = UNet(n_channels=1, n_classes=1).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    dice_scores = []
    with torch.no_grad():
        for imgs, masks in test_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            outputs = model(imgs)
            dice = dice_coefficient(outputs, masks)
            dice_scores.append(dice.item())

    avg_dice = sum(dice_scores) / len(dice_scores)
    print(f"\nTest Dice Coefficient on unseen test set: {avg_dice:.4f}")
    return avg_dice

def hyperparam_tuning():
    configs = [
        {"lr": 1e-3, "bce_ratio": 0.3},
        {"lr": 7e-4, "bce_ratio": 0.4},
        {"lr": 5e-4, "bce_ratio": 0.5},
        {"lr": 3e-4, "bce_ratio": 0.6},
        {"lr": 2e-4, "bce_ratio": 0.4}
    ]

    for cfg in configs:
        print("="*70)
        print(f"Running config → LR: {cfg['lr']} | BCE ratio: {cfg['bce_ratio']}")
        print("="*70)

        model = train_model(
            epochs=50,
            lr=cfg["lr"],
            batch_size=2,
            bce_ratio=cfg["bce_ratio"]
        )
    print("\nHyperparameter tuning completed. Compare validation curves or Dice scores to select best combo.")


if __name__ == "__main__":
    train_model(epochs=100, lr=1e-4, batch_size=8)
    test_model("recognition/unet_curtisshyu/checkpoints/unet_best.pth", batch_size=2)