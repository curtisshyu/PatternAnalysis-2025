from recognition.unet_curtisshyu.modules import UNet
from recognition.unet_curtisshyu.utils import param_check
from recognition.unet_curtisshyu.dataset import get_datasets
from recognition.unet_curtisshyu.utils import plot_training_curves
from recognition.unet_curtisshyu.utils import weighted_bce_dice_loss, calculate_weight_map
from recognition.unet_curtisshyu.utils import soft_dice_coefficient
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
def train_model(epochs, lr, batch_size, bce_ratio=0.4, save_path="recognition/unet_curtisshyu/checkpoints/unet_best.pth"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on device: {device}")

    # Load data
    train_set, val_set, _ = get_datasets()
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, num_workers=2)

    model = UNet(n_channels=1, n_classes=1).to(device)
    #if os.path.exists(save_path):
        #model.load_state_dict(torch.load(save_path, map_location=device))
        #print("Loaded previous checkpoint for fine-tuning.")

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer, total_iters=epochs, power=0.9)


    _, params = param_check(model)
    print(f"Model initialized with {params:,} parameters")

    best_val_dice = 0
    train_losses, val_dices = [], []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0

        for imgs, masks in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()

            # Forward pass
            outputs = model(imgs)

            # Compute weight map
            weight_np = calculate_weight_map(masks.detach().cpu().numpy())
            weights = torch.tensor(weight_np, device=device, dtype=torch.float32)

            # Weighted BCE + Dice loss
            loss = weighted_bce_dice_loss(outputs, masks, weights, bce_ratio=bce_ratio)

            # Backpropagation
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

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
        avg_val_dice = float(np.mean(val_dice_epoch))


        print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {avg_train_loss:.4f} | Val Dice: {avg_val_dice:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")

        train_losses.append(avg_train_loss)
        val_dices.append(avg_val_dice)

        if avg_val_dice > best_val_dice:
            best_val_dice = avg_val_dice
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(model.state_dict(), save_path)
            print(f"New best model saved with Dice: {best_val_dice:.4f}")

    plot_training_curves(train_losses, val_dices, save_path="recognition/unet_curtisshyu/checkpoints/training_curve.png")
    print("Training completed.")
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
    train_model(epochs=40, lr=3e-4, batch_size=2, bce_ratio=0.3)
    test_model("recognition/unet_curtisshyu/checkpoints/unet_best.pth", batch_size=2)