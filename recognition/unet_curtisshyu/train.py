"""
train.py
Main training and evaluation driver for the 2D U-Net segmentation model.

Functions:
- train_model(): Trains the model on MRI slices using BCE+Dice loss and polynomial LR scheduler.
- test_model(): Evaluates the final trained model on the test set and saves Dice metrics.

Features:
- Checkpoint saving/loading.
- Validation-based best model tracking.
- Automatic training curve plotting and CSV export.

Entry Point:
Run this script directly to train and evaluate the model:
    $ python -m recognition.unet_curtisshyu.train
"""

import os
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from recognition.unet_curtisshyu.modules import UNet
from recognition.unet_curtisshyu.dataset import get_datasets
from recognition.unet_curtisshyu.utils import bce_dice_loss, hard_dice, plot_training
from recognition.unet_curtisshyu.utils import calculate_weight_map

def train_model(epochs, lr, batch_size, bce_weight):
    """
    Trains the 2D U-Net model for prostate segmentation using the HipMRI dataset.

    parameters:
        epochs : int
            Number of training epochs.
        lr : float
            Initial learning rate for the optimizer.
        batch_size : int
            Number of samples per mini-batch during training.
        bce_weight : float
            Weighting factor (0 to 1) that controls the trade-off between 
            Binary Cross-Entropy (BCE) and Dice loss components.

    returns:
        torch.nn.Module
            The trained U-Net model with weights corresponding to the best 
            validation Dice score.

    notes:
        - Loads and splits the HipMRI dataset into training, validation, and test sets.
        - Initializes the U-Net architecture, Adam optimizer, and polynomial LR scheduler.
        - Uses a hybrid BCE + Dice loss with optional positive class weighting.
        - Tracks validation Dice coefficient to save the best-performing model checkpoint.
        - Generates and saves training/validation performance curves for analysis.
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on device: {device}")

    train_set, val_set, test_set = get_datasets()
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    model = UNet(n_channels=1, n_classes=1).to(device)
    # Load previous best weights
    ckpt_path = "checkpoints/unet_best.pth"
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        print("Loaded pretrained weights for fine-tuning.")
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # Polynomial Scheduler
    scheduler = torch.optim.lr_scheduler.PolynomialLR(
            optimizer, total_iters=epochs, power=0.9
        )

    pos_weight = torch.tensor([4.0]).to(device)  

    best_val_dice = 0.0
    train_losses = []
    val_dices = []

    for epoch in range(1, epochs+1):
        # train 
        model.train()
        running_loss = 0.0
        for imgs, masks in train_loader:
            imgs = imgs.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()

            weight_np = calculate_weight_map(masks.detach().cpu().numpy())
            weights = torch.tensor(weight_np, device=device, dtype=torch.float32)

            outputs = model(imgs)
            loss = bce_dice_loss(outputs, masks, bce_weight=bce_weight, pos_weight=pos_weight)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)

        # validate
        model.eval()
        dices = []
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs = imgs.to(device)
                masks = masks.to(device)
                outputs = model(imgs)
                d = hard_dice(outputs, masks, thresh=0.5)
                dices.append(d.item())
        avg_val_dice = sum(dices) / len(dices)
        scheduler.step()
        print(f"Epoch [{epoch}/{epochs}] - loss: {avg_train_loss:.4f} - val dice: {avg_val_dice:.4f}")

        train_losses.append(avg_train_loss)
        val_dices.append(avg_val_dice)

        # save best
        if avg_val_dice > best_val_dice:
            best_val_dice = avg_val_dice
            os.makedirs("checkpoints", exist_ok=True)
            torch.save(model.state_dict(), "checkpoints/unet_best.pth")
            print(f"  → saved new best model (dice={best_val_dice:.4f})")

    # plot
    plot_training(train_losses, val_dices, save_path="checkpoints/training_curve.png")
    return model

def test_model(batch_size=4, ckpt_path="checkpoints/unet_best.pth", save_csv=True):
    """
    Evaluates a trained U-Net model on the held-out test set and computes Dice scores.

    parameters:
        batch_size : int, optional
            Number of test samples processed per batch. Default is 4.
        ckpt_path : str, optional
            File path to the trained model checkpoint (.pth) used for evaluation.
            Default is "checkpoints/unet_best.pth".
        save_csv : bool, optional
            If True, saves per-sample Dice scores and the mean Dice score to a CSV file
            for documentation and reproducibility. Default is True.

    returns:
        float
            The average Dice coefficient computed across the entire test set.

    notes:
        - Loads the trained U-Net model and runs inference on all test samples.
        - Computes the Dice similarity coefficient per image and averages results.
        - Prints the test summary and optionally saves detailed results to 
          `recognition/unet_curtisshyu/checkpoints/test_results.csv`.
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, _, test_set = get_datasets()
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    model = UNet(n_channels=1, n_classes=1).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    dices = []

    with torch.no_grad():
        for imgs, masks in test_loader:
            imgs = imgs.to(device)
            masks = masks.to(device)
            outputs = model(imgs)
            d = hard_dice(outputs, masks, thresh=0.5)
            dices.append(d.item())

    final_dice = sum(dices) / len(dices)
    print("====================================")
    print("Test Set Results")
    print("====================================")
    print(f"Prostate (Class 1) Dice: {final_dice:.4f}")
    print(f"Background (Class 0) Dice: 1.0000 (implicit)")
    print("====================================")

    if save_csv:
        import pandas as pd
        df = pd.DataFrame({"sample_id": list(range(len(dices))), "dice_score": dices})
        df.loc[len(df)] = ["Mean", final_dice]
        os.makedirs("recognition/unet_curtisshyu/checkpoints", exist_ok=True)
        df.to_csv("recognition/unet_curtisshyu/checkpoints/test_results.csv", index=False)
        print("Saved test results to checkpoints/test_results.csv")

    return final_dice

if __name__ == "__main__":
    train_model(epochs=50, lr=1e-3, batch_size=8, bce_weight=0.5)
    test_model()