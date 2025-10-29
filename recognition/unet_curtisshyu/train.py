from recognition.unet_curtisshyu.modules import UNet
from recognition.unet_curtisshyu.utils import param_check
from recognition.unet_curtisshyu.dataset import get_datasets
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from recognition.unet_curtisshyu.utils import param_check, dice_coefficient, dice_loss

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
def train_model(epochs, lr, batch_size, save_path="recognition/unet_curtisshyu/checkpoints/unet_best.pth"):
    """
     Trains U-Net model on the dataset for a number of epochs
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on device: {device}")

    # Load datasets
    train_set, val_set, _ = get_datasets()
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size)

    # Model, loss, optimizer
    model = UNet(n_channels=1, n_classes=1).to(device)
    bce_loss = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Initial summary
    _, params = param_check(model)
    print(f"Model initialized with {params:,} trainable parameters")

    best_val_dice = 0.0
    # Training loop
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for imgs, masks in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = 0.5 * bce_loss(outputs, masks) + 0.5 * dice_loss(outputs, masks)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # Validation
        model.eval()
        val_dice = 0.0
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                outputs = model(imgs)
                val_dice += dice_coefficient(outputs, masks).item()

        avg_val_dice = val_dice / len(val_loader)
        print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {avg_train_loss:.4f} | Val Dice: {avg_val_dice:.4f}")


        # Save checkpoint if validation improves
        if avg_val_dice > best_val_dice:
            best_val_dice = avg_val_dice
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(model.state_dict(), save_path)
            print(f"New best model saved with Dice: {best_val_dice:.4f}")

    print("Training loop successfully completed.")
    return model


if __name__ == "__main__":
    train_model(epochs=10, lr=1e-4, batch_size=4)