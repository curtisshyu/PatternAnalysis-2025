import os, sys
import torch
from torch.utils.data import DataLoader
import torch.optim as optim

from recognition.unet_curtisshyu.modules import UNet
from recognition.unet_curtisshyu.dataset import get_datasets
from recognition.unet_curtisshyu.utils import hard_dice, plot_training

import torch
import torch.backends.cudnn as cudnn
import gc, os, sys


torch.cuda.empty_cache()
gc.collect()


torch.backends.cudnn.enabled = False
cudnn.benchmark = False
cudnn.deterministic = True



def train_model(epochs, lr, batch_size, bce_weight=0.5):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on device: {device}")

    train_set, val_set, test_set = get_datasets()
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    model = UNet(n_channels=1, n_classes=4).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer, total_iters=epochs, power=0.9)

    loss_fn = torch.nn.CrossEntropyLoss()

    best_val_dice = 0.0
    train_losses, val_dices = [], []

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0

        for imgs, masks in train_loader:
            imgs = imgs.to(device)
            masks = masks.squeeze(1).long().to(device)  # [B, H, W]

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = loss_fn(outputs, masks)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)

        # VALIDATION
        model.eval()
        dices = []
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs = imgs.to(device)
                masks = masks.squeeze(1).long().to(device)

                outputs = model(imgs)
                preds = torch.argmax(outputs, dim=1, keepdim=True)   # [B,1,H,W]

                # compute Dice for prostate only (class 3)
                d = hard_dice((preds == 3).float(), (masks.unsqueeze(1) == 3).float())
                dices.append(d.item())

        avg_val_dice = sum(dices) / len(dices)
        scheduler.step()

        print(f"Epoch [{epoch}/{epochs}] - loss: {avg_train_loss:.4f} - val prostate dice: {avg_val_dice:.4f}")

        train_losses.append(avg_train_loss)
        val_dices.append(avg_val_dice)

        # Ssave best model
        if avg_val_dice > best_val_dice:
            best_val_dice = avg_val_dice
            os.makedirs("checkpoints", exist_ok=True)
            torch.save(model.state_dict(), "checkpoints/unet_best.pth")
            print(f"  → saved new best model (dice={best_val_dice:.4f})")

    plot_training(train_losses, val_dices, save_path="checkpoints/training_curve.png")
    return model


def test_model(batch_size=4, ckpt_path="checkpoints/unet_best.pth", save_csv=True):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, _, test_set = get_datasets()
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    model = UNet(n_channels=1, n_classes=4).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    n_classes = 4
    dice_per_class = torch.zeros(n_classes, device=device)

    with torch.no_grad():
        for imgs, masks in test_loader:
            imgs = imgs.to(device)
            masks = masks.squeeze(1).long().to(device)
            outputs = model(imgs)
            preds = torch.argmax(outputs, dim=1, keepdim=True)

            for c in range(n_classes):
                d = hard_dice((preds == c).float(), (masks.unsqueeze(1) == c).float())
                dice_per_class[c] += d

    dice_per_class /= len(test_loader)
    class_names = ["Background", "Bladder", "Rectum", "Prostate"]

    print("\n====================================")
    print("Test Set Results")
    print("====================================")
    for c, name in enumerate(class_names):
        print(f"{name} (Class {c}): Dice = {dice_per_class[c]:.4f}")
    print("====================================")

    if save_csv:
        import pandas as pd
        df = pd.DataFrame({"Class": class_names, "Dice": dice_per_class.cpu().numpy()})
        os.makedirs("recognition/unet_curtisshyu/checkpoints", exist_ok=True)
        df.to_csv("recognition/unet_curtisshyu/checkpoints/test_results.csv", index=False)
        print("Saved per-class Dice to checkpoints/test_results.csv")

    return dice_per_class.cpu().numpy()


if __name__ == "__main__":
    train_model(epochs=100, lr=1e-3, batch_size=4)
    test_model()
