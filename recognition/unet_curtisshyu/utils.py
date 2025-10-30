import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os
import pandas as pd

def soft_dice(pred, target, eps=1e-6):
    # pred: logits
    pred = torch.sigmoid(pred)
    target = torch.clamp(target, 0, 1)
    inter = (pred * target).sum(dim=(1,2,3))
    denom = pred.sum(dim=(1,2,3)) + target.sum(dim=(1,2,3))
    dice = (2*inter + eps) / (denom + eps)
    return dice.mean()

def hard_dice(pred, target, thresh=0.5, eps=1e-6):
    pred = torch.sigmoid(pred)
    pred = (pred > thresh).float()
    target = torch.clamp(target, 0, 1)
    inter = (pred * target).sum(dim=(1,2,3))
    denom = pred.sum(dim=(1,2,3)) + target.sum(dim=(1,2,3))
    dice = (2*inter + eps) / (denom + eps)
    return dice.mean()

def bce_dice_loss(pred, target, bce_weight=0.5):
    bce = nn.BCEWithLogitsLoss()(pred, target)
    d = soft_dice(pred, target)
    return bce_weight * bce + (1 - bce_weight) * (1 - d)

def plot_training(train_losses, val_dices, save_path="training_curve.png"):
    epochs = range(1, len(train_losses)+1)
    plt.figure(figsize=(6,4))
    plt.plot(epochs, train_losses, label="train loss")
    plt.plot(epochs, val_dices, label="val dice")
    plt.legend()
    plt.xlabel("epoch")
    plt.grid(True)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()

    pd.DataFrame({
        "epoch": epochs,
        "train_loss": train_losses,
        "val_dice": val_dices
    }).to_csv(os.path.splitext(save_path)[0] + ".csv", index=False)

def tversky_loss(pred, target, alpha=0.7, beta=0.3, eps=1e-6):
    pred = torch.sigmoid(pred)
    tp = (pred * target).sum(dim=(1,2,3))
    fp = ((1 - target) * pred).sum(dim=(1,2,3))
    fn = (target * (1 - pred)).sum(dim=(1,2,3))
    tversky = (tp + eps) / (tp + alpha * fp + beta * fn + eps)
    return 1 - tversky.mean()
