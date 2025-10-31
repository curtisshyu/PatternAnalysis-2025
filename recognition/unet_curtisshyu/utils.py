import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np


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

def bce_dice_loss(pred, target, bce_weight=0.5, pos_weight=None):
    """
    Combined BCE + Dice loss.
    Optionally supports class weighting via pos_weight (for rare prostate pixels).
    """
    # If a pos_weight is provided, use it in BCE
    if pos_weight is not None:
        bce_loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        bce_loss_fn = nn.BCEWithLogitsLoss()

    bce = bce_loss_fn(pred, target)
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


def calculate_weight_map(masks, w0=10, sigma=5):
    """
    Compute spatial weight maps to emphasize object borders.
    masks: numpy array of shape (B, 1, H, W)
    Returns weight maps of same shape.
    """
    import cv2
    weights = np.zeros_like(masks, dtype=np.float32)
    for i in range(masks.shape[0]):
        mask = masks[i, 0]
        dist_fore = cv2.distanceTransform((mask > 0).astype(np.uint8), cv2.DIST_L2, 3)
        dist_back = cv2.distanceTransform((mask == 0).astype(np.uint8), cv2.DIST_L2, 3)
        weights[i, 0] = w0 * np.exp(-((dist_fore + dist_back) ** 2) / (2 * sigma ** 2))
    return weights

def weighted_bce_dice_loss(pred, target, weight_map, bce_ratio=0.5, eps=1e-6):
    """
    Weighted BCE + Dice loss that down-weights background and boosts boundary regions.
    weight_map : torch.Tensor of same shape as target
    """
    pred_sig = torch.sigmoid(pred)
    bce = -(weight_map * (target * torch.log(pred_sig + eps) +
                          (1 - target) * torch.log(1 - pred_sig + eps))).mean()
    inter = (weight_map * pred_sig * target).sum(dim=(1,2,3))
    denom = (weight_map * (pred_sig + target)).sum(dim=(1,2,3))
    dice = (2 * inter + eps) / (denom + eps)
    return bce_ratio * bce + (1 - bce_ratio) * (1 - dice.mean())