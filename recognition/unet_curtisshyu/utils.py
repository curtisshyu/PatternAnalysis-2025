"""
utils.py
Contains supporting functions for training and evaluation of the U-Net segmentation model.

Includes:
- Loss functions: BCE+Dice, Weighted BCE+Dice, Tversky loss.
- Evaluation metrics: soft/hard Dice similarity coefficients.
- Visualization: Training curve plotting.
- Spatial weighting: Distance transform-based weight maps for edge emphasis.

Used by: train.py
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
import cv2

def soft_dice(pred, target, eps=1e-6):
    """
    Computes the differentiable (soft) Dice coefficient between predicted and target masks.

    parameters:
        pred : torch.Tensor
            Raw logits output from the model of shape (B, 1, H, W).
            These are passed through a sigmoid to obtain probabilities in [0, 1].
        target : torch.Tensor
            Ground-truth binary segmentation masks of the same shape as `pred`.
        eps : float, optional
            Small constant (default=1e-6) added to numerator and denominator 
            to prevent division by zero in edge cases.

    returns:
        torch.Tensor
            A scalar tensor representing the **mean Dice coefficient** across the batch.
    """
    # pred: logits
    pred = torch.sigmoid(pred)
    target = torch.clamp(target, 0, 1)
    inter = (pred * target).sum(dim=(1,2,3))
    denom = pred.sum(dim=(1,2,3)) + target.sum(dim=(1,2,3))
    dice = (2*inter + eps) / (denom + eps)
    return dice.mean()

def hard_dice(pred, target, thresh=0.5, eps=1e-6):
    """
    Computes the thresholded (hard) Dice coefficient between predicted and target masks.

    parameters:
        pred : torch.Tensor
            Raw logits output from the model of shape (B, 1, H, W).
            These are passed through a sigmoid and thresholded to obtain binary predictions.
        target : torch.Tensor
            Ground-truth binary segmentation masks of the same shape as `pred`.
        thresh : float, optional
            Threshold value (default=0.5) used to convert probabilities into binary predictions.
        eps : float, optional
            Small constant (default=1e-6) added to numerator and denominator 
            to prevent division by zero in edge cases.

    returns:
        torch.Tensor
            A scalar tensor representing the **mean hard Dice coefficient** across the batch.
    """
    pred = torch.sigmoid(pred)
    pred = (pred > thresh).float()
    target = torch.clamp(target, 0, 1)
    inter = (pred * target).sum(dim=(1,2,3))
    denom = pred.sum(dim=(1,2,3)) + target.sum(dim=(1,2,3))
    dice = (2*inter + eps) / (denom + eps)
    return dice.mean()

def bce_dice_loss(pred, target, bce_weight=0.5, pos_weight=None):
    """
    Computes a hybrid loss combining Binary Cross-Entropy (BCE) and Dice loss.

    parameters:
        pred : torch.Tensor
            Raw logits output from the model of shape (B, 1, H, W).
            These are passed through a sigmoid within BCE and Dice computations.
        target : torch.Tensor
            Ground-truth binary segmentation masks of the same shape as `pred`.
        bce_weight : float, optional
            Weighting factor (0 to 1) that balances BCE and Dice components.
            A higher value emphasizes pixel-wise accuracy (BCE), while a lower
            value emphasizes overlap accuracy (Dice). Default is 0.5.
        pos_weight : torch.Tensor or None, optional
            Optional weighting tensor applied to the positive (foreground) class 
            in BCE to address class imbalance. Default is None.

    returns:
        torch.Tensor
            A scalar tensor representing the combined BCE + Dice loss.
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
    """
    Plots training loss and validation Dice score curves across epochs, and saves both 
    the visual plot and corresponding metrics as a CSV file.

    parameters:
        train_losses : list[float]
            List of training loss values recorded at each epoch.
        val_dices : list[float]
            List of validation Dice scores recorded at each epoch.
        save_path : str, optional
            File path (including filename) to save the plot image (.png) 
            and accompanying CSV file of metrics. Default is "training_curve.png".

    returns:
        None
    """

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

def calculate_weight_map(masks, w0=10, sigma=5):
    """
    Computes a spatial weight map to emphasize boundary regions in segmentation masks.  
    This is based on distance transforms from object and background boundaries, 
    producing higher weights near edges to improve model focus on transitions.

    parameters:
        masks : np.ndarray
            Numpy array of shape (B, 1, H, W) containing binary segmentation masks.
            Each pixel is either 0 (background) or 1 (foreground).
        w0 : float, optional
            Scaling factor that controls the maximum weight near object boundaries.
            Default is 10.
        sigma : float, optional
            Standard deviation parameter controlling the Gaussian decay of weights 
            away from the boundary. Default is 5.

    returns:
        np.ndarray
            Weight maps of shape (B, 1, H, W), where higher values correspond to 
            boundary-adjacent pixels.
    """
    weights = np.zeros_like(masks, dtype=np.float32)
    for i in range(masks.shape[0]):
        mask = masks[i, 0]
        dist_fore = cv2.distanceTransform((mask > 0).astype(np.uint8), cv2.DIST_L2, 3)
        dist_back = cv2.distanceTransform((mask == 0).astype(np.uint8), cv2.DIST_L2, 3)
        d = dist_fore + dist_back

        # Clamp very large distance values to avoid overflow
        d = np.clip(d, 0, 255)

        weights[i, 0] = w0 * np.exp(- (d ** 2) / (2 * sigma ** 2))
    return weights


def weighted_bce_dice_loss(pred, target, weight_map, bce_ratio=0.5, eps=1e-6):
    """
    Computes a spatially weighted hybrid loss combining Binary Cross-Entropy (BCE) 
    and Dice loss. The weighting map emphasizes boundary regions, reducing the 
    dominance of background pixels and improving segmentation of fine structures.

    parameters:
        pred : torch.Tensor
            Raw logits output from the model of shape (B, 1, H, W).
        target : torch.Tensor
            Ground-truth binary segmentation masks of the same shape as `pred`.
        weight_map : torch.Tensor
            Per-pixel weighting map of shape (B, 1, H, W) used to boost 
            the importance of boundary and minority regions.
        bce_ratio : float, optional
            Factor (0 to 1) controlling the trade-off between BCE and Dice components. 
            Default is 0.5.
        eps : float, optional
            Small constant (default=1e-6) added for numerical stability to 
            prevent division by zero.

    returns:
        torch.Tensor
            A scalar tensor representing the combined spatially weighted 
            BCE + Dice loss.
    """

    pred_sig = torch.sigmoid(pred)
    bce = -(weight_map * (target * torch.log(pred_sig + eps) +
                          (1 - target) * torch.log(1 - pred_sig + eps))).mean()
    inter = (weight_map * pred_sig * target).sum(dim=(1,2,3))
    denom = (weight_map * (pred_sig + target)).sum(dim=(1,2,3))
    dice = (2 * inter + eps) / (denom + eps)
    return bce_ratio * bce + (1 - bce_ratio) * (1 - dice.mean())