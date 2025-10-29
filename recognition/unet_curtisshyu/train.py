from recognition.unet_curtisshyu.modules import UNet
from recognition.unet_curtisshyu.utils import param_check
from recognition.unet_curtisshyu.dataset import get_datasets
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
import torch
from torch.utils.data import DataLoader

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


if __name__ == "__main__":
    sanity_check()