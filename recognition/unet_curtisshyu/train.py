import os, sys
import torch
from torch.utils.data import DataLoader
import torch.optim as optim

from recognition.unet_curtisshyu.modules import UNet
from recognition.unet_curtisshyu.dataset import get_datasets
from recognition.unet_curtisshyu.utils import bce_dice_loss, soft_dice, hard_dice, plot_training
from recognition.unet_curtisshyu.utils import tversky_loss
from recognition.unet_curtisshyu.utils import calculate_weight_map
from recognition.unet_curtisshyu.utils import weighted_bce_dice_loss

def train_model(epochs, lr, batch_size, bce_weight):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on device: {device}")

    train_set, val_set, test_set = get_datasets()
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    model = UNet(n_channels=1, n_classes=1).to(device)
    # Load previous best weights
    #ckpt_path = "checkpoints/unet_best.pth"
    #if os.path.exists(ckpt_path):
       # model.load_state_dict(torch.load(ckpt_path, map_location=device))
        #print("Loaded pretrained weights for fine-tuning.")
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
