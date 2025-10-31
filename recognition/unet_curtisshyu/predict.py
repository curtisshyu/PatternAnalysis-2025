import torch
import matplotlib.pyplot as plt
import numpy as np
from recognition.unet_curtisshyu.modules import UNet
from recognition.unet_curtisshyu.dataset import get_datasets
from recognition.unet_curtisshyu.utils import hard_dice
from recognition.unet_curtisshyu.train import test_model

# consistent colour map per class: background transparent, others coloured
CLASS_COLORS = {
    1: (0, 0, 1),   # blue  = bladder
    2: (0, 1, 0),   # green = rectum
    3: (1, 0, 0),   # red   = prostate
}
CLASS_NAMES = ["Background", "Bladder", "Rectum", "Prostate"]


def visualize_prediction(model_path="recognition/unet_curtisshyu/checkpoints/unet_best.pth", idx=5):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, _, test_set = get_datasets()

    # Load model
    model = UNet(n_channels=1, n_classes=4).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Get one image-mask pair
    img, mask = test_set[idx]
    img = img.unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(img)
        pred = torch.argmax(logits, dim=1).squeeze().cpu().numpy()   # [H,W]
        gt = mask.squeeze().cpu().numpy()

    # Dice for prostate (class 3)
    dice_val = hard_dice(
        (torch.tensor(pred == 3).float().unsqueeze(0).unsqueeze(0)),
        (torch.tensor(gt == 3).float().unsqueeze(0).unsqueeze(0))
    )
    print(f"Example Prostate Dice Score: {dice_val:.4f}")

    # Convert prediction to RGB overlay
    overlay = np.zeros((*pred.shape, 3))
    for c, color in CLASS_COLORS.items():
        overlay[pred == c] = color

    # Plot
    fig, axs = plt.subplots(1, 3, figsize=(14, 5))
    axs[0].imshow(img.cpu().squeeze(), cmap="gray")
    axs[0].set_title("Input MRI")

    axs[1].imshow(gt, cmap="tab10", vmin=0, vmax=3)
    axs[1].set_title("Ground Truth Segmentation")

    axs[2].imshow(img.cpu().squeeze(), cmap="gray")
    axs[2].imshow(overlay, alpha=0.5)
    axs[2].set_title("Predicted Overlay")

    for ax in axs:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig("recognition/unet_curtisshyu/checkpoints/prediction_example.png", bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    print("====================================")
    print("Running test set evaluation...")
    print("====================================")
    test_model(batch_size=4)
    print("\nGenerating visualisation for one example slice...")
    visualize_prediction(idx=5)
    print("\nAll results and visuals saved to checkpoints/")
