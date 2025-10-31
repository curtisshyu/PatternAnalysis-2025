import torch
import matplotlib.pyplot as plt
from recognition.unet_curtisshyu.modules import UNet
from recognition.unet_curtisshyu.dataset import get_datasets
from recognition.unet_curtisshyu.utils import hard_dice
from recognition.unet_curtisshyu.train import test_model

def visualize_prediction(model_path="recognition/unet_curtisshyu/checkpoints/unet_best.pth", idx=5):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, _, test_set = get_datasets()
    
    # Load model
    model = UNet(n_channels=1, n_classes=1).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Get one image-mask pair
    img, mask = test_set[idx]
    img = img.unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model(img)
        pred_sigmoid = torch.sigmoid(pred).squeeze().cpu().numpy()
        pred_bin = (pred_sigmoid > 0.5).astype(float)

    # Plot
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].imshow(img.cpu().squeeze(), cmap='gray')
    axs[0].set_title("Input MRI")
    axs[1].imshow(mask.cpu().squeeze(), cmap='gray')
    axs[1].set_title("Ground Truth")
    axs[2].imshow(img.cpu().squeeze(), cmap='gray')
    axs[2].imshow(pred_bin, cmap='jet', alpha=0.4)
    axs[2].set_title("Predicted Mask (Overlay)")

    for ax in axs:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig("recognition/unet_curtisshyu/checkpoints/prediction_example.png", bbox_inches='tight')
    plt.show()

    dice_val = hard_dice(pred, mask.unsqueeze(0).to(device))
    print(f"Example Dice Score: {dice_val:.4f}")

if __name__ == "__main__":
    print("====================================")
    print("Running test set evaluation...")
    print("====================================")
    test_model(batch_size=4)
    print("\nGenerating visualisation for one example slice...")
    visualize_prediction(idx=5)
    print("\nAll results and visuals saved to checkpoints/")