import os
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm

class HipMRIDataset(Dataset):
    """
    Loads 2D MRI slice images and corresponding segmentation masks 
    for prostate segmentation using the HipMRI dataset.
    """
    def __init__(self, image_dir, mask_dir, transform=None, normalize=True, target_size=(128, 128)):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.normalize = normalize
        self.target_size = target_size

        # Get matching filenames
        self.image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(('.nii', '.nii.gz'))])
        self.mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith(('.nii', '.nii.gz'))])

        assert len(self.image_files) == len(self.mask_files), \
            f"Number of images ({len(self.image_files)}) and masks ({len(self.mask_files)}) must match."

        self.resize = transforms.Resize(self.target_size, antialias=True)

    def __getitem__(self, idx):
            image_path = os.path.join(self.image_dir, self.image_files[idx])
            mask_path = os.path.join(self.mask_dir, self.mask_files[idx])

            image = nib.load(image_path).get_fdata(caching='unchanged').astype(np.float32)
            mask = nib.load(mask_path).get_fdata(caching='unchanged').astype(np.float32)

            if self.normalize:
                image = (image - np.mean(image)) / np.std(image)

            # Add channel dimension
            image = np.expand_dims(image, axis=0)
            mask = np.expand_dims(mask, axis=0)

            image = torch.tensor(image, dtype=torch.float32)
            mask = torch.tensor(mask, dtype=torch.float32)

            # Resize both image and mask to fixed size
            image = self.resize(image)
            mask = self.resize(mask)

            if self.transform:
                image = self.transform(image)
                mask = self.transform(mask)

            return image, mask

# Helper function to create train/val/test datasets
def get_datasets(base_path="recognition/unet_curtisshyu/data/keras_slices_data"):
    """
    Prepares train/val/test datasets using the actual folder names
    in kera_slices_data.
    """

    train_imgs = os.path.join(base_path, "keras_slices_train")
    train_masks = os.path.join(base_path, "keras_slices_seg_train")

    val_imgs = os.path.join(base_path, "keras_slices_validate")
    val_masks = os.path.join(base_path, "keras_slices_seg_validate")

    test_imgs = os.path.join(base_path, "keras_slices_test")
    test_masks = os.path.join(base_path, "keras_slices_seg_test")

    train_set = HipMRIDataset(train_imgs, train_masks)
    val_set = HipMRIDataset(val_imgs, val_masks)
    test_set = HipMRIDataset(test_imgs, test_masks)

    return train_set, val_set, test_set

if __name__ == "__main__":
    # Quick test to verify loading works
    train_set, _, _ = get_datasets()
    print(f"Train set size: {len(train_set)} samples")

    img, mask = train_set[0]
    print(f"Image shape: {img.shape}, Mask shape: {mask.shape}")



