import os
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

class HipMRIDataset(Dataset):
    """
    Loads 2D MRI slice images and corresponding segmentation masks 
    for prostate segmentation using the HipMRI dataset.
    """
    def __init__(self, image_dir, mask_dir, normalize=True, target_size=(256, 256), augment=False):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.normalize = normalize
        self.target_size = target_size
        self.augment = augment

        # Albumentations transform (synchronised image-mask aug)
        if augment:
            self.transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.2),
                A.RandomRotate90(p=0.3),
                A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=10, p=0.5, border_mode=0),
                A.Resize(height=self.target_size[0], width=self.target_size[1]),
                A.Normalize(mean=0.0, std=1.0),
                ToTensorV2()
            ])
        else:
            self.transform = A.Compose([
                A.Resize(height=self.target_size[0], width=self.target_size[1]),
                A.Normalize(mean=0.0, std=1.0),
                ToTensorV2()
            ])

        # Match filenames
        self.image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(('.nii', '.nii.gz'))])
        self.mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith(('.nii', '.nii.gz'))])

        assert len(self.image_files) == len(self.mask_files), \
            f"Number of images ({len(self.image_files)}) and masks ({len(self.mask_files)}) must match."

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_files[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_files[idx])
        
        # Load both NIfTI volumes
        img_nii = nib.load(image_path)
        mask_nii = nib.load(mask_path)

        # Check spatial alignment
        if not np.allclose(img_nii.affine, mask_nii.affine):
            print(f"[WARNING] Affine mismatch: {self.image_files[idx]}")
            print("Image affine:\n", img_nii.affine)
            print("Mask affine:\n", mask_nii.affine)


        image = nib.load(image_path).get_fdata(caching='unchanged').astype(np.float32)
        mask = nib.load(mask_path).get_fdata(caching='unchanged').astype(np.float32)
        # Keep only prostate class
        mask = mask.astype(np.int64)

        # Clip + normalize image
        if self.normalize:
            image = np.clip(image, np.percentile(image, 1), np.percentile(image, 99))
            image = (image - np.mean(image)) / (np.std(image) + 1e-5)

        # Albumentation
# Convert mask to integer class labels
        mask = mask.astype(np.int64)

        augmented = self.transform(image=image, mask=mask)
        image = augmented["image"]
        mask = augmented["mask"].long()   # ensure integer labels

        return image, mask


def get_datasets(base_path="recognition/unet_curtisshyu/data/keras_slices_data"):
    """
    Prepares train/val/test datasets using the actual folder names
    in keras_slices_data.
    """
    train_imgs = os.path.join(base_path, "keras_slices_train")
    train_masks = os.path.join(base_path, "keras_slices_seg_train")

    val_imgs = os.path.join(base_path, "keras_slices_validate")
    val_masks = os.path.join(base_path, "keras_slices_seg_validate")

    test_imgs = os.path.join(base_path, "keras_slices_test")
    test_masks = os.path.join(base_path, "keras_slices_seg_test")

    train_set = HipMRIDataset(train_imgs, train_masks, augment=True)
    val_set = HipMRIDataset(val_imgs, val_masks, augment=False)
    test_set = HipMRIDataset(test_imgs, test_masks, augment=False)

    return train_set, val_set, test_set

import numpy as np, nibabel as nib

#m = nib.load("recognition/unet_curtisshyu/data/keras_slices_data/keras_slices_seg_train/seg_004_week_0_slice_0.nii.gz").get_fdata()
#print(np.unique(m))



if __name__ == "__main__":
    train_set, _, _ = get_datasets()
    print(f"Train set size: {len(train_set)} samples")
    img, mask = train_set[0]
    print(f"Image shape: {img.shape}, Mask shape: {mask.shape}")