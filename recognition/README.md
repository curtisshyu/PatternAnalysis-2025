# 2D U-Net Segmentation on HIPMRI Prostate Dataset

## Project Overview
This project implements a **2D U-Net convolutional neural network** for **automatic prostate segmentation** on the *HipMRI Study for Prostate Cancer Radiotherapy*.  
The goal is to segment the prostate region from MRI slices to support clinical workflows such as **radiotherapy planning** and **organ delineation**.

The model is trained using processed 2D slices of the HipMRI dataset and evaluated on held-out test images, targeting a **minimum Dice similarity coefficient of 0.75** on the prostate label.

## Algorithm Description

### Model Architechture
The algorithm is based on the **U-Net** architecture (Ronneberger et al., 2015), a fully convolutional encoder–decoder network designed for biomedical image segmentation.  
It captures both **low-level spatial features** and **high-level semantic features** through symmetric skip connections.

- **Encoder:** Sequential convolution–batchnorm–ReLU blocks with max pooling (downsampling)
- **Bottleneck:** Deepest layer capturing context
- **Decoder:** Upsampling via bilinear interpolation + concatenation with encoder features
- **Output Layer:** 1×1 convolution → single-channel sigmoid mask prediction

## Visualisation
- TO COMPLETE

## How it works

1. **Pre-processing:**
   - MRI slices (`.nii`/`.nii.gz`) are normalized between the 1st and 99th percentile of intensity values.  
   - Values are standardized to zero mean and unit variance.  
   - Masks are normalized to [0,1] and resized to **256×256**.
   - Spatial Augmentation (random flipping and transformations) are applied to increase robustness to orientaiton and variability in MRI scans

2. **Training:**  
   - Uses the **Tversky loss** (α=0.7, β=0.3) to handle strong class imbalance between prostate and background pixels.  
   - Optimizer: **Adam** with learning rate `1e-3`  
   - Batch size: 4, 50 epochs  
   - Validation Dice used to save the best-performing model.
   - Training Curves plotted to examine behviour

3. **Evaluation:**  
   - Performance measured using the **Dice Similarity Coefficient (DSC)** on the held-out test set.  
   - Qualitative evaluation includes overlaying predicted segmentation masks on MRI slices.


## Dataset

The dataset used is the **HipMRI Study Open Dataset**:  
`/home/groups/comp3710/HipMRI_Study_open/keras_slices_data`

Data sets are splot as follows
| **Train** | `keras_slices_train` / `keras_slices_seg_train` | Model fitting | Contains majority of samples for learning |
| **Validation** | `keras_slices_validate` / `keras_slices_seg_validate` | Hyperparameter & checkpoint selection | Prevents overfitting, unseen during training |
| **Test** | `keras_slices_test` / `keras_slices_seg_test` | Final model evaluation | Provides unbiased estimate of performance |

--- 

## Example Usage
- TO COMPLETE

## Reproducibility
- No Random seeds to consider fixed across the codebase
- Dataset splits are static
- All results saved under checkpoints/unet_best.pth
- training_curve.png and training_curve.csv 

## Dependencies
Python 3.10.2
MacOS 15.6.1
torch 2.9.0
pandas
matplotlib
numpy
nibabel
albumentations

Install via:

```bash
pip install pandas
