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

### Binary Segmentation
- Although the dataset contains four classes (0, 1, 2, 3), only the label 3 corresponds to the prostate gland.
- The other labels represent surrounding anatomy (e.g., rectum, bladder) or background that are not relevant for the task’s evaluation metric (Dice for prostate segmentation).
- Therefore, the problem was simplified to binary segmentation: Class 1 = prostate, class 0 = everything else
- Implemented via mask = (mask == 3).astype(np.float32)

## How it works

1. **Pre-processing:**
   - MRI slices (`.nii`/`.nii.gz`) are normalized between the 1st and 99th percentile of intensity values.  
   - Values are standardized to zero mean and unit variance.  
   - Masks are normalized to [0,1] and resized to **256×256**.
   - Spatial Augmentation (random flipping and transformations) are applied to increase robustness to orientaiton and variability in MRI scans

2. **Training:**  
   - The loss uses a Binary Cross Entroy and Dice loss to ensure stable pixel gradients, and optimise overlaps between predicted and ground truth
   - The Weighted BCE term incorporates per-pixel weighting derived from distance transforms, giving higher importance to prostate boundaries and reducing background dominance.
   - The Dice component enforces region-level overlap accuracy.
   - Optimizer: **Adam** with learning rate `1e-3`  
   - Batch size: 8, 50 epochs  
   - BCE_Weight = 0.5
   - Validation Dice used to save the best-performing model.
   - Training Curves plotted to examine behviour
   - Polynomial learning-rate decay (power = 0.9) for smooth, monotonic LR reduction—improving stability and preventing stagnation.
   - checkpoint to save current weights

3. **Evaluation:**  
   - Performance measured using the **Dice Similarity Coefficient (DSC)** on the held-out test set.  
   - Qualitative evaluation includes overlaying predicted segmentation masks on MRI slices.

## Visualisation
- TO COMPLETE

## Dataset

The dataset used is the **HipMRI Study Open Dataset**:  
`/home/groups/comp3710/HipMRI_Study_open/keras_slices_data`

Data sets are splot as follows
| **Train** | `keras_slices_train` / `keras_slices_seg_train` | Model fitting | Contains majority of samples for learning |
| **Validation** | `keras_slices_validate` / `keras_slices_seg_validate` | Hyperparameter & checkpoint selection | Prevents overfitting, unseen during training |
| **Test** | `keras_slices_test` / `keras_slices_seg_test` | Final model evaluation | Provides unbiased estimate of performance |

--- 

## Example Usage
- To train the model run python -m recognition.unet_curtisshyu.train
- To Generate visualisationsand make predictions python -m recognition.unet_curtisshyu.predict
- All results are saved down to recognition/unet_curtisshyu/checkpoints/

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
os
sys

Install via:

```bash
pip install torch pandas matplotlib numpy nibabel albumentations os sys

