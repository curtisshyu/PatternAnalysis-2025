# Working README/notes
This module defines the data loading and preprocessing pipeline for the HipMRI prostate cancer segmentation task.
It loads 2D MRI slices and their corresponding segmentation masks from NIfTI (.nii / .nii.gz) files, applies normalization, and prepares the data for training a 2D U-Net segmentation model.

The dataset represents a pixel-wise classification problem — each MRI slice is paired with a mask identifying which pixels correspond to the prostate region. The loader ensures that every image and mask pair is correctly aligned and converted into PyTorch tensors suitable for GPU training.

Accurate and efficient data loading is critical for deep learning pipelines.
This module automates:

- Reading the HipMRI dataset structure (train, validation, test splits)
- Performing data normalization for stable neural network training
- Formatting the MRI slices and segmentation masks into PyTorch tensors with channel-first convention ([C, H, W])
- Optionally applying image transformations for augmentation or preprocessing
- This enables seamless integration with PyTorch’s DataLoader, allowing mini-batch training and shuffling across thousands of MRI slices.

Normalization:
MRI intensities vary significantly across scans; zero-mean, unit-variance normalization standardizes pixel values, allowing the network to focus on structural patterns rather than brightness variations.

Channel-first tensors:
PyTorch models expect inputs in [C, H, W] format; hence, a channel dimension is explicitly added even for grayscale (single-channel) MRI data.

NIfTI format (.nii, .nii.gz):
This medical imaging format retains full voxel intensity data, preserving anatomical accuracy over lossy formats like .png.

Sorted filenames:
Sorting ensures that image and mask pairs remain synchronized during iteration.

# Overview module.py
This module implements the 2D U-Net convolutional neural network used for prostate segmentation in the HipMRI Study dataset.
The U-Net is a fully convolutional encoder–decoder architecture with skip connections, originally proposed for biomedical image segmentation.
It allows the model to capture both global context and fine spatial details, which is essential when identifying structures like the prostate in MRI slices.

DoubleConv → Down → Up → OutConv → UNet

Input: A single channel 2D MRI slice
Target: 0 = background, 1 = prostate (of interest)

Input MRI -> U-net -> output is comapred to ground truth mask - loss function measures how close to real mask - gradients back propogate through network and weights get upadted

# Utils.py
- Testing the param count, we can use this for any model we pass later to for a sanity check
- Ensure input and output tensors align
- loss functions
- coefficient functions
- plotting functions

# Train.py
## Sanity Check
- Loads the datsetets using dataset.py
- instantiates modules
- performs a single pass to match dimensions
## train model
- loads data into pytorch
- initialises model
- defines binary corss entropy corss logits
- optimises weights
- performs forward/backward propogation
## Validtion/Testing
When we “implement validation” before training the full model, we’re not training on it — we’re simply setting up a diagnostic mechanism that tells us:
If you only track training loss:
You might see it go to near 0,
But your model could fail miserably on new data → overfitting.
The validation set acts as a checkpoint: it’s not used for gradient updates, but after each epoch we:
Freeze the model.
Run it on the validation data.
Compute metrics (e.g. Dice score).
Compare to previous epochs.
## Scheduler
- Montiors validation, if it does not improve for 3 pochs, reduces lr by a fctor of 0.5
Allows us to escape plateaus
- mode = max monitors validation
- factor is how much to reduce by
- patience is count wihtout improvement to reduce

dataset.py
- justify the transforms augmentation
- mimic realsitic mri variablity
- improve generalisation, and reduce overfitting
- exploit symmetry in left-right anatomical symmetry
- simulate minor slice orientaiton that occur during acuqistion
- mimic small patient alignment or scanner rotation