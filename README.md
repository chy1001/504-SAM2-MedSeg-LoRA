# 504-SAM2-MedSeg-LoRA

Fine-tuning of SAM2 for 3D brain tumor segmentation using LoRA and slice-based pseudo-video propagation.

---

## Overview

This project adapts the Segment Anything Model 2 (SAM2) to medical image segmentation on the BraTS dataset.  
Instead of training directly on 3D volumes, we convert MRI scans into 2D slices and perform efficient fine-tuning using LoRA.

To improve consistency across slices, we introduce a pseudo-video propagation strategy during inference.

---

## Key Features

- 3D → 2D slice-based training pipeline  
- Parameter-efficient fine-tuning with LoRA (~0.3% parameters updated)  
- Bounding box prompting aligned with SAM2 design  
- Pseudo-video propagation across slices for consistent segmentation  
- Fast training with preprocessed `.npz` data  

---

## Pipeline

1. Convert 3D MRI volumes into 2D axial slices  
2. Normalize and resize images  
3. Generate bounding box prompts from ground truth masks  
4. Fine-tune SAM2 using LoRA  
5. Perform slice-wise inference with propagation  

## Data and Checkpoints

Due to GitHub file size limitations, the project resources are organized as follows:

### Included in this repository:
- Training and evaluation scripts
- Evaluation dataset (`eval_gray_1024/`)
- Example dataset (`BRATS_228-Example.zip`)
- Demo video (`Sam2_Video.mp4`)

### Model checkpoints and full data:
Pretrained model checkpoints and full datasets are hosted on Google Drive:

https://drive.google.com/drive/folders/15f9I8Yg0bef4ClR0yRfSKf_M2BLN5dqZ?usp=sharing

Contents include:
- Trained model checkpoints
