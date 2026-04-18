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

---

## Repository Structure

```bash
Final_preprocess_brats_slices.py   # Preprocessing: 3D → 2D slices
Final_train_gray_1024.py           # Training with SAM2 + LoRA
Final_evaluate_gray_fast.py        # Evaluation
Final_visualize_case.py            # Visualization
train_val_split.json               # Train/test split (case-level)
Sam2_Video.mp4                    
README.md                          # Project documentation
