# 504-SAM2-MedSeg-LoRA
SAM2 3D Medical Image Segmentation Baseline
========================================================================

A robust baseline evaluation pipeline for 3D medical image segmentation using SAM2 (Segment Anything Model 2). This implementation employs a 2D bounding-box bidirectional propagation strategy to assess SAM2's zero-shot performance on 3D MRI volumes (e.g., BraTS) without any fine-tuning.

PROJECT OVERVIEW
-------------------------------------------------------------------------------

Traditional 3D medical image segmentation often requires specialized 3D architectures or extensive domain-specific training. This project provides a simple and reproducible baseline that treats a 3D volume as a sequence of 2D slices and uses SAM2 in a zero-shot setting.

Unlike standard video object segmentation pipelines that rely on memory banks and temporal modeling, this baseline:

- treats each volume as an ordered stack of 2D slices
- propagates bounding-box prompts bidirectionally from an initialization slice
- provides a clean lower-bound baseline for SAM2 on medical imaging tasks

KEY FEATURES
-------------------------------------------------------------------------------

- Zero-shot evaluation with pretrained SAM2 weights
- Bidirectional propagation from the slice with the largest foreground region
- Pure baseline design with no video APIs, no multi-axis fusion, and no test-time augmentation
- Native NIfTI support for `.nii.gz` inputs and outputs
- Automatic 3D Dice score computation

PROJECT STRUCTURE
-------------------------------------------------------------------------------

```text
.
├── model/
│   └── sam2-main/                    # SAM2 source code cloned from the official repository
├── README.md
├── requirements.txt
├── sam2_brain_tumour_baseline.py     # Main evaluation script
└── Task01_BrainTumour/
    ├── dataset.json
    ├── imagesTr/                     # 4D MRI volumes (H, W, D, C)
    ├── imagesTs/
    └── labelsTr/                     # 3D ground-truth masks (H, W, D)
```

INSTALLATION
========================================================================

PREREQUISITES
-------------------------------------------------------------------------------

- Python 3.10 or higher
- CUDA 12.1 or higher recommended for GPU inference

CLONE THE REPOSITORY
-------------------------------------------------------------------------------

```bash
# Clone this project
git clone <your-repo-url>
cd <your-project-directory>

# Clone the official SAM2 repository
mkdir -p model
git clone https://github.com/facebookresearch/segment-anything-2.git model/sam2-main
```

INSTALL DEPENDENCIES
-------------------------------------------------------------------------------

```bash
pip install -r requirements.txt
cd model/sam2-main && pip install -e .
```

DOWNLOAD SAM2 CHECKPOINTS
-------------------------------------------------------------------------------

Download a SAM2 checkpoint from the official repository and place it at:

```text
model/sam2-main/checkpoints/sam2.1_hiera_tiny.pt
```

You may also use other SAM2 variants if your script is configured accordingly.

DATASET PREPARATION
========================================================================

The script expects a dataset layout compatible with Task01_BrainTumour:

```text
Task01_BrainTumour/
├── imagesTr/       # 4D MRI volumes (H, W, D, C=4)
├── imagesTs/       # Optional test images
└── labelsTr/       # 3D ground-truth masks (H, W, D)
```

BraTS modality indices:

- `0`: FLAIR
- `1`: T1
- `2`: T1Gd
- `3`: T2

RUNNING THE BASELINE
========================================================================

BASIC EXAMPLE
-------------------------------------------------------------------------------

```bash
python sam2_brain_tumour_baseline.py \
    --dataset-root ./Task01_BrainTumour \
    --num-samples 10 \
    --modality-idx 0 \
    --save-prediction-nii
```

IMPORTANT ARGUMENTS
-------------------------------------------------------------------------------

```text
--dataset-root         Path to the dataset directory
--num-samples          Number of cases to evaluate
--modality-idx         MRI modality index (0=FLAIR, 1=T1, 2=T1Gd, 3=T2)
--save-prediction-nii  Save predicted masks as NIfTI files
--output-dir           Directory for outputs
--device               Inference device, e.g. cuda or cpu
--seed                 Random seed for reproducibility
```

ADDITIONAL EXAMPLES
-------------------------------------------------------------------------------

```bash
# Evaluate 5 cases using T1
python sam2_brain_tumour_baseline.py \
    --dataset-root ./Task01_BrainTumour \
    --num-samples 5 \
    --modality-idx 1

# Evaluate 10 cases using T2 and save predictions
python sam2_brain_tumour_baseline.py \
    --dataset-root ./Task01_BrainTumour \
    --num-samples 10 \
    --modality-idx 3 \
    --save-prediction-nii

# Current implementation
python train_sam2_brats_qlora.py \
   --dataset-root ./Task01_BrainTumour \
   --output-dir ./outputs/sam2_brats_qlora \
   --checkpoint ./model/sam2-main/checkpoints/sam2.1_hiera_large.pt \
   --device cuda
```

PIPELINE LOGIC
========================================================================

The evaluation process consists of the following stages:

1. Initialization  
   Identify the slice `z_init` with the largest tumor area in the ground-truth mask.

2. Forward propagation  
   For each slice `z > z_init`, extract a bounding box from the predicted mask on slice `z - 1` and use it as the prompt for slice `z`.

3. Backward propagation  
   For each slice `z < z_init`, extract a bounding box from the predicted mask on slice `z + 1` and use it as the prompt for slice `z`.

4. Volume assembly  
   Stack all 2D predictions into a final 3D prediction volume.

5. Metric computation  
   Compute the global 3D Dice coefficient between the predicted volume and the ground truth.

OUTPUTS
========================================================================

Results are written to the output directory, typically `outputs/`.

FILES
-------------------------------------------------------------------------------

```text
outputs/
├── metrics_summary.json
├── per_case_metrics.csv
└── predictions_nii/     # created if --save-prediction-nii is enabled
```

METRICS SUMMARY
-------------------------------------------------------------------------------

```json
{
  "mean_dice": 0.452,
  "std_dice": 0.081,
  "num_samples": 10,
  "modality": "FLAIR (idx=0)",
  "timestamp": "2024-01-15T10:30:00"
}
```

PER-CASE METRICS
-------------------------------------------------------------------------------

```text
case_id    dice_score  init_slice  volume_shape      processing_time
---------  ----------  ----------  ----------------  ---------------
BRATS_001  0.512       78          (240, 240, 155)   45.2s
BRATS_002  0.487       82          (240, 240, 155)   43.8s
```

PREDICTION FILES
-------------------------------------------------------------------------------

If `--save-prediction-nii` is enabled, predicted binary masks are saved as `.nii.gz` files for inspection in tools such as ITK-SNAP or 3D Slicer.

NOTES
========================================================================

- This repository is intended as a baseline, not a fully optimized medical segmentation system.
- The method is designed to measure zero-shot behavior of SAM2 under a simple and controlled prompt propagation setup.
- It does not include fine-tuning, ensembling, or post-processing intended to maximize benchmark performance.

ACKNOWLEDGEMENTS
========================================================================

- Segment Anything 2 (SAM2) by Meta Research
- The BraTS dataset and the medical imaging research community

DISCLAIMER
========================================================================

This code is provided for research purposes only. It is not intended for clinical use.
