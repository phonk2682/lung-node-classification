# Lung Nodule Malignancy Classification

Binary classification of lung nodules (Benign / Malignant) from CT scans.
Built on the LUNA25 challenge framework — Group 12.

---

## Table of Contents

1. [Project Structure](#1-project-structure)
2. [Environment Setup](#2-environment-setup)
3. [Training](#3-training)
4. [Inference](#4-inference)

---

## 1. Project Structure

```
lung_nodule_pipeline/
├── lung_nodule/                    # Main Python package
│   ├── config.py                   # Hyperparameters & settings
│   ├── classification/             # 2D / 3D malignancy classifier
│   ├── data/                       # Dataset, patch extraction, augmentation
│   ├── detection/                  # MONAI RetinaNet nodule detector
│   ├── models/                     # Model architectures (ResNet152, UNet3D, ViT, ...)
│   ├── pipeline/                   # End-to-end orchestration + DICOM→NIfTI
│   ├── reporting/                  # Batch report generation
│   └── training/                   # Trainer, loss functions, k-fold splits
│
├── docs/
│   ├── TRAINING.md                 # Data format, training guide, parameters
│   └── INFERENCE.md                # Inference modes, checkpoint setup, MTN guide
│
├── data/                           # Training data (gitignored — download separately)
│   ├── image/                      # Nodule patches: <AnnotationID>.npy
│   ├── metadata/                   # Spatial metadata: <AnnotationID>.npy
│   └── csv/                        # 5-fold split CSVs
│
├── weights/                        # Pre-trained checkpoints (gitignored — download separately)
│   ├── dt_model.ts                 # RetinaNet detection model (TorchScript)
│   ├── ResNet152-confirmed/        # 2D classification ensemble (fold_1..5)
│   └── unet3D_encoder_scse/        # 3D classification ensemble (fold0..4)
│
├── train.py                        # Train 5-fold cross-validation
├── infer.py                        # Classify known nodule coordinates (single / batch CSV)
├── predict.py                      # End-to-end DICOM → detect → classify
├── run_report.py                   # Batch report across a dataset directory
├── infer_mtn.sh                    # One-shot MTN dataset inference → CSV
│
├── setup.py
├── requirements.txt
└── README.md
```

---

## 2. Environment Setup

```bash
# Create and activate environment
conda create -n lung_nodule python=3.11 -y
conda activate lung_nodule

# Install PyTorch (adjust cu121 to match your CUDA version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install all remaining dependencies
pip install -r requirements.txt
```

Verify:

```bash
python -c "import torch, monai, SimpleITK, timm; print('OK')"
```

---

## 3. Training

See **[docs/TRAINING.md](docs/TRAINING.md)** for the full guide, including:

- Data directory layout and CSV format
- Generating 5-fold splits
- All `train.py` arguments and available model architectures
- Expected output structure and metrics

Quick start:

```bash
python train.py \
    --image_dir  ./data \
    --csv_dir    ./data/csv \
    --output_dir ./checkpoints \
    --model      ResNet152 \
    --epochs     200
```

---

## 4. Inference

See **[docs/INFERENCE.md](docs/INFERENCE.md)** for the full guide, including:

- Checkpoint setup (pre-trained vs. custom)
- **Mode A** — single nodule from known coordinates
- **Mode B** — batch CSV inference
- **Mode C** — MTN dataset: extract ZIPs → detect → classify → CSV (`infer_mtn.sh`)
- Model types: 2D ResNet152 / 3D UNet3D+scSE / both
- Coordinate system reference and troubleshooting

Quick start (MTN dataset):

```bash
bash infer_mtn.sh /path/to/MTN/ --output_dir ./output/mtn
```

Single nodule:

```bash
python infer.py \
    --ct      patient.nii.gz \
    --coord_x -34.3 \
    --coord_y  44.2 \
    --coord_z -49.3
```
