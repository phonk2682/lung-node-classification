# Lung Nodule Malignancy Classification

Binary classification of lung nodules (Benign / Malignant) from CT scans.
Built on the LUNA25 challenge framework — Group 12.

---

## Table of Contents

1. [Project Structure](#1-project-structure)
2. [Environment Setup](#2-environment-setup)
3. [Training](#3-training)
4. [Inference](#4-inference)
5. [End-to-End Pipeline](#5-end-to-end-pipeline-dicom--detect--classify)

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

> **Note:** The `data/` directory (raw DICOM files, training CSVs, and `.npy` patches)
> is **not** included. See §3 for the expected layout.

---

## 2. Environment Setup

### Requirements

- Python 3.10 or 3.11
- CUDA GPU (strongly recommended — CPU is very slow for inference)

### Step-by-step installation

```bash
# 1. Create and activate environment
conda create -n lung_nodule python=3.11 -y
conda activate lung_nodule

# 2. Install PyTorch  (adjust cu121 to match your CUDA version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 3. Install all remaining dependencies
#    numpy<2 is required — nibabel otherwise upgrades it and breaks scipy
pip install -r requirements.txt
```

### Verify installation

```bash
python -c "import torch, monai, SimpleITK, timm; print('OK')"
```

Expected output: `OK`

---

## 3. Training

### 3.1 Data preparation

Training requires pre-extracted nodule patches. The `data/` directory must have this layout:

```
data/
├── image/
│   ├── <AnnotationID_1>.npy    # 3-D numpy array, shape (64, 128, 128) — raw HU values
│   ├── <AnnotationID_2>.npy
│   └── ...
├── metadata/
│   ├── <AnnotationID_1>.npy    # dict with keys: origin, spacing, transform
│   ├── <AnnotationID_2>.npy
│   └── ...
└── csv/
    ├── data_fold_1_train.csv
    ├── data_fold_1_test.csv
    ├── data_fold_2_train.csv
    ├── data_fold_2_test.csv
    ├── ...
    └── data_fold_5_test.csv
```

**Required CSV columns:**

| Column | Type | Description |
|---|---|---|
| `AnnotationID` | str | Unique nodule ID (matches filename in `image/` and `metadata/`) |
| `label` | int | Ground truth: `0` = Benign, `1` = Malignant |
| `PatientID` | str | Used to prevent patient leakage across folds |

### 3.2 Running training

```bash
python train.py \
    --image_dir  ./data \
    --csv_dir    ./data/csv \
    --output_dir ./checkpoints
```

**All arguments:**

| Argument | Required | Default | Description |
|---|---|---|---|
| `--image_dir` | ✅ | — | Root data dir containing `image/` and `metadata/` subdirs |
| `--csv_dir` | ✅ | — | Dir with `data_fold_<N>_train.csv` and `data_fold_<N>_test.csv` |
| `--output_dir` | | `./checkpoints` | Where to save model checkpoints |
| `--model` | | `ResNet152` | Architecture: `ResNet18/50/101/152`, `EfficientNetB3/B4/B5`, `ConvNeXtBase`, `DenseNet121`, `ViTBase`, ... |
| `--epochs` | | 200 | Override max epochs |
| `--batch_size` | | 64 | Override batch size |
| `--folds` | | 5 | Number of folds to train (1–5) |

### 3.3 Expected output

```
============================================================
TRAINING CONFIGURATION
============================================================
  Model      : ResNet152
  Mode       : 2D
  Image dir  : ./data
  CSV dir    : ./data/csv
  Output dir : ./checkpoints/ResNet152-2D-20260322-CV
  Epochs     : 200  (patience=50)
  Batch size : 64
  Folds      : 5
============================================================

Fold 1: 9842 train / 2461 val samples
==================================================Fold 1 | Train: 9842 | Val: 2461==================================================
Fold 1 | Epoch 1/200 | Train Loss: 0.4231 | Val Loss: 0.3987 | AUC: 0.6821 | AP: 0.4102 | Acc: 0.6543 | F1: 0.3201
✓ New best model saved (AP: 0.3201)
...
Fold 1 complete - Best F1: 0.7812 at epoch 87

============================================================
CROSS-VALIDATION RESULTS
============================================================
  Fold 1: AUC=0.8923  AP=0.8341  Acc=0.8102  F1=0.7812
  Fold 2: AUC=0.8756  AP=0.8190  Acc=0.7934  F1=0.7643
  Fold 3: AUC=0.9011  AP=0.8512  Acc=0.8234  F1=0.7987
  Fold 4: AUC=0.8834  AP=0.8267  Acc=0.8043  F1=0.7701
  Fold 5: AUC=0.8945  AP=0.8398  Acc=0.8178  F1=0.7834
------------------------------------------------------------
  Mean AUC        : 0.8894 ± 0.0089
  Mean AP         : 0.8342 ± 0.0112
  Mean ACCURACY   : 0.8098 ± 0.0109
  Mean F1         : 0.7795 ± 0.0123

Checkpoints saved to: ./checkpoints/ResNet152-2D-20260322-CV/fold_<N>/best_metric_model.pth
```

**Checkpoint directory structure after training:**

```
checkpoints/ResNet152-2D-20260322-CV/
├── fold_1/
│   ├── best_metric_model.pth   # Best model weights (saved when F1 improves)
│   └── config.npy              # Metrics at best checkpoint
├── fold_2/
│   └── ...
...
└── fold_5/
    └── ...
```

---

## 4. Inference

`infer.py` supports two input modes (**single nodule** or **batch CSV**) and three model types.

### 4.0 Model types

| `--model_type` | Architecture | Checkpoints |
|---|---|---|
| `2d` *(default)* | ResNet152, 2D axial patch (64×64 px, 50×50 mm) | `weights/ResNet152-confirmed/fold_*/best_metric_model.pth` |
| `3d` | UNet3D encoder + scSE attention, 3D patch (64³ px, 50³ mm) | `weights/unet3D_encoder_scse/best_metric_model_fold*.pth` |
| `both` | Average of 2D and 3D ensemble probabilities | both directories above |

Both models use 5-fold ensembles: logits are averaged across folds before sigmoid activation.

### 4.1 Single nodule

```bash
python infer.py \
    --ct      patient.nii.gz \
    --coord_x -34.3 \
    --coord_y  44.2 \
    --coord_z -49.3
```

Coordinates must be in **ITK/LPS world space** (millimetres). These are the coordinates
typically found in LUNA25 challenge CSVs and DICOM metadata.

**Expected output:**

```
┌──────────────────────────────────────────────────────────────┐
│  Nodule Malignancy Classification                            │
│  CT      : patient.nii.gz                                    │
│  Coord   : x=-34.30  y=44.20  z=-49.30  (world, LPS mm)     │
│  ──────────────────────────────────────────────────────────  │
│  Probability : 0.1165                                        │
│  Label       : 0 — Benign                                    │
└──────────────────────────────────────────────────────────────┘
```

### 4.2 Batch mode

Prepare an input CSV:

```csv
ct_path,coord_x,coord_y,coord_z
/data/patient_001.nii.gz,-34.3,44.2,-49.3
/data/patient_001.nii.gz,-42.8,-16.4,-47.6
/data/patient_002.nii.gz,94.6,-61.9,-190.3
```

Run:

```bash
python infer.py --csv nodules.csv --output results.csv
```

**Expected terminal output:**

```
[INFO] Batch mode: 3 nodules from nodules.csv
  [1/3] /data/patient_001.nii.gz  (-34.3, 44.2, -49.3) ... Benign  (p=0.1165)
  [2/3] /data/patient_001.nii.gz  (-42.8, -16.4, -47.6) ... Benign  (p=0.1055)
  [3/3] /data/patient_002.nii.gz  (94.6, -61.9, -190.3) ... Benign  (p=0.0469)

Results: 3 Benign  |  0 Malignant
Saved to: results.csv
```

**Output CSV columns:**

| Column | Description |
|---|---|
| `ct_path` | Input CT path (copied from input) |
| `coord_x/y/z` | Input coordinates (copied from input) |
| `probability` | Malignancy probability in [0.0, 1.0] |
| `label` | `0` = Benign, `1` = Malignant |
| `label_str` | `"Benign"` or `"Malignant"` |

### 4.3 Optional arguments

| Argument | Default | Description |
|---|---|---|
| `--weights_dir` | `weights/ResNet152-confirmed/` | Path to 2D fold checkpoint directory |
| `--weights_dir_3d` | `weights/unet3D_encoder_scse/` | Path to 3D fold checkpoint directory |
| `--model_type` | `2d` | `2d` (ResNet152), `3d` (UNet3D+scSE), or `both` (average of 2D and 3D) |
| `--threshold` | `0.5` | Probability threshold for Malignant |

**Use 3D model only:**

```bash
python infer.py \
    --ct patient.nii.gz \
    --coord_x -34.3 --coord_y 44.2 --coord_z -49.3 \
    --model_type 3d
```

**Use both models (ensemble average):**

```bash
python infer.py \
    --ct patient.nii.gz \
    --coord_x -34.3 --coord_y 44.2 --coord_z -49.3 \
    --model_type both
```

**Use custom checkpoints** (e.g., after training your own model):

```bash
python infer.py \
    --ct patient.nii.gz \
    --coord_x -34.3 --coord_y 44.2 --coord_z -49.3 \
    --weights_dir ./checkpoints/ResNet152-2D-20260322-CV
```

---

## 5. End-to-End Pipeline (DICOM → Detect → Classify)

If you have raw DICOM files and no known nodule coordinates, use `predict.py` which
chains DICOM conversion, automatic nodule detection, and classification:

```bash
python predict.py \
    --dicom_dir /path/to/patient_dicom/ \
    --output_dir ./output/
```

The same `--model_type` flag from inference is available here:

```bash
# Use 3D model for classification
python predict.py --dicom_dir /path/to/patient_dicom/ --model_type 3d

# Use both models (averaged)
python predict.py --dicom_dir /path/to/patient_dicom/ --model_type both
```

**All arguments:**

| Argument | Default | Description |
|---|---|---|
| `--dicom_dir` | *(required)* | Patient DICOM folder |
| `--output_dir` | temp dir | Where to save NIfTI and JSON results |
| `--model_type` | `2d` | Classification model: `2d`, `3d`, or `both` |
| `--weights_dir` | `weights/ResNet152-confirmed/` | 2D checkpoint directory |
| `--weights_dir_3d` | `weights/unet3D_encoder_scse/` | 3D checkpoint directory |
| `--model_path` | `weights/dt_model.ts` | Detection TorchScript weights |
| `--score_keep` | `0.3` | Minimum detection confidence score |
| `--threshold` | `0.5` | Malignancy probability threshold |

**Expected output:**

```
[Step 1] Converting DICOM → NIfTI ...
[Step 2] Detecting nodules (score_keep=0.3) ...
[INFO] Found 3 nodule(s).
[Step 3] Classifying nodules (model_type=2d) ...
  Nodule 1/3: RAS=(-34.3, 44.2, -49.3) mm  score=0.925
    → Benign (probability=0.1165)
  Nodule 2/3: RAS=(-42.8, -16.4, -47.6) mm  score=0.914
    → Benign (probability=0.1055)
  Nodule 3/3: RAS=(94.6, -61.9, -190.3) mm  score=0.890
    → Benign (probability=0.0469)

──────────────────────────────────────────────────
SUMMARY — 3 nodule(s) detected
──────────────────────────────────────────────────
  Nodule 1: Benign      p=0.1165  detection_score=0.925
  Nodule 2: Benign      p=0.1055  detection_score=0.914
  Nodule 3: Benign      p=0.0469  detection_score=0.890

[Done] Results saved to: ./output/patient_results.json
```

---

## Troubleshooting

| Error | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'monai'` | `pip install monai itk nibabel` |
| `No module named 'nibabel'` | `pip install nibabel` |
| `numpy.core.multiarray failed to import` | `pip install "numpy<2"` — nibabel may upgrade numpy to 2.x |
| `No DICOM series found` | Check the directory contains `.dcm` files from one series only |
| `FileNotFoundError: weights/dt_model.ts` | Place `dt_model.ts` in the `weights/` directory |
| `No fold checkpoints found` | Check `--weights_dir` points to a folder with `fold_*/best_metric_model.pth` |
| `No 3D fold checkpoints found` | Check `--weights_dir_3d` points to a folder with `best_metric_model_fold*.pth` |
| CUDA out of memory | The detector needs ~8 GB VRAM; set `CUDA_VISIBLE_DEVICES=""` to use CPU |
