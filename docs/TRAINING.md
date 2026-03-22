# Training Guide — Lung Nodule Malignancy Classification

This guide explains how to prepare data, configure training, and interpret outputs for the lung nodule binary classification pipeline (Benign = 0, Malignant = 1).

---

## Table of Contents

0. [Environment Setup](#0-environment-setup)
1. [Data Structure](#1-data-structure)
2. [Label Format & CSV Schema](#2-label-format--csv-schema)
3. [Generating Fold CSVs](#3-generating-fold-csvs)
4. [Running Training](#4-running-training)
5. [Configurable Parameters](#5-configurable-parameters)
6. [Output Structure](#6-output-structure)
7. [Training Internals](#7-training-internals)

---

## 0. Environment Setup

> See `README.md` for full installation instructions. Quick summary:

```bash
# Activate the conda environment
conda activate cs-problem

# Install dependencies (first time only)
pip install -r requirements.txt

# Verify GPU is available (optional but recommended)
python -c "import torch; print(torch.cuda.is_available())"
```

**Requirements:** Python 3.10+, CUDA 11.8+ (for GPU). CPU-only works but training is ~10× slower.

Change to the project directory before running any commands:

```bash
cd /path/to/lung_nodule_pipeline
```

### Pre-flight checklist

Before starting training, verify:

```bash
# 1. Image patches exist
ls data/image/*.npy | wc -l       # Should be ≥ 1

# 2. Metadata exists for every image
ls data/metadata/*.npy | wc -l    # Must match count above exactly

# 3. Fold CSVs exist
ls data/csv/data_fold_*_{train,test}.csv   # Must show 10 files (5 folds × train/val)
```

> **Critical:** Every `AnnotationID` in your CSVs must have **both** `image/<AnnotationID>.npy` **and** `metadata/<AnnotationID>.npy`. Missing either file will crash the dataloader mid-epoch.

---

## 1. Data Structure

Training requires two subdirectories under your `--image_dir`:

```
data/
├── image/
│   ├── 100471_3_20000102.npy       # Pre-extracted nodule patch
│   ├── 104822_3_19990102.npy
│   └── ...                         # One file per nodule
└── metadata/
    ├── 100471_3_20000102.npy       # Corresponding spatial metadata
    ├── 104822_3_19990102.npy
    └── ...
```

### Image files (`image/*.npy`)

- **Shape:** `(64, 128, 128)` — raw HU (Hounsfield Unit) values
- **Format:** NumPy array, float32
- **Content:** A voxel patch centred on the nodule, in the original CT resolution

### Metadata files (`metadata/*.npy`)

- **Format:** NumPy array wrapping a Python dict
- **Keys:**

| Key | Type | Description |
|-----|------|-------------|
| `origin` | list[float] | World-space origin of the patch (mm) |
| `spacing` | list[float] | Voxel spacing [z, y, x] (mm/voxel) |
| `transform` | 3×3 array | Direction cosine matrix |

### Filename convention

The filename stem is the `AnnotationID` from the CSV. Every row in the CSV must have a matching `.npy` file in both `image/` and `metadata/`.

**Example:**
CSV row with `AnnotationID = 100471_3_20000102`
→ requires `image/100471_3_20000102.npy`
→ requires `metadata/100471_3_20000102.npy`

---

## 2. Label Format & CSV Schema

Each fold CSV must contain at minimum these three columns:

| Column | Type | Values | Description |
|--------|------|--------|-------------|
| `AnnotationID` | str | e.g. `100471_3_20000102` | Unique nodule ID; matches filenames in `image/` and `metadata/` |
| `label` | int | `0` or `1` | Ground truth: **0 = Benign**, **1 = Malignant** |
| `PatientID` | str | e.g. `100471` | Patient identifier used to prevent data leakage across folds |

Additional columns are allowed and ignored during training:

```
PatientID, SeriesInstanceUID, StudyDate, CoordX, CoordY, CoordZ,
LesionID, AnnotationID, NoduleID, label, Age_at_StudyDate, Gender
```

### Example CSV rows

```csv
PatientID,SeriesInstanceUID,StudyDate,CoordX,CoordY,CoordZ,LesionID,AnnotationID,NoduleID,label,Age_at_StudyDate,Gender
100471,1.2.840.113654.2.55.12332...915,20000102,-80.52,80.95,1399.13,3,100471_3_20000102,100471_3,0,75,Female
104822,1.2.840.113654.2.55.19139...866,19990102,140.66,15.86,-164.35,3,104822_3_19990102,104822_3,0,62,Male
212849,1.3.6.1.4.1.14519.5.2.1.7...621,20000102,108.05,67.82,-227.01,2,212849_2_20000102,212849_2,1,72,Female
```

---

## 3. Generating Fold CSVs

If you have a single labelled CSV (`all_data.csv`), use the built-in splitter:

```bash
python - <<'EOF'
from lung_nodule.training.splits import create_kfold_splits
from pathlib import Path

fold_data = create_kfold_splits("data/all_data.csv", n_splits=5, random_state=2025)
out = Path("data/csv")
out.mkdir(parents=True, exist_ok=True)

for i, (train_df, val_df) in enumerate(fold_data, 1):
    train_df.to_csv(out / f"data_fold_{i}_train.csv", index=False)
    val_df.to_csv(out   / f"data_fold_{i}_test.csv",  index=False)
    print(f"Fold {i}: {len(train_df)} train | {len(val_df)} val")
EOF
```

**Strategy:** `StratifiedGroupKFold`
- Groups by `PatientID` → no patient appears in both train and val
- Stratified by `label` → each fold preserves the overall Benign/Malignant ratio
- Default ratio: ~80% train / 20% val per fold

**Expected output for 5546 nodules:**

```
Fold 1: 4445 train | 1101 val
Fold 2: 4447 train | 1099 val
Fold 3: 4448 train | 1098 val
Fold 4: 4445 train | 1101 val
Fold 5: 4444 train | 1102 val
```

> **Naming note:** The generated files are named `data_fold_<N>_test.csv` to match what `train.py` expects, but they are **validation folds** (used for per-epoch evaluation), not a held-out test set. There is no separate test set in this pipeline.

---

## 4. Running Training

### Minimal command

```bash
python train.py \
    --image_dir ./data \
    --csv_dir   ./data/csv \
    --output_dir ./checkpoints \
    --model ResNet152
```

### Full command with all options

```bash
python train.py \
    --image_dir  ./data \
    --csv_dir    ./data/csv \
    --output_dir ./checkpoints \
    --model      ResNet152 \
    --epochs     200 \
    --batch_size 64 \
    --folds      5
```

### Quick test run (10 epochs)

```bash
python train.py \
    --image_dir  ./data \
    --csv_dir    ./data/csv \
    --output_dir ./checkpoints/test \
    --model      ResNet152 \
    --epochs     10 \
    --folds      5
```

---

## 5. Configurable Parameters

### CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--image_dir` | *(required)* | Root data directory (must contain `image/` and `metadata/` subdirs) |
| `--csv_dir` | *(required)* | Directory with fold CSVs (`data_fold_<N>_train/test.csv`) |
| `--output_dir` | `./checkpoints` | Where to save model checkpoints |
| `--model` | `ResNet152` | Model architecture (see table below) |
| `--epochs` | `200` (config default) | Maximum training epochs per fold |
| `--batch_size` | `64` (config default) | Training batch size |
| `--folds` | `5` | Number of folds to train (1–5) |

### Available Model Architectures

All models produce a single binary output (logit).

| `--model` | Architecture | Params | Notes |
|-----------|-------------|--------|-------|
| `ResNet18` | ResNet-18 | ~11M | Fast, good baseline |
| `ResNet50` | ResNet-50 | ~26M | |
| `ResNet101` | ResNet-101 | ~45M | |
| `ResNet152` | ResNet-152 | ~60M | **Default, pre-trained** |
| `EfficientNetB3` | EfficientNet-B3 | ~12M | |
| `EfficientNetB4` | EfficientNet-B4 | ~19M | |
| `EfficientNetB5` | EfficientNet-B5 | ~30M | |
| `ConvNeXtTiny` | ConvNeXt-Tiny | ~29M | |
| `ConvNeXtSmall` | ConvNeXt-Small | ~50M | Used in ablation studies |
| `ConvNeXtBase` | ConvNeXt-Base | ~89M | |
| `ConvNeXtLarge` | ConvNeXt-Large | ~198M | |
| `DenseNet121` | DenseNet-121 | ~7M | Lightweight |
| `DenseNet169` | DenseNet-169 | ~14M | |
| `ViTBase` | ViT-Base/16 | ~87M | |
| `ViTLarge` | ViT-Large/16 | ~304M | Requires more VRAM |

### Model-specific Learning Rates

Each model has a tuned learning rate automatically applied:

| Model | Optimizer | LR |
|-------|-----------|-----|
| ResNet18, ResNet50 | Adam | `1e-4` |
| ResNet101, ResNet152 | Adam | `5e-5` |
| EfficientNetB3/B4 | AdamW | `5e-5` |
| EfficientNetB5 | AdamW | `3e-5` |
| ConvNeXtTiny | AdamW | `3e-5` |
| ConvNeXtSmall/Base | AdamW | `1e-5` |
| ConvNeXtLarge | AdamW | `5e-6` |
| DenseNet121 | Adam | `1e-4` |
| DenseNet169 | Adam | `5e-5` |
| ViTBase | AdamW | `5e-6` |
| ViTLarge | AdamW | `3e-6` |

### Advanced Parameters (edit `lung_nodule/config.py`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `PATIENCE` | `50` | Early stopping: stop if F1 doesn't improve for this many epochs |
| `WEIGHT_DECAY` | `5e-4` | L2 regularization |
| `SEED` | `2025` | Random seed for reproducibility |
| `NUM_WORKERS` | `8` | DataLoader worker processes (reduce if OOM) |
| `SIZE_MM` | `50` | Physical size of extracted patch (mm) |
| `SIZE_PX` | `64` | Pixel size of 2D patch (64×64) |
| `ROTATION` | `(±30°, ±30°, ±30°)` | Random rotation range per axis |
| `TRANSLATION_RADIUS` | `3.5 mm` | Max random translation |
| `INTENSITY_SHIFT_RANGE` | `(-0.1, 0.1)` | Intensity offset range |
| `INTENSITY_SCALE_RANGE` | `(0.9, 1.1)` | Intensity scale factor range |
| `GAUSSIAN_NOISE_STD` | `0.02` | Gaussian noise standard deviation |

> **Note on Learning Rate:** There is no `--lr` CLI argument. Learning rates are model-specific and hardcoded in `lung_nodule/models/registry.py` under `MODEL_LR_CONFIG`. To change a model's LR, edit that dict directly.

### Loss Function

**`FocalLossWithSmoothing`** (fixed, not configurable via CLI):

```python
FocalLossWithSmoothing(alpha=0.9, gamma=3, smoothing=0.1)
```

- `alpha=0.9`: Upweights malignant class (rare positives)
- `gamma=3`: Focuses on hard examples
- `smoothing=0.1`: Label smoothing to prevent overconfidence

---

## 6. Output Structure

After training, checkpoints are saved under `--output_dir`:

```
{output_dir}/
└── {model}-2D-{YYYYMMDD}-CV/            # e.g. ResNet152-2D-20260322-CV
    ├── fold_1/
    │   ├── best_metric_model.pth         # Best weights (by F1 score)
    │   └── config.npy                    # Metrics at best checkpoint
    ├── fold_2/
    │   └── ...
    ├── fold_3/
    │   └── ...
    ├── fold_4/
    │   └── ...
    ├── fold_5/
    │   └── ...
    └── cross_validation.log              # Full training log
```

### `config.npy` contents (per fold)

```python
{
    "fold":      1,
    "config":    <Config object>,
    "best_f1":   0.8542,
    "best_auc":  0.9218,
    "best_acc":  0.8913,
    "best_prec": 0.8401,
    "best_rec":  0.8689,
    "epoch":     87
}
```

### Console output (per epoch)

```
Fold 1 | Epoch 45/200 | Train Loss: 0.3214 | Val Loss: 0.4102 | AUC: 0.9218 | AP: 0.8891 | Acc: 0.8913 | F1: 0.8542
✓ New best model saved (F1: 0.8542)
current epoch: 45 current F1: 0.8542 best F1: 0.8542 at epoch 45
```

### Best model selection

The model is saved whenever **validation F1 improves**. Early stopping triggers after `PATIENCE` (default 50) epochs with no improvement.

### Cross-validation summary

```
============================================================
CROSS-VALIDATION RESULTS
============================================================
Fold 0: AUC=0.9218 | AP=0.8891 | Acc=0.8913 | F1=0.8542
Fold 1: AUC=0.9104 | AP=0.8756 | Acc=0.8834 | F1=0.8421
...
------------------------------------------------------------
Mean AUC: 0.9156 +/- 0.0062
Mean AP: 0.8823 +/- 0.0085
Mean F1: 0.8497 +/- 0.0052
```

---

## 7. Training Internals

### Class imbalance handling

Training uses a `WeightedRandomSampler` to oversample the minority class (Malignant). This ensures each epoch sees roughly equal numbers of benign and malignant examples regardless of dataset imbalance.

### Patch preprocessing pipeline

For each training sample:
1. Load raw HU patch from `image/{AnnotationID}.npy` (shape: 64×128×128)
2. Load spatial metadata from `metadata/{AnnotationID}.npy`
3. Extract 64×64 px 2D slice via affine resampling (50 mm physical window)
4. Apply augmentations: random rotation, translation, intensity shift, Gaussian noise
5. Clip HU values to `[-1000, 400]` and normalize to `[0, 1]`
6. Output tensor shape: `(1, 64, 64)` (channel-first)

### No learning rate scheduler

The scheduler (`CosineAnnealingWarmRestarts`) is intentionally disabled. This matches the experimental finding that training without a scheduler produced better results for this task (verified against notebook ablation `"ConvNeXtSmall-without-scheduler-teammate"`).
