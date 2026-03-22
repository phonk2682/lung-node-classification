# Inference Guide — Lung Nodule Malignancy Classification

This guide covers all inference modes: single nodule, batch CSV, and end-to-end DICOM pipeline.

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Checkpoint Setup](#2-checkpoint-setup)
3. [Mode A — Single Nodule (`infer.py`)](#3-mode-a--single-nodule-inferpy)
4. [Mode B — Batch CSV (`infer.py`)](#4-mode-b--batch-csv-inferpy)
5. [Mode C — MTN Dataset (`infer_mtn.sh`)](#5-mode-c--mtn-dataset-infer_mtnsh)
6. [Model Types: 2D vs 3D vs Both](#6-model-types-2d-vs-3d-vs-both)
7. [Detection Setup (`dt_model.ts`)](#7-detection-setup-dt_modelts)
8. [Coordinate System Reference](#8-coordinate-system-reference)
9. [Troubleshooting](#9-troubleshooting)

---

## 1. Prerequisites

### Environment setup

```bash
# Activate the conda environment
conda activate cs-problem

# Install dependencies (first time only)
pip install -r requirements.txt

# Move into the project root — all commands below assume you are here
cd /path/to/lung_nodule_pipeline
```

**Requirements:** Python 3.10+, CUDA 11.8+ (recommended). CPU-only works but is ~5–10× slower.

### Required weights (minimum for 2D inference)

```
weights/
├── dt_model.ts                       # Detection model (RetinaNet, required for predict.py only)
└── ResNet152-confirmed/
    ├── fold_1/best_metric_model.pth
    ├── fold_2/best_metric_model.pth
    ├── fold_3/best_metric_model.pth
    ├── fold_4/best_metric_model.pth
    └── fold_5/best_metric_model.pth
```

Optional (for 3D inference):

```
weights/
└── unet3D_encoder_scse/
    ├── best_metric_model_fold0.pth
    ├── best_metric_model_fold1.pth
    ├── best_metric_model_fold2.pth
    ├── best_metric_model_fold3.pth
    └── best_metric_model_fold4.pth
```

---

## 2. Checkpoint Setup

### Using pre-trained weights (default)

No setup needed. The scripts default to `weights/ResNet152-confirmed/` for 2D and `weights/unet3D_encoder_scse/` for 3D.

### Using custom-trained checkpoints

After training with `train.py`, your checkpoints will be at:

```
checkpoints/ResNet152-2D-{YYYYMMDD}-CV/
├── fold_1/best_metric_model.pth
├── fold_2/best_metric_model.pth
├── fold_3/best_metric_model.pth
├── fold_4/best_metric_model.pth
└── fold_5/best_metric_model.pth
```

Pass the directory via `--weights_dir`:

```bash
python infer.py \
    --ct patient.nii.gz \
    --coord_x -34.3 --coord_y 44.2 --coord_z -49.3 \
    --weights_dir checkpoints/ResNet152-2D-20260322-CV/
```

### Finding the checkpoint directory after training

```bash
# The training script prints:  "Checkpoints at: <path>"
# Or find it manually:
ls -d checkpoints/ResNet152-2D-*/
```

---

## 3. Mode A — Single Nodule (`infer.py`)

Classify **one nodule** from a CT volume, given its world coordinates.

### When to use

- You already know the nodule location (from a radiologist report, detection tool, or another pipeline)
- You want to quickly test inference on a known nodule

### Command

```bash
python infer.py \
    --ct      /path/to/patient.nii.gz \
    --coord_x -34.3 \
    --coord_y  44.2 \
    --coord_z -49.3
```

### All arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--ct` | ✅ | — | CT image path (`.nii.gz`, `.nii`, `.mha`, `.mhd`) |
| `--coord_x` | ✅ | — | Nodule X coordinate in world space, LPS, mm |
| `--coord_y` | ✅ | — | Nodule Y coordinate in world space, LPS, mm |
| `--coord_z` | ✅ | — | Nodule Z coordinate in world space, LPS, mm |
| `--weights_dir` | | `weights/ResNet152-confirmed/` *(relative to script)* | Directory with `fold_*/best_metric_model.pth` |
| `--weights_dir_3d` | | `weights/unet3D_encoder_scse/` *(relative to script)* | Directory with `best_metric_model_fold*.pth` |
| `--model_type` | | `3d` | `2d`, `3d`, or `both` (see §6) |
| `--threshold` | | `0.5` | Probability threshold to call Malignant |

### Example output

```
┌────────────────────────────────────────────────────────────┐
│  Nodule Malignancy Classification                          │
│  CT      : /path/to/patient.nii.gz                         │
│  Coord   : x=-34.30  y=44.20  z=-49.30  (world, LPS mm)  │
│  ──────────────────────────────────────────────────────── │
│  Probability : 0.1165                                      │
│  Label       : 0 — Benign                                  │
└────────────────────────────────────────────────────────────┘
```

### With custom weights

```bash
python infer.py \
    --ct patient.nii.gz \
    --coord_x -34.3 --coord_y 44.2 --coord_z -49.3 \
    --weights_dir checkpoints/ResNet152-2D-20260322-CV/ \
    --threshold 0.4
```

---

## 4. Mode B — Batch CSV (`infer.py`)

Classify **multiple nodules** from a CSV file.

### When to use

- You have a list of nodule coordinates from multiple patients
- You want to process a whole dataset at once

### Input CSV format

The input CSV must have these columns:

| Column | Type | Description |
|--------|------|-------------|
| `ct_path` | str | Absolute or relative path to CT image |
| `coord_x` | float | X coordinate (world, LPS, mm) |
| `coord_y` | float | Y coordinate (world, LPS, mm) |
| `coord_z` | float | Z coordinate (world, LPS, mm) |

**Example input CSV (`nodules.csv`):**

```csv
ct_path,coord_x,coord_y,coord_z
/data/nifti/patient001.nii.gz,-34.3,44.2,-49.3
/data/nifti/patient001.nii.gz,12.5,-88.1,103.7
/data/nifti/patient002.nii.gz,-56.0,23.4,-210.5
```

### Command

```bash
python infer.py \
    --csv    nodules.csv \
    --output results.csv
```

### All arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--csv` | ✅ | — | Input CSV with nodule list (see format above) |
| `--output` | | `./infer_results.csv` | Output CSV path |
| `--weights_dir` | | `weights/ResNet152-confirmed/` *(relative to script)* | 2D checkpoint directory |
| `--weights_dir_3d` | | `weights/unet3D_encoder_scse/` *(relative to script)* | 3D checkpoint directory |
| `--model_type` | | `3d` | `2d`, `3d`, or `both` |
| `--threshold` | | `0.5` | Probability threshold |

### Output CSV format

The output CSV contains all input columns plus:

| Column | Type | Description |
|--------|------|-------------|
| `probability` | float | Malignancy probability [0.0 – 1.0] |
| `label` | int | `0` = Benign, `1` = Malignant |
| `label_str` | str | `"Benign"` or `"Malignant"` |

**Example output:**

```csv
ct_path,coord_x,coord_y,coord_z,probability,label,label_str
/data/nifti/patient001.nii.gz,-34.3,44.2,-49.3,0.1165,0,Benign
/data/nifti/patient001.nii.gz,12.5,-88.1,103.7,0.6823,1,Malignant
/data/nifti/patient002.nii.gz,-56.0,23.4,-210.5,0.2341,0,Benign
```

### Preparing the MTN nodule CSV

To run batch inference on the MTN dataset nodules, first create the input CSV:

```bash
cat > /tmp/mtn_nodules.csv << 'EOF'
ct_path,coord_x,coord_y,coord_z
/path/to/nifti/patient001.nii.gz,-34.3,44.2,-49.3
/path/to/nifti/patient002.nii.gz,12.5,-88.1,103.7
EOF

python infer.py \
    --csv /tmp/mtn_nodules.csv \
    --output /tmp/mtn_results.csv
```

---

## 5. Mode C — MTN Dataset (`infer_mtn.sh`)

Run the full pipeline on the MTN dataset: extract ZIPs → detect nodules in each series → classify → save everything to a single CSV.

### Pipeline steps

```
MTN/
├── *.zip  (raw DICOM archives)
    ↓  [unzip]
extracted/
└── patient/study/series/*.dcm
    ↓  [SimpleITK] per series
NIfTI (.nii.gz)
    ↓  [MONAI RetinaNet — dt_model.ts]
Detected nodule bounding boxes
    ↓  [ResNet152 2D ensemble]
Malignancy probability per nodule
    ↓
output/mtn/results.csv
```

### The MTN dataset structure

The 5 ZIP files in `MTN/` are all from the same patient (`22018841-NGUYEN THI MECH`) and contain 4 different CT series:

| ZIP file | Series inside |
|----------|--------------|
| `...zip` | `3-1.25mm NHU MO PHOI` — lung parenchyma (primary series for nodule detection) |
| `...(1).zip` | `4-1.25mm TRUNG THAT` — mediastinum window |
| `...(2).zip` | `7-1.25mm DM` |
| `...(3).zip` + `...(4).zip` | `8-1.25mm TM` (series split across two ZIPs) |

The script handles extraction and deduplication automatically.

### Command

```bash
bash infer_mtn.sh /home/phonk/kaggle/MTN --output_dir ./output/mtn
```

That's it. The script extracts, detects, classifies, and writes the CSV.

### All options

| Option | Default | Description |
|--------|---------|-------------|
| `MTN_DIR` (positional) | *(required)* | Directory containing the MTN `.zip` files |
| `--output_dir` | `./output/mtn` | Where to save extracted files, JSON, and final CSV |
| `--model_type` | `3d` | `2d`, `3d`, or `both` |
| `--score_keep` | `0.3` | Detection confidence threshold — lower = more nodules found |
| `--threshold` | `0.5` | Malignancy probability cutoff |
| `--weights_dir` | `weights/ResNet152-confirmed/` | 2D checkpoint directory (optional override) |
| `--weights_dir_3d` | `weights/unet3D_encoder_scse/` | 3D checkpoint directory (optional override) |

### Output

```
output/mtn/
├── extracted/                          # Unzipped DICOM files (kept for re-runs)
│   └── 22018841-NGUYEN THI MECH/
│       └── HA251209.117_CT/
│           ├── 3-1.25mm NHU MO PHOI/
│           ├── 4-1.25mm TRUNG THAT/
│           └── ...
├── json/                               # Per-series raw results
│   ├── 3-1.25mm NHU MO PHOI_results.json
│   └── ...
└── results.csv                         # Final aggregated CSV
```

**`results.json` format** — one entry per series (case-level):

```json
[
  {
    "seriesInstanceUID": "3-1.25mm NHU MO PHOI",
    "probability": 0.1165,
    "predictionLabel": 0,
    "processingTimeMs": 4821,
    "CoordX": 34.30,
    "CoordY": -44.20,
    "CoordZ": -49.30
  },
  {
    "seriesInstanceUID": "4-1.25mm TRUNG THAT",
    "probability": 0.0,
    "predictionLabel": 0,
    "processingTimeMs": 1203,
    "CoordX": null,
    "CoordY": null,
    "CoordZ": null
  }
]
```

| Field | Description |
|-------|-------------|
| `seriesInstanceUID` | Series folder name |
| `probability` | Max malignancy probability across all detected nodules |
| `predictionLabel` | `0` = Benign, `1` = Malignant (case-level) |
| `processingTimeMs` | Total processing time for this series |
| `CoordX/Y/Z` | LPS world coordinates (mm) of the most suspicious nodule; `null` if no nodules detected |

### Re-runs

ZIPs are extracted with `-n` (no overwrite), so re-running the script is safe and fast — extraction is skipped if files already exist.

---

## 6. Model Types: 2D vs 3D vs Both

| `--model_type` | Architecture | Input patch | Checkpoint path |
|----------------|-------------|-------------|-----------------|
| `2d` | ResNet152, 5-fold ensemble | `(1, 64, 64)`, 50×50 mm | `weights/ResNet152-confirmed/fold_*/best_metric_model.pth` |
| `3d` *(default)* | UNet3D + scSE attention, 5-fold ensemble | `(1, 64, 64, 64)`, 50³ mm | `weights/unet3D_encoder_scse/best_metric_model_fold*.pth` |
| `both` | Average of 2D and 3D ensemble probabilities | both | both directories |

### When to use each

- **`3d`**: Captures volumetric context — default choice
- **`2d`**: Faster inference; useful for quick checks or when 3D weights are unavailable
- **`both`**: Most conservative — only flags Malignant when both models agree; reduces false positives

### Example: 3D inference

```bash
python infer.py \
    --ct patient.nii.gz \
    --coord_x -34.3 --coord_y 44.2 --coord_z -49.3 \
    --model_type 3d
```

### Example: Combined inference

```bash
python infer.py \
    --ct patient.nii.gz \
    --coord_x -34.3 --coord_y 44.2 --coord_z -49.3 \
    --model_type both \
    --threshold 0.45
```

---

## 7. Detection Setup (`dt_model.ts`)

The detection model is a MONAI RetinaNet exported to TorchScript format.

### Location

```
weights/dt_model.ts   (≈80 MB)
```

### How it works

1. Takes a NIfTI CT volume as input
2. Applies a sliding-window inference across the 3D volume
3. Outputs bounding boxes in RAS world coordinates: `[cx, cy, cz, w, h, d]` (mm)
4. Boxes with `score < --score_keep` are discarded

### The detection model is called automatically by `predict.py`

You do not need to interact with it directly. If you want to use a different detection model:

```bash
python predict.py \
    --dicom_dir /path/to/dicom/ \
    --model_path /path/to/custom_model.ts
```

### GPU memory requirements

- Detection: ~4 GB VRAM (sliding window, 512×512×N volumes)
- 2D classification: ~2 GB VRAM
- 3D classification: ~3 GB VRAM

Run on CPU if no GPU available (significantly slower):
```bash
# No special flag needed — falls back automatically if no CUDA
python predict.py --dicom_dir /path/to/dicom/
```

---

## 8. Coordinate System Reference

Understanding coordinate systems is critical for passing the right values.

### Input coordinates for `infer.py`

`infer.py` expects **ITK/LPS world coordinates** in mm.

- **L** (Left): +x points Left
- **P** (Posterior): +y points Posterior
- **S** (Superior): +z points Superior

These come from:
- DICOM `ImagePositionPatient` tags
- SimpleITK `GetOrigin()` + affine transform
- Radiologist reports (usually in LPS or RAS — check convention)

### Where coordinates come from in the pipeline

```
MONAI RetinaNet
  → outputs RAS bounding boxes: [cx_RAS, cy_RAS, cz_RAS, w, h, d]

Conversion to LPS (for classifier):
  x_LPS = -cx_RAS
  y_LPS = -cy_RAS
  z_LPS =  cz_RAS

predict.py handles this conversion automatically.
```

### If you have RAS coordinates from another tool

Convert before passing to `infer.py`:

```python
x_lps = -x_ras
y_lps = -y_ras
z_lps =  z_ras  # z is the same in RAS and LPS
```

Then:

```bash
python infer.py --ct patient.nii.gz \
    --coord_x $x_lps --coord_y $y_lps --coord_z $z_lps
```

---

## 9. Troubleshooting

| Problem | Cause | Fix |
|---------|-------|-----|
| `FileNotFoundError: weights/ResNet152-confirmed/fold_1/best_metric_model.pth` | Missing checkpoints | Ensure all 5 fold weights are present; or pass `--weights_dir` pointing to your trained checkpoints |
| `FileNotFoundError: weights/unet3D_encoder_scse/...` | Missing 3D checkpoints | Required by default. Copy weights to `weights/unet3D_encoder_scse/` or pass `--weights_dir_3d` to a custom path |
| `FileNotFoundError: weights/dt_model.ts` | Missing detection model | Required only for `predict.py`. Place `dt_model.ts` in `weights/` or pass `--model_path` |
| `RuntimeError: CUDA out of memory` | Insufficient VRAM | Reduce batch size (detection), or switch to CPU (slower) |
| Probability always near 0 | Wrong coordinate system | Verify coords are in LPS (not voxel indices). Check if your tool outputs RAS — if so, negate x and y |
| Probability always near 1 | Wrong coordinate system or threshold | Try lowering `--threshold` to 0.3; verify coordinate values |
| `No DICOM files found` | Wrong directory | Pass the folder that directly contains `.dcm` files, not a parent folder |
| `Multiple DICOM series found` | Mixed series in folder | Separate each series into its own folder |
| `No nodules detected` | Low-confidence detections | Lower `--score_keep` (e.g., `0.1`) to include more candidates |
| Inference very slow (no GPU) | Running on CPU | Normal — CPU inference takes ~30–60s per CT volume |
