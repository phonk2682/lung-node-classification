#!/usr/bin/env bash
# infer_mtn.sh — Extract MTN ZIPs, run end-to-end nodule detection + classification,
#                 and save all results as a single CSV.
#
# Usage:
#   bash infer_mtn.sh <MTN_DIR> [--output_dir DIR] [--weights_dir DIR]
#                                [--model_type 2d|3d|both]
#                                [--score_keep FLOAT] [--threshold FLOAT]
#
# Example:
#   bash infer_mtn.sh /home/phonk/kaggle/MTN --output_dir ./output/mtn
#
# Requirements:
#   - conda activate cs-problem (or equivalent env with pipeline deps installed)
#   - weights/ directory with dt_model.ts and ResNet152-confirmed/fold_*/
#   - unzip installed
set -euo pipefail

# ── Argument parsing ──────────────────────────────────────────────────────────
if [[ $# -lt 1 ]]; then
    echo "Usage: bash infer_mtn.sh <MTN_DIR> [options]"
    echo ""
    echo "Options:"
    echo "  --output_dir DIR        Output directory (default: ./output/mtn)"
    echo "  --weights_dir DIR       2D checkpoint dir (default: weights/ResNet152-confirmed)"
    echo "  --weights_dir_3d DIR    3D checkpoint dir (default: weights/unet3D_encoder_scse)"
    echo "  --model_type TYPE       2d | 3d | both  (default: 3d)"
    echo "  --score_keep FLOAT      Detection confidence threshold (default: 0.3)"
    echo "  --threshold FLOAT       Malignancy threshold (default: 0.5)"
    exit 1
fi

MTN_DIR="$1"; shift

# Defaults
OUTPUT_DIR="./output/mtn"
WEIGHTS_DIR=""
WEIGHTS_DIR_3D=""
MODEL_TYPE="3d"
SCORE_KEEP="0.3"
THRESHOLD="0.5"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --output_dir)    OUTPUT_DIR="$2";    shift 2 ;;
        --weights_dir)   WEIGHTS_DIR="$2";   shift 2 ;;
        --weights_dir_3d) WEIGHTS_DIR_3D="$2"; shift 2 ;;
        --model_type)    MODEL_TYPE="$2";    shift 2 ;;
        --score_keep)    SCORE_KEEP="$2";    shift 2 ;;
        --threshold)     THRESHOLD="$2";     shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MTN_DIR="$(realpath "$MTN_DIR")"
OUTPUT_DIR="$(realpath "$OUTPUT_DIR")"
EXTRACTED_DIR="$OUTPUT_DIR/extracted"
JSON_DIR="$OUTPUT_DIR/json"

mkdir -p "$EXTRACTED_DIR" "$JSON_DIR"

# ── Optional weights args ─────────────────────────────────────────────────────
EXTRA_ARGS=""
[[ -n "$WEIGHTS_DIR"    ]] && EXTRA_ARGS="$EXTRA_ARGS --weights_dir \"$WEIGHTS_DIR\""
[[ -n "$WEIGHTS_DIR_3D" ]] && EXTRA_ARGS="$EXTRA_ARGS --weights_dir_3d \"$WEIGHTS_DIR_3D\""

echo "============================================================"
echo " MTN Inference Pipeline"
echo "============================================================"
echo " MTN dir    : $MTN_DIR"
echo " Output dir : $OUTPUT_DIR"
echo " Model type : $MODEL_TYPE"
echo " Score keep : $SCORE_KEEP"
echo " Threshold  : $THRESHOLD"
echo "============================================================"

# ── Step 1: Extract ZIPs ──────────────────────────────────────────────────────
echo ""
echo "[Step 1] Extracting ZIP files..."

shopt -s nullglob
zip_files=("$MTN_DIR"/*.zip)

if [[ ${#zip_files[@]} -eq 0 ]]; then
    echo "[ERROR] No ZIP files found in: $MTN_DIR"
    exit 1
fi

for zip in "${zip_files[@]}"; do
    echo "  Extracting: $(basename "$zip")"
    unzip -q -n "$zip" -d "$EXTRACTED_DIR"
done

echo "  Done. Extracted to: $EXTRACTED_DIR"

# ── Step 2: Find DICOM series directories ─────────────────────────────────────
echo ""
echo "[Step 2] Finding DICOM series directories..."

# A DICOM series dir is a directory that directly contains .dcm files
mapfile -t SERIES_DIRS < <(
    find "$EXTRACTED_DIR" -type f -name "*.dcm" \
        | xargs -I{} dirname {} \
        | sort -u
)

if [[ ${#SERIES_DIRS[@]} -eq 0 ]]; then
    echo "[ERROR] No DICOM files found under: $EXTRACTED_DIR"
    exit 1
fi

echo "  Found ${#SERIES_DIRS[@]} series:"
for d in "${SERIES_DIRS[@]}"; do
    echo "    - $(realpath --relative-to="$EXTRACTED_DIR" "$d")"
done

# ── Step 3: Run predict.py for each series ────────────────────────────────────
echo ""
echo "[Step 3] Running pipeline on each series..."

total=${#SERIES_DIRS[@]}
idx=0

for series_dir in "${SERIES_DIRS[@]}"; do
    idx=$((idx + 1))
    series_name="$(basename "$series_dir")"
    patient_name="$(basename "$(dirname "$(dirname "$series_dir")")")"

    echo ""
    echo "  [$idx/$total] Patient : $patient_name"
    echo "              Series  : $series_name"

    eval python "$SCRIPT_DIR/predict.py" \
        --dicom_dir "\"$series_dir\"" \
        --output_dir "\"$JSON_DIR\"" \
        --model_type "$MODEL_TYPE" \
        --score_keep "$SCORE_KEEP" \
        --threshold  "$THRESHOLD" \
        $EXTRA_ARGS

done

# ── Step 4: Aggregate JSON results ───────────────────────────────────────────
JSON_OUT="$OUTPUT_DIR/results.json"

echo ""
echo "[Step 4] Building results.json ..."

python - <<PYEOF
import json
from pathlib import Path

json_dir = Path("$JSON_DIR")
out_path = Path("$JSON_OUT")

results = []
for jf in sorted(json_dir.glob("*_results.json")):
    try:
        data = json.loads(jf.read_text())
    except Exception as e:
        print(f"  [WARN] Could not read {jf.name}: {e}")
        continue
    results.append(data)

if not results:
    print("  [WARN] No JSON result files found.")
else:
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Wrote {len(results)} case(s) to: {out_path}")
PYEOF

echo ""
echo "============================================================"
echo " ALL DONE"
echo " Results     : $JSON_OUT"
echo " Raw JSON    : $JSON_DIR/"
echo "============================================================"
