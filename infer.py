"""
infer.py — Binary lung nodule malignancy classification.

Accepts a CT image (any format readable by SimpleITK: .mha, .nii.gz, .nii, etc.)
and one or more nodule world coordinates, then outputs Benign / Malignant for each.

Supports two modes:
  1. Single nodule via CLI flags
  2. Batch mode via a CSV file

-----------------------------------------------------------------------
SINGLE NODULE
-----------------------------------------------------------------------
    python infer.py \\
        --ct      patient.nii.gz \\
        --coord_x -34.3 \\
        --coord_y  44.2 \\
        --coord_z -49.3

    Output:
        ┌────────────────────────────────────────────────────────────┐
        │  Nodule Malignancy Classification                          │
        │  CT      : patient.nii.gz                                  │
        │  Coord   : x=-34.30  y=44.20  z=-49.30  (world, LPS mm)  │
        │  ──────────────────────────────────────────────────────── │
        │  Probability : 0.1165                                      │
        │  Label       : 0 — Benign                                  │
        └────────────────────────────────────────────────────────────┘

-----------------------------------------------------------------------
BATCH MODE  (CSV must have columns: ct_path, coord_x, coord_y, coord_z)
-----------------------------------------------------------------------
    python infer.py --csv nodules.csv --output results.csv

    Output CSV columns:
        ct_path, coord_x, coord_y, coord_z, probability, label, label_str

-----------------------------------------------------------------------
OPTIONS
-----------------------------------------------------------------------
    --weights_dir   Path to 2D fold checkpoint directory
                    (default: weights/ResNet152-confirmed/)
    --weights_dir_3d Path to 3D fold checkpoint directory
                    (default: weights/unet3D_encoder_scse/)
    --model_type    2d | 3d | both  (default: 3d)
                    2d   — ResNet152 2D ensemble
                    3d   — UNet3D encoder+scSE ensemble
                    both — average probabilities from 2D and 3D
    --threshold     Probability threshold for Malignant (default: 0.5)
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
import SimpleITK

from lung_nodule import data as dataloader
from lung_nodule.models.model_2d import ResNet152
from lung_nodule.models.unet3d_encoder_se import UNet3DEncoderClassifier


# ── image utilities ──────────────────────────────────────────────────────────

def _transform(itk_image, point):
    return np.array(
        list(reversed(
            itk_image.TransformContinuousIndexToPhysicalPoint(list(reversed(point)))
        ))
    )


def _load_ct(ct_path: str):
    """Load a CT image and return (numpy_array, header_dict)."""
    itk_image = SimpleITK.ReadImage(ct_path)
    numpy_image = SimpleITK.GetArrayFromImage(itk_image)
    origin  = np.array(list(reversed(itk_image.GetOrigin())))
    spacing = np.array(list(reversed(itk_image.GetSpacing())))

    t_origin = _transform(itk_image, np.zeros((numpy_image.ndim,)))
    components = [None] * numpy_image.ndim
    for i in range(numpy_image.ndim):
        v = [0] * numpy_image.ndim
        v[i] = 1
        components[i] = _transform(itk_image, v) - t_origin
    transform = np.vstack(components).dot(np.diag(1.0 / spacing))

    header = {"origin": origin, "spacing": spacing, "transform": transform}
    return numpy_image, header


# ── patch extraction ─────────────────────────────────────────────────────────

SIZE_PX = 64
SIZE_MM = 50


def _extract_patch(numpy_image, header, coord_zyx: np.ndarray) -> np.ndarray:
    patch = dataloader.extract_patch(
        CTData=numpy_image,
        coord=coord_zyx,
        srcVoxelOrigin=header["origin"],
        srcWorldMatrix=header["transform"],
        srcVoxelSpacing=header["spacing"],
        output_shape=[1, SIZE_PX, SIZE_PX],
        voxel_spacing=(SIZE_MM / SIZE_PX,) * 3,
        coord_space_world=True,
        mode="2D",
    )
    return dataloader.clip_and_scale(patch.astype(np.float32))


def _extract_patch_3d(numpy_image, header, coord_zyx: np.ndarray) -> np.ndarray:
    patch = dataloader.extract_patch(
        CTData=numpy_image,
        coord=coord_zyx,
        srcVoxelOrigin=header["origin"],
        srcWorldMatrix=header["transform"],
        srcVoxelSpacing=header["spacing"],
        output_shape=[SIZE_PX, SIZE_PX, SIZE_PX],
        voxel_spacing=(SIZE_MM / SIZE_PX,) * 3,
        coord_space_world=True,
        mode="3D",
    )
    return dataloader.clip_and_scale(patch.astype(np.float32))


# ── ensemble inference ───────────────────────────────────────────────────────

def _predict(numpy_image, header, coord_xyz, weights_dir: Path, threshold: float) -> dict:
    """Core inference: world coord [x,y,z] (LPS) -> probability + label."""
    fold_ckpts = sorted([
        p / "best_metric_model.pth"
        for p in sorted(weights_dir.iterdir())
        if p.is_dir() and (p / "best_metric_model.pth").exists()
    ])
    if not fold_ckpts:
        raise FileNotFoundError(f"No fold checkpoints found in: {weights_dir}")

    x, y, z = coord_xyz
    coord_zyx = np.array([z, y, x])          # ITK [x,y,z] -> numpy [z,y,x]
    patch = _extract_patch(numpy_image, header, coord_zyx)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    patch_tensor = torch.from_numpy(patch).unsqueeze(0).to(device)

    all_logits = []
    for ckpt_path in fold_ckpts:
        model = ResNet152(weights=None).to(device)
        model.load_state_dict(
            torch.load(str(ckpt_path), map_location=device, weights_only=False)
        )
        model.eval()
        with torch.no_grad():
            logit = model(patch_tensor).cpu().numpy()[0, 0]
        all_logits.append(float(logit))
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    prob  = float(1.0 / (1.0 + np.exp(-np.mean(all_logits))))
    label = int(prob >= threshold)
    return {"probability": prob, "label": label, "label_str": "Malignant" if label else "Benign"}


def _predict_3d(numpy_image, header, coord_xyz, weights_dir_3d: Path, threshold: float) -> dict:
    """3D UNet ensemble inference: world coord [x,y,z] (LPS) -> probability + label."""
    fold_ckpts = sorted(weights_dir_3d.glob("best_metric_model_fold*.pth"))
    if not fold_ckpts:
        raise FileNotFoundError(f"No 3D fold checkpoints found in: {weights_dir_3d}")

    x, y, z = coord_xyz
    coord_zyx = np.array([z, y, x])
    patch = _extract_patch_3d(numpy_image, header, coord_zyx)  # (D, H, W)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # dataloader already adds channel dim -> patch is (1, D, H, W); add batch dim -> (1, 1, D, H, W)
    patch_tensor = torch.from_numpy(patch).unsqueeze(0).to(device)

    all_logits = []
    for ckpt_path in fold_ckpts:
        model = UNet3DEncoderClassifier(in_channels=1, num_classes=1).to(device)
        model.load_state_dict(
            torch.load(str(ckpt_path), map_location=device, weights_only=False)
        )
        model.eval()
        with torch.no_grad():
            logit = model(patch_tensor).cpu().numpy()[0]   # model returns shape [B] when num_classes=1
        all_logits.append(float(logit))
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    prob  = float(1.0 / (1.0 + np.exp(-np.mean(all_logits))))
    label = int(prob >= threshold)
    return {"probability": prob, "label": label, "label_str": "Malignant" if label else "Benign"}


def _predict_combined(numpy_image, header, coord_xyz,
                       weights_dir: Path, weights_dir_3d: Path, threshold: float) -> dict:
    """Average 2D and 3D ensemble probabilities."""
    res2d = _predict(numpy_image, header, coord_xyz, weights_dir, threshold=0.5)
    res3d = _predict_3d(numpy_image, header, coord_xyz, weights_dir_3d, threshold=0.5)
    prob  = (res2d["probability"] + res3d["probability"]) / 2.0
    label = int(prob >= threshold)
    return {"probability": prob, "label": label, "label_str": "Malignant" if label else "Benign"}


# ── single-nodule mode ────────────────────────────────────────────────────────

def run_single(ct_path, coord_x, coord_y, coord_z, weights_dir, weights_dir_3d,
               model_type, threshold):
    print(f"[INFO] Loading CT: {ct_path}")
    numpy_image, header = _load_ct(ct_path)

    coord_xyz = [coord_x, coord_y, coord_z]
    if model_type == "3d":
        result = _predict_3d(numpy_image, header, coord_xyz, weights_dir_3d, threshold)
    elif model_type == "both":
        result = _predict_combined(numpy_image, header, coord_xyz,
                                   weights_dir, weights_dir_3d, threshold)
    else:
        result = _predict(numpy_image, header, coord_xyz=coord_xyz,
                          weights_dir=weights_dir, threshold=threshold)

    width = 62
    print("\n┌" + "─" * width + "┐")
    print(f"│  {'Nodule Malignancy Classification':<{width-2}}│")
    print(f"│  {'CT      : ' + str(ct_path):<{width-2}}│")
    print(f"│  {f'Coord   : x={coord_x:.2f}  y={coord_y:.2f}  z={coord_z:.2f}  (world, LPS mm)':<{width-2}}│")
    print("│  " + "─" * (width - 4) + "  │")
    prob_line  = f'Probability : {result["probability"]:.4f}'
    label_line = f'Label       : {result["label"]} \u2014 {result["label_str"]}'
    print(f"│  {prob_line:<{width-2}}│")
    print(f"│  {label_line:<{width-2}}│")
    print("└" + "─" * width + "┘\n")

    return result


# ── batch mode ────────────────────────────────────────────────────────────────

def run_batch(csv_input: str, csv_output: str, weights_dir: Path, weights_dir_3d: Path,
              model_type: str, threshold: float):
    rows = []
    with open(csv_input, newline="") as f:
        reader = csv.DictReader(f)
        required = {"ct_path", "coord_x", "coord_y", "coord_z"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            print(f"[ERROR] CSV is missing columns: {missing}", file=sys.stderr)
            sys.exit(1)
        rows = list(reader)

    print(f"[INFO] Batch mode: {len(rows)} nodules from {csv_input}")

    # Cache loaded CT volumes so we don't re-read the same file multiple times
    ct_cache = {}
    results = []

    for i, row in enumerate(rows, 1):
        ct_path = row["ct_path"].strip()
        cx = float(row["coord_x"])
        cy = float(row["coord_y"])
        cz = float(row["coord_z"])

        print(f"  [{i}/{len(rows)}] {ct_path}  ({cx:.1f}, {cy:.1f}, {cz:.1f})", end=" ... ")

        if ct_path not in ct_cache:
            ct_cache[ct_path] = _load_ct(ct_path)
        numpy_image, header = ct_cache[ct_path]

        try:
            coord_xyz = [cx, cy, cz]
            if model_type == "3d":
                res = _predict_3d(numpy_image, header, coord_xyz, weights_dir_3d, threshold)
            elif model_type == "both":
                res = _predict_combined(numpy_image, header, coord_xyz,
                                        weights_dir, weights_dir_3d, threshold)
            else:
                res = _predict(numpy_image, header, coord_xyz=coord_xyz,
                               weights_dir=weights_dir, threshold=threshold)
            print(f"{res['label_str']}  (p={res['probability']:.4f})")
        except Exception as e:
            print(f"ERROR: {e}")
            res = {"probability": None, "label": None, "label_str": "ERROR"}

        results.append({
            "ct_path":     ct_path,
            "coord_x":     cx,
            "coord_y":     cy,
            "coord_z":     cz,
            "probability": res["probability"],
            "label":       res["label"],
            "label_str":   res["label_str"],
        })

    # Write output CSV
    out_path = Path(csv_output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["ct_path", "coord_x", "coord_y", "coord_z",
                           "probability", "label", "label_str"]
        )
        writer.writeheader()
        writer.writerows(results)

    n_mal = sum(1 for r in results if r["label"] == 1)
    n_ben = sum(1 for r in results if r["label"] == 0)
    print(f"\nResults: {n_ben} Benign  |  {n_mal} Malignant")
    print(f"Saved to: {out_path}")

    return results


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Lung nodule binary malignancy classification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # --- common ---
    parser.add_argument(
        "--weights_dir", default=None,
        help="Folder containing fold_*/best_metric_model.pth for 2D model "
             "(default: weights/ResNet152-confirmed/ relative to this script)",
    )
    parser.add_argument(
        "--weights_dir_3d", default=None,
        help="Folder containing best_metric_model_fold*.pth for 3D model "
             "(default: weights/unet3D_encoder_scse/ relative to this script)",
    )
    parser.add_argument(
        "--model_type", default="3d", choices=["2d", "3d", "both"],
        help="Model to use: 2d (ResNet152), 3d (UNet3D), both (average) — default: 3d",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.5,
        help="Probability threshold for Malignant label (default: 0.5)",
    )

    # --- single-nodule mode ---
    single = parser.add_argument_group("Single-nodule mode")
    single.add_argument("--ct",      help="CT image file (.mha, .nii.gz, ...)")
    single.add_argument("--coord_x", type=float, help="Nodule X (world, LPS, mm)")
    single.add_argument("--coord_y", type=float, help="Nodule Y (world, LPS, mm)")
    single.add_argument("--coord_z", type=float, help="Nodule Z (world, LPS, mm)")

    # --- batch mode ---
    batch = parser.add_argument_group("Batch mode")
    batch.add_argument(
        "--csv", help="Input CSV with columns: ct_path, coord_x, coord_y, coord_z",
    )
    batch.add_argument(
        "--output", default="./infer_results.csv",
        help="Output CSV path for batch results (default: ./infer_results.csv)",
    )

    args = parser.parse_args()

    # Resolve weights dirs
    weights_dir = Path(args.weights_dir) if args.weights_dir else \
                  Path(__file__).parent / "weights" / "ResNet152-confirmed"
    weights_dir_3d = Path(args.weights_dir_3d) if args.weights_dir_3d else \
                     Path(__file__).parent / "weights" / "unet3D_encoder_scse"

    if args.model_type in ("2d", "both") and not weights_dir.exists():
        print(f"[ERROR] 2D weights directory not found: {weights_dir}", file=sys.stderr)
        sys.exit(1)
    if args.model_type in ("3d", "both") and not weights_dir_3d.exists():
        print(f"[ERROR] 3D weights directory not found: {weights_dir_3d}", file=sys.stderr)
        sys.exit(1)

    # Dispatch
    if args.csv:
        run_batch(args.csv, args.output, weights_dir, weights_dir_3d,
                  args.model_type, args.threshold)
    elif args.ct and args.coord_x is not None and args.coord_y is not None and args.coord_z is not None:
        run_single(args.ct, args.coord_x, args.coord_y, args.coord_z,
                   weights_dir, weights_dir_3d, args.model_type, args.threshold)
    else:
        parser.print_help()
        print("\n[ERROR] Provide either --ct + --coord_x/y/z  OR  --csv <file>",
              file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
