"""
predict.py — End-to-end lung nodule detection and classification pipeline.

Workflow:
  1. Convert a patient's DICOM series to NIfTI (.nii.gz) via preprocess.py
  2. Detect nodules in the NIfTI volume via lung_nodule_detection.py (MONAI RetinaNet)
  3. For each detected nodule, classify it as Benign / Malignant via classify.py
  4. Print and optionally save results

Usage:
    python predict.py --dicom_dir /path/to/patient_dicom/
    python predict.py --dicom_dir /path/to/patient_dicom/ --output_dir ./output --score_keep 0.3

Notes on coordinate conventions
---------------------------------
The MONAI detection pipeline (lung_nodule_detection.py) outputs bounding boxes in
world (physical) space with RAS orientation after applying affine_lps_to_ras=True.
Box format after ConvertBoxModed(src_mode="xyzxyz", dst_mode="cccwhd"):
    [cx_RAS, cy_RAS, cz_RAS, w, h, d]  (all in mm)

The classification pipeline (classify.py) expects world coordinates in ITK/LPS space
as [x_LPS, y_LPS, z_LPS].

Conversion from RAS -> LPS:
    x_LPS = -cx_RAS
    y_LPS = -cy_RAS
    z_LPS =  cz_RAS
"""

import argparse
from lung_nodule.pipeline.pipeline import run_pipeline


def main():
    parser = argparse.ArgumentParser(
        description="End-to-end lung nodule detection and classification pipeline"
    )
    parser.add_argument(
        "--dicom_dir",
        required=True,
        help="Path to a folder containing one patient's DICOM series",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Directory to save intermediate NIfTI and results JSON (default: temp dir)",
    )
    parser.add_argument(
        "--weights_dir",
        default=None,
        help="2D classification fold directory (default: weights/ResNet152-confirmed/)",
    )
    parser.add_argument(
        "--weights_dir_3d",
        default=None,
        help="3D classification fold directory (default: weights/unet3D_encoder_scse/)",
    )
    parser.add_argument(
        "--model_type", default="3d", choices=["2d", "3d", "both"],
        help="Classification model: 2d (ResNet152), 3d (UNet3D), both (default: 3d)",
    )
    parser.add_argument(
        "--model_path",
        default=None,
        help="Detection model path (default: weights/dt_model.ts)",
    )
    parser.add_argument(
        "--score_keep",
        type=float,
        default=0.3,
        help="Minimum detection confidence score (default: 0.3)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Malignancy probability threshold (default: 0.5)",
    )
    args = parser.parse_args()

    run_pipeline(
        dicom_dir=args.dicom_dir,
        output_dir=args.output_dir,
        weights_dir=args.weights_dir,
        weights_dir_3d=args.weights_dir_3d,
        model_type=args.model_type,
        model_path=args.model_path,
        score_keep=args.score_keep,
        threshold=args.threshold,
    )


if __name__ == "__main__":
    main()
