"""
preprocess.py — Convert a DICOM series directory to a NIfTI (.nii.gz) file.

Usage:
    python preprocess.py --dicom_dir /path/to/dicom_folder --output /path/to/output.nii.gz

The output .nii.gz preserves the spatial metadata (origin, spacing, direction)
required by lung_nodule_detection.py and classify.py.
"""

import argparse
import sys
from pathlib import Path

import SimpleITK as sitk


def dicom_to_nifti(dicom_dir: str, output_path: str) -> None:
    """Read a DICOM series and write a single compressed NIfTI volume.

    Parameters
    ----------
    dicom_dir : str
        Path to a directory containing one DICOM series.
    output_path : str
        Destination path; must end in .nii.gz or .nii.
    """
    dicom_dir = Path(dicom_dir)
    output_path = Path(output_path)

    if not dicom_dir.is_dir():
        raise NotADirectoryError(f"DICOM directory not found: {dicom_dir}")

    reader = sitk.ImageSeriesReader()
    series_ids = reader.GetGDCMSeriesIDs(str(dicom_dir))

    if not series_ids:
        raise RuntimeError(f"No DICOM series found in: {dicom_dir}")

    if len(series_ids) > 1:
        print(
            f"[WARNING] Multiple DICOM series found ({len(series_ids)}). "
            f"Using the first series: {series_ids[0]}"
        )

    dicom_files = reader.GetGDCMSeriesFileNames(str(dicom_dir), series_ids[0])
    reader.SetFileNames(dicom_files)

    print(f"[INFO] Reading {len(dicom_files)} DICOM slices from: {dicom_dir}")
    image = reader.Execute()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(image, str(output_path))
    print(f"[INFO] Saved NIfTI volume to: {output_path}")
    print(f"       Size:    {image.GetSize()}")
    print(f"       Spacing: {image.GetSpacing()}")
    print(f"       Origin:  {image.GetOrigin()}")
