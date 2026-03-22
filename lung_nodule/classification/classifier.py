"""
classifier.py — Lung nodule malignancy classification using a 5-fold ResNet152 ensemble.

Accepts a CT image file (.mha, .nii.gz, etc.) and a nodule center coordinate in world
space, and returns a malignancy probability together with a binary label.

Usage as a module:
    from lung_nodule.classification.classifier import classify_nodule
    result = classify_nodule(
        ct_path="patient.nii.gz",
        coord_world_xyz=[x, y, z],       # world coords (ITK/LPS, in mm)
        weights_dir="weights/ResNet152-confirmed",
    )
    print(result)  # {"probability": 0.73, "label": 1, "label_str": "Malignant"}
"""

import os
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import SimpleITK

from lung_nodule import data as dataloader
from lung_nodule.models.model_2d import ResNet152
from lung_nodule.models.unet3d_encoder_se import UNet3DEncoderClassifier


# ---------------------------------------------------------------------------
# Image utilities (mirrored from inference.py)
# ---------------------------------------------------------------------------

def _transform(itk_image, point):
    return np.array(
        list(reversed(itk_image.TransformContinuousIndexToPhysicalPoint(list(reversed(point)))))
    )


def itk_image_to_numpy(itk_image):
    """Convert a SimpleITK image to a numpy array plus spatial header."""
    numpy_image = SimpleITK.GetArrayFromImage(itk_image)
    origin = np.array(list(reversed(itk_image.GetOrigin())))    # [z, y, x]
    spacing = np.array(list(reversed(itk_image.GetSpacing())))  # [z, y, x]

    t_origin = _transform(itk_image, np.zeros((numpy_image.ndim,)))
    t_components = [None] * numpy_image.ndim
    for i in range(numpy_image.ndim):
        v = [0] * numpy_image.ndim
        v[i] = 1
        t_components[i] = _transform(itk_image, v) - t_origin
    transform = np.vstack(t_components).dot(np.diag(1.0 / spacing))

    header = {"origin": origin, "spacing": spacing, "transform": transform}
    return numpy_image, header


# ---------------------------------------------------------------------------
# Ensemble classifier
# ---------------------------------------------------------------------------

SIZE_PX = 64
SIZE_MM = 50


def _load_model_2d(weights_path: str) -> torch.nn.Module:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ResNet152(weights=None).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=False))
    model.eval()
    return model


def _load_model_3d(weights_path: str) -> torch.nn.Module:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = UNet3DEncoderClassifier(in_channels=1, num_classes=1).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=False))
    model.eval()
    return model


def _extract_patch(numpy_image, header, coord_zyx: np.ndarray) -> np.ndarray:
    """Extract and normalise a 64x64 patch centred on coord_zyx (world, [z,y,x])."""
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
    """Extract and normalise a 64^3 patch centred on coord_zyx (world, [z,y,x])."""
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


def _ensemble_2d(patch, fold_paths):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    patch_tensor = torch.from_numpy(patch).unsqueeze(0).to(device)
    logits = []
    for p in fold_paths:
        model = _load_model_2d(str(p))
        with torch.no_grad():
            logits.append(float(model(patch_tensor).cpu().numpy()[0, 0]))
        del model
        if device == "cuda":
            torch.cuda.empty_cache()
    return float(np.mean(logits))


def _ensemble_3d(patch, fold_paths):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # dataloader adds channel dim for 3D -> (1, D, H, W); add batch -> (1, 1, D, H, W)
    patch_tensor = torch.from_numpy(patch).unsqueeze(0).to(device)
    logits = []
    for p in fold_paths:
        model = _load_model_3d(str(p))
        with torch.no_grad():
            logits.append(float(model(patch_tensor).cpu().numpy()[0]))
        del model
        if device == "cuda":
            torch.cuda.empty_cache()
    return float(np.mean(logits))


def classify_nodule(
    ct_path: str,
    coord_world_xyz: List[float],
    weights_dir: str = None,
    weights_dir_3d: str = None,
    model_type: str = "2d",
    threshold: float = 0.5,
) -> Dict:
    """Classify a single lung nodule as benign or malignant.

    Parameters
    ----------
    ct_path : str
        Path to a CT volume (any format readable by SimpleITK: .mha, .nii.gz, etc.).
    coord_world_xyz : list of float
        Nodule centre in ITK/LPS world coordinates as [x, y, z] in millimetres.
    weights_dir : str, optional
        Directory containing fold_*/best_metric_model.pth (2D model).
        Defaults to weights/ResNet152-confirmed/ relative to this file.
    weights_dir_3d : str, optional
        Directory containing best_metric_model_fold*.pth (3D model).
        Defaults to weights/unet3D_encoder_scse/ relative to this file.
    model_type : str
        "2d", "3d", or "both" (default "3d").
    threshold : float
        Probability threshold for the binary decision (default 0.5).

    Returns
    -------
    dict
        {
            "probability": float,   # malignancy probability in [0, 1]
            "label": int,           # 0 = Benign, 1 = Malignant
            "label_str": str,       # "Benign" or "Malignant"
        }
    """
    pipeline_root = Path(__file__).parent.parent.parent
    if weights_dir is None:
        weights_dir = str(pipeline_root / "weights" / "ResNet152-confirmed")
    if weights_dir_3d is None:
        weights_dir_3d = str(pipeline_root / "weights" / "unet3D_encoder_scse")

    # Load CT image
    itk_image = SimpleITK.ReadImage(str(ct_path))
    numpy_image, header = itk_image_to_numpy(itk_image)

    x, y, z = coord_world_xyz
    coord_zyx = np.array([z, y, x])

    # Run requested model(s)
    if model_type == "2d":
        fold_paths = sorted([
            p / "best_metric_model.pth"
            for p in sorted(Path(weights_dir).iterdir())
            if p.is_dir() and (p / "best_metric_model.pth").exists()
        ])
        if not fold_paths:
            raise FileNotFoundError(f"No 2D fold checkpoints found in: {weights_dir}")
        patch = _extract_patch(numpy_image, header, coord_zyx)
        mean_logit = _ensemble_2d(patch, fold_paths)

    elif model_type == "3d":
        fold_paths = sorted(Path(weights_dir_3d).glob("best_metric_model_fold*.pth"))
        if not fold_paths:
            raise FileNotFoundError(f"No 3D fold checkpoints found in: {weights_dir_3d}")
        patch = _extract_patch_3d(numpy_image, header, coord_zyx)
        mean_logit = _ensemble_3d(patch, fold_paths)

    else:  # both
        fold_paths_2d = sorted([
            p / "best_metric_model.pth"
            for p in sorted(Path(weights_dir).iterdir())
            if p.is_dir() and (p / "best_metric_model.pth").exists()
        ])
        fold_paths_3d = sorted(Path(weights_dir_3d).glob("best_metric_model_fold*.pth"))
        if not fold_paths_2d:
            raise FileNotFoundError(f"No 2D fold checkpoints found in: {weights_dir}")
        if not fold_paths_3d:
            raise FileNotFoundError(f"No 3D fold checkpoints found in: {weights_dir_3d}")
        patch_2d = _extract_patch(numpy_image, header, coord_zyx)
        patch_3d = _extract_patch_3d(numpy_image, header, coord_zyx)
        logit_2d = _ensemble_2d(patch_2d, fold_paths_2d)
        logit_3d = _ensemble_3d(patch_3d, fold_paths_3d)
        prob_2d = 1.0 / (1.0 + np.exp(-logit_2d))
        prob_3d = 1.0 / (1.0 + np.exp(-logit_3d))
        probability = float((prob_2d + prob_3d) / 2.0)
        label = int(probability >= threshold)
        return {"probability": probability, "label": label,
                "label_str": "Malignant" if label else "Benign"}

    probability = float(1.0 / (1.0 + np.exp(-mean_logit)))
    label = int(probability >= threshold)
    return {
        "probability": probability,
        "label": label,
        "label_str": "Malignant" if label == 1 else "Benign",
    }
