"""
pipeline.py — End-to-end lung nodule detection and classification pipeline.

Workflow:
  1. Convert a patient's DICOM series to NIfTI (.nii.gz) via preprocess.py
  2. Detect nodules in the NIfTI volume via the MONAI RetinaNet detector
  3. For each detected nodule, classify it as Benign / Malignant
  4. Print and optionally save results

Notes on coordinate conventions
---------------------------------
The MONAI detection pipeline outputs bounding boxes in world (physical) space with
RAS orientation after applying affine_lps_to_ras=True.
Box format after ConvertBoxModed(src_mode="xyzxyz", dst_mode="cccwhd"):
    [cx_RAS, cy_RAS, cz_RAS, w, h, d]  (all in mm)

The classification pipeline expects world coordinates in ITK/LPS space
as [x_LPS, y_LPS, z_LPS].

Conversion from RAS -> LPS:
    x_LPS = -cx_RAS
    y_LPS = -cy_RAS
    z_LPS =  cz_RAS
"""

import json
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

from lung_nodule.pipeline.preprocess import dicom_to_nifti
from lung_nodule.classification.classifier import classify_nodule


# ---------------------------------------------------------------------------
# Detection helpers
# ---------------------------------------------------------------------------

def detect_nodules(nii_path: str, model_path: str, score_keep: float = 0.3):
    """Run the MONAI RetinaNet detector on a NIfTI volume.

    Parameters
    ----------
    nii_path : str
        Path to the .nii.gz CT volume.
    model_path : str
        Path to the TorchScript detector weights (dt_model.ts).
    score_keep : float
        Confidence score threshold; detections below this are discarded.

    Returns
    -------
    list of dict
        Each dict has keys: cx, cy, cz (world RAS coords, mm), w, h, d, score.
        Returns an empty list if no nodules pass the threshold.
    """
    import numpy as np
    from monai.data import Dataset, DataLoader
    from monai.data.utils import no_collation

    # Import detection building blocks from the local module
    from lung_nodule.detection.detector import (
        build_detector,
        build_preprocess,
        build_postprocess,
        device,
    )

    detector = build_detector(model_path=model_path, device=device)
    preprocess = build_preprocess()
    postprocess = build_postprocess()

    data = [{"image": nii_path}]
    ds = Dataset(data=data, transform=preprocess)
    dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=no_collation)

    detections = []
    for item in dl:
        item = item[0]
        image_4d = item["image"].to(device)

        if image_4d.dim() == 4:
            image_for_detector = image_4d.unsqueeze(0)
        else:
            image_for_detector = image_4d

        use_fp16 = device == "cuda"
        with torch.no_grad(), torch.amp.autocast(
            device_type="cuda" if device == "cuda" else "cpu",
            enabled=use_fp16,
            dtype=torch.float16 if use_fp16 else torch.float32,
        ):
            out = detector(image_for_detector, use_inferer=True)

        out0 = out[0]
        boxes  = out0.get("box",          out0.get("boxes"))
        labels = out0.get("label",        out0.get("labels"))
        scores = out0.get("label_scores", out0.get("scores"))

        boxes  = boxes.detach().cpu().numpy()  if torch.is_tensor(boxes)  else np.asarray(boxes)
        labels = labels.detach().cpu().numpy() if torch.is_tensor(labels) else np.asarray(labels)
        scores = scores.detach().cpu().numpy() if torch.is_tensor(scores) else np.asarray(scores)

        pred = {"box": boxes, "label": labels, "label_scores": scores}
        post_in = {**pred, "image": image_4d}
        post_out = postprocess(post_in)

        boxes_world = np.asarray(post_out["box"])      # [N, 6]: cccwhd in RAS world
        scores_world = np.asarray(post_out["label_scores"])

        if score_keep is not None and len(scores_world) > 0:
            keep = scores_world >= float(score_keep)
            boxes_world = boxes_world[keep]
            scores_world = scores_world[keep]

        for box, score in zip(boxes_world, scores_world):
            cx, cy, cz, w, h, d = box
            detections.append({
                "cx_ras": float(cx),
                "cy_ras": float(cy),
                "cz_ras": float(cz),
                "w": float(w),
                "h": float(h),
                "d": float(d),
                "score": float(score),
            })

    return detections


def ras_to_lps_xyz(cx_ras: float, cy_ras: float, cz_ras: float):
    """Convert RAS world coordinates to LPS (ITK) world coordinates."""
    return -cx_ras, -cy_ras, cz_ras


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    dicom_dir: str,
    output_dir: str = None,
    weights_dir: str = None,
    weights_dir_3d: str = None,
    model_path: str = None,
    model_type: str = "2d",
    score_keep: float = 0.3,
    threshold: float = 0.5,
):
    """Full end-to-end pipeline for one patient.

    Parameters
    ----------
    dicom_dir : str
        Path to a folder containing one patient's DICOM series.
    output_dir : str, optional
        Directory where intermediate .nii.gz and results JSON are saved.
        Uses a temporary directory if None.
    weights_dir : str, optional
        Path to 2D ResNet152 fold checkpoints (default: weights/ResNet152-confirmed/).
    weights_dir_3d : str, optional
        Path to 3D UNet fold checkpoints (default: weights/unet3D_encoder_scse/).
    model_path : str, optional
        Path to dt_model.ts (default: weights/dt_model.ts).
    model_type : str
        "2d", "3d", or "both" (default "3d").
    score_keep : float
        Minimum detection confidence score to keep (default 0.3).
    threshold : float
        Malignancy probability threshold (default 0.5).

    Returns
    -------
    dict
        Pipeline results including detected nodules and their classifications.
    """
    pipeline_root = Path(__file__).parent.parent.parent

    if model_path is None:
        model_path = str(pipeline_root / "weights" / "dt_model.ts")
    if weights_dir is None:
        weights_dir = str(pipeline_root / "weights" / "ResNet152-confirmed")
    if weights_dir_3d is None:
        weights_dir_3d = str(pipeline_root / "weights" / "unet3D_encoder_scse")

    t0 = time.time()

    # ---- Step 1: DICOM -> NIfTI ----
    use_tmp = output_dir is None
    tmp_dir = tempfile.mkdtemp() if use_tmp else None
    work_dir = Path(tmp_dir if use_tmp else output_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    patient_id = Path(dicom_dir).name
    nii_path = work_dir / f"{patient_id}.nii.gz"

    print(f"\n[Step 1] Converting DICOM -> NIfTI: {nii_path}")
    dicom_to_nifti(dicom_dir, str(nii_path))

    # ---- Step 2: Nodule detection ----
    print(f"\n[Step 2] Detecting nodules (score_keep={score_keep}) ...")
    detections = detect_nodules(str(nii_path), model_path=model_path, score_keep=score_keep)

    if not detections:
        print("[INFO] No nodules detected above the confidence threshold.")
        elapsed_ms = int((time.time() - t0) * 1000)
        result = {
            "seriesInstanceUID": patient_id,
            "probability": 0.0,
            "predictionLabel": 0,
            "processingTimeMs": elapsed_ms,
            "CoordX": None,
            "CoordY": None,
            "CoordZ": None,
        }
        _save_and_print(result, work_dir, patient_id)
        return result

    print(f"[INFO] Found {len(detections)} nodule(s).")

    # ---- Step 3: Classification ----
    print(f"\n[Step 3] Classifying nodules (model_type={model_type}) ...")
    nodule_results = []
    for i, det in enumerate(detections, start=1):
        x_lps, y_lps, z_lps = ras_to_lps_xyz(det["cx_ras"], det["cy_ras"], det["cz_ras"])

        print(f"  Nodule {i}/{len(detections)}: "
              f"RAS=({det['cx_ras']:.1f}, {det['cy_ras']:.1f}, {det['cz_ras']:.1f}) mm  "
              f"score={det['score']:.3f}")

        classification = classify_nodule(
            ct_path=str(nii_path),
            coord_world_xyz=[x_lps, y_lps, z_lps],
            weights_dir=weights_dir,
            weights_dir_3d=weights_dir_3d,
            model_type=model_type,
            threshold=threshold,
        )

        nodule_results.append({
            "nodule_id": i,
            "detection_score": det["score"],
            "coord_lps_mm": [x_lps, y_lps, z_lps],
            "malignancy_probability": classification["probability"],
            "label": classification["label"],
            "label_str": classification["label_str"],
        })

        print(f"    -> {classification['label_str']} (probability={classification['probability']:.4f})")

    # ---- Case-level aggregation ----
    # A case is malignant if any nodule is malignant.
    # Report the probability and coordinates of the most suspicious nodule.
    best = max(nodule_results, key=lambda n: n["malignancy_probability"])
    case_prob = best["malignancy_probability"]
    case_label = 1 if case_prob >= threshold else 0
    coord_x, coord_y, coord_z = best["coord_lps_mm"]

    elapsed_ms = int((time.time() - t0) * 1000)

    result = {
        "seriesInstanceUID": patient_id,
        "probability": round(case_prob, 6),
        "predictionLabel": case_label,
        "processingTimeMs": elapsed_ms,
        "CoordX": round(coord_x, 2),
        "CoordY": round(coord_y, 2),
        "CoordZ": round(coord_z, 2),
    }

    _save_and_print(result, work_dir, patient_id)
    return result


def _save_and_print(result: dict, work_dir: Path, patient_id: str):
    out_json = work_dir / f"{patient_id}_results.json"
    with open(out_json, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n[Done] Results saved to: {out_json}")

    label_str = "Malignant" if result["predictionLabel"] == 1 else "Benign"
    print(f"\n{'─'*50}")
    print(f"SUMMARY — {label_str}  (probability={result['probability']:.4f})")
    if result["CoordX"] is not None:
        print(f"Most suspicious nodule: CoordX={result['CoordX']}  CoordY={result['CoordY']}  CoordZ={result['CoordZ']}  (LPS mm)")
    print(f"Processing time: {result['processingTimeMs']} ms")
    print(f"{'─'*50}")
