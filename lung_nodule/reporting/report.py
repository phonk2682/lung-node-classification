
from pathlib import Path
from datetime import datetime


def find_dicom_series(root: Path):
    """Return list of (patient_label, series_label, dicom_dir) tuples."""
    series = []
    for dcm_file in sorted(root.rglob("*.dcm")):
        d = dcm_file.parent
        # Walk up to find patient folder (2 levels above series for our structure)
        parts = d.relative_to(root).parts
        patient = parts[0] if len(parts) >= 1 else "unknown"
        series_name = parts[-1] if len(parts) >= 1 else d.name
        entry = (patient, series_name, d)
        if entry not in series:
            series.append(entry)
    return series


def _write_markdown(all_results, md_path: Path, args, total_elapsed: float):
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    n_series   = len(all_results)
    n_ok       = sum(1 for r in all_results if r["status"] == "ok")
    n_err      = n_series - n_ok
    all_nodules   = [n for r in all_results for n in r.get("nodules", [])]
    n_detected    = len(all_nodules)
    n_malignant   = sum(1 for n in all_nodules if n["label"] == 1)
    n_benign      = n_detected - n_malignant

    lines = []
    lines += [
        f"# Lung Nodule Detection & Classification Report",
        f"",
        f"**Generated:** {now}  ",
        f"**Dataset:** `{args.dataset_dir}`  ",
        f"**Detection threshold:** score >= {args.score_keep}  ",
        f"**Classification threshold:** probability >= {args.threshold}  ",
        f"**Model:** ResNet152 x 5-fold ensemble (2D)  ",
        f"**Detector:** MONAI RetinaNet (TorchScript)  ",
        f"",
        f"---",
        f"",
        f"## Summary",
        f"",
        f"| Metric | Value |",
        f"|---|---|",
        f"| Series processed | {n_ok} / {n_series} |",
        f"| Errors | {n_err} |",
        f"| Total nodules detected | {n_detected} |",
        f"| Malignant | {n_malignant} |",
        f"| Benign | {n_benign} |",
        f"| Total runtime | {total_elapsed/60:.1f} min |",
        f"",
        f"---",
        f"",
        f"## Results by Series",
        f"",
    ]

    for r in all_results:
        patient     = r.get("patient", "?")
        series_name = r.get("series_name", "?")
        status      = r.get("status", "?")
        elapsed     = r.get("elapsed_sec", 0)
        nodules     = r.get("nodules", [])

        lines += [
            f"### {patient} — `{series_name}`",
            f"",
            f"**Status:** {status}  ",
            f"**Processing time:** {elapsed}s  ",
        ]

        if status == "error":
            lines += [f"**Error:** {r.get('error', 'unknown')}  ", ""]
            continue

        if not nodules:
            lines += [f"**Result:** No nodules detected above score threshold.  ", ""]
            continue

        n_mal = sum(1 for n in nodules if n["label"] == 1)
        lines += [
            f"**Nodules found:** {len(nodules)}  ",
            f"**Malignant:** {n_mal}  **Benign:** {len(nodules) - n_mal}  ",
            f"",
            f"| # | Label | Probability | Det. Score | Center (RAS mm) | Box (mm) |",
            f"|---|---|---|---|---|---|",
        ]
        for n in nodules:
            c = n["center_ras_mm"]
            b = n["bounding_box_mm"]
            lines.append(
                f"| {n['nodule_id']} "
                f"| **{n['label_str']}** "
                f"| {n['malignancy_probability']:.4f} "
                f"| {n['detection_score']:.3f} "
                f"| ({c['x']:.1f}, {c['y']:.1f}, {c['z']:.1f}) "
                f"| {b['w']:.1f}x{b['h']:.1f}x{b['d']:.1f} |"
            )
        lines.append("")

    lines += [
        f"---",
        f"",
        f"## Methodology",
        f"",
        f"1. **Preprocessing** — Each DICOM series is converted to a 3-D NIfTI volume",
        f"   (`.nii.gz`) using SimpleITK, preserving spatial metadata.",
        f"",
        f"2. **Detection** — A 3-D MONAI RetinaNet sliding-window detector localises",
        f"   candidate nodules. Input is resampled to 0.703x0.703x1.25 mm.",
        f"   Detections with score < {args.score_keep} are discarded.",
        f"",
        f"3. **Classification** — For each detected nodule a 64x64 px axial patch",
        f"   (50x50 mm physical) is extracted and passed through a 5-fold ResNet152",
        f"   ensemble. Logits are averaged across folds before sigmoid activation.",
        f"   Nodules with probability >= {args.threshold} are labelled **Malignant**.",
        f"",
        f"4. **Coordinate conversion** — Detection outputs world (RAS) coordinates;",
        f"   classification uses ITK/LPS space: `x_LPS = -x_RAS`, `y_LPS = -y_RAS`,",
        f"   `z_LPS = z_RAS`.",
    ]

    md_path.write_text("\n".join(lines))
