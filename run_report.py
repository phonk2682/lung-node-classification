"""
run_report.py — Batch-process all DICOM series under a root directory and
generate a Markdown + JSON report.

Usage:
    python run_report.py --dataset_dir /path/to/MTN/extracted/ \
                         --output_dir  ./report/
"""

import argparse
import json
import sys
import time
from pathlib import Path

from lung_nodule.reporting.report import find_dicom_series, _write_markdown
from lung_nodule.pipeline.pipeline import run_pipeline


def main():
    parser = argparse.ArgumentParser(description="Batch pipeline + report generator")
    parser.add_argument("--dataset_dir", required=True,
                        help="Root directory containing patient DICOM folders")
    parser.add_argument("--output_dir",  default="./report",
                        help="Directory for NIfTI intermediates, JSON, and Markdown report")
    parser.add_argument("--score_keep",  type=float, default=0.3,
                        help="Detection confidence threshold (default 0.3)")
    parser.add_argument("--threshold",   type=float, default=0.5,
                        help="Malignancy probability threshold (default 0.5)")
    parser.add_argument("--model_type",  default="3d", choices=["2d", "3d", "both"],
                        help="Classification model: 2d, 3d, or both (default: 3d)")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    output_dir  = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_series = find_dicom_series(dataset_dir)
    if not all_series:
        print(f"[ERROR] No DICOM files found under: {dataset_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(all_series)} series across dataset.")
    print("="*60)

    all_results = []
    total_nodules = 0
    total_malignant = 0
    start_global = time.time()

    for idx, (patient, series_name, dicom_dir) in enumerate(all_series, 1):
        print(f"\n[{idx}/{len(all_series)}] Patient: {patient}")
        print(f"           Series:  {series_name}")
        t0 = time.time()
        try:
            result = run_pipeline(
                dicom_dir=str(dicom_dir),
                output_dir=str(output_dir / "nifti"),
                score_keep=args.score_keep,
                threshold=args.threshold,
                model_type=args.model_type,
            )
            elapsed = time.time() - t0
            result["patient"]     = patient
            result["series_name"] = series_name
            result["elapsed_sec"] = round(elapsed, 1)
            result["status"]      = "ok"

            n_nodules   = len(result.get("nodules", []))
            n_malignant = sum(1 for n in result.get("nodules", []) if n["label"] == 1)
            total_nodules   += n_nodules
            total_malignant += n_malignant

            print(f"  -> {n_nodules} nodule(s) detected, {n_malignant} malignant "
                  f"[{elapsed:.1f}s]")

        except Exception as e:
            elapsed = time.time() - t0
            print(f"  [ERROR] {e}")
            result = {
                "patient":     patient,
                "series_name": series_name,
                "status":      "error",
                "error":       str(e),
                "elapsed_sec": round(elapsed, 1),
                "nodules":     [],
            }

        all_results.append(result)

    total_elapsed = time.time() - start_global

    # ── save raw JSON ─────────────────────────────────────────────────────────
    json_path = output_dir / "results.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nRaw results saved to: {json_path}")

    # ── build Markdown report ─────────────────────────────────────────────────
    md_path = output_dir / "report.md"
    _write_markdown(all_results, md_path, args, total_elapsed)
    print(f"Report saved to:      {md_path}")


if __name__ == "__main__":
    main()
