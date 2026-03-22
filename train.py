
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

from lung_nodule.config import config
from lung_nodule.models.registry import MODEL_REGISTRY
from lung_nodule.training.trainer import train_fold


def main():
    parser = argparse.ArgumentParser(
        description="Train lung nodule malignancy classifier (5-fold cross-validation)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--image_dir", required=True,
        help=(
            "Root data directory. Must contain two subdirectories:\n"
            "  image/    — one .npy file per nodule named <AnnotationID>.npy\n"
            "  metadata/ — one .npy file per nodule named <AnnotationID>.npy"
        ),
    )
    parser.add_argument(
        "--csv_dir", required=True,
        help=(
            "Directory containing the 5-fold CSV split files:\n"
            "  data_fold_1_train.csv, data_fold_1_test.csv, ..., data_fold_5_test.csv\n"
            "Each CSV must have columns: AnnotationID, label, PatientID"
        ),
    )
    parser.add_argument(
        "--output_dir", default="./checkpoints",
        help="Root directory for saving model checkpoints and metrics",
    )
    parser.add_argument(
        "--model", default="ResNet152",
        choices=list(MODEL_REGISTRY.keys()),
        help="Model architecture to train",
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Override max epochs from config",
    )
    parser.add_argument(
        "--batch_size", type=int, default=None,
        help="Override batch size from config",
    )
    parser.add_argument(
        "--folds", type=int, default=5,
        help="Number of folds to train (1-5)",
    )
    args = parser.parse_args()

    # Apply CLI overrides to config
    config.DATADIR   = Path(args.image_dir)
    config.MODEL_NAME = args.model
    if args.epochs is not None:
        config.EPOCHS = args.epochs
    if args.batch_size is not None:
        config.BATCH_SIZE = args.batch_size

    csv_dir  = Path(args.csv_dir)
    exp_root = Path(args.output_dir) / f"{args.model}-{config.MODE}-{datetime.today().strftime('%Y%m%d')}-CV"
    exp_root.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("TRAINING CONFIGURATION")
    print("=" * 60)
    print(f"  Model      : {args.model}")
    print(f"  Mode       : {config.MODE}")
    print(f"  Image dir  : {config.DATADIR}")
    print(f"  CSV dir    : {csv_dir}")
    print(f"  Output dir : {exp_root}")
    print(f"  Epochs     : {config.EPOCHS}  (patience={config.PATIENCE})")
    print(f"  Batch size : {config.BATCH_SIZE}")
    print(f"  Folds      : {args.folds}")
    print("=" * 60)

    fold_metrics = []

    for i in range(1, args.folds + 1):
        train_path = csv_dir / f"data_fold_{i}_train.csv"
        val_path   = csv_dir / f"data_fold_{i}_test.csv"
        if not train_path.exists():
            raise FileNotFoundError(f"Training CSV not found: {train_path}")
        if not val_path.exists():
            raise FileNotFoundError(f"Validation CSV not found: {val_path}")
        train_df = pd.read_csv(train_path)
        valid_df = pd.read_csv(val_path)
        print(f"\nFold {i}: {len(train_df)} train / {len(valid_df)} val samples")
        metric = train_fold(train_df, valid_df, exp_save_root=exp_root, fold_idx=i)
        fold_metrics.append(metric)

    metrics_names = ['auc', 'ap', 'accuracy', 'precision', 'recall', 'f1']
    mean_metrics = {m: np.mean([f[m] for f in fold_metrics]) for m in metrics_names}
    std_metrics  = {m: np.std( [f[m] for f in fold_metrics]) for m in metrics_names}

    print("\n" + "=" * 60)
    print("CROSS-VALIDATION RESULTS")
    print("=" * 60)
    for fold_idx, md in enumerate(fold_metrics, 1):
        print(f"  Fold {fold_idx}: AUC={md['auc']:.4f}  AP={md['ap']:.4f}  "
              f"Acc={md['accuracy']:.4f}  F1={md['f1']:.4f}")
    print("-" * 60)
    for m in metrics_names:
        print(f"  Mean {m.upper():12s}: {mean_metrics[m]:.4f} +/- {std_metrics[m]:.4f}")
    print(f"\nCheckpoints saved to: {exp_root}/fold_<N>/best_metric_model.pth")


if __name__ == "__main__":
    main()
