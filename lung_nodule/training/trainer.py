
import logging
import numpy as np
import torch
import sklearn.metrics as metrics
from tqdm import tqdm
import random
import pandas as pd
from datetime import datetime
from pathlib import Path
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from lung_nodule.config import config
from lung_nodule import data as dataloader
from lung_nodule.models.model_3d import I3D
from lung_nodule.training.losses import FocalLoss, FocalLossWithSmoothing
from lung_nodule.training.splits import create_kfold_splits, make_weights_for_balanced_classes
from lung_nodule.models.registry import MODEL_REGISTRY, MODEL_LR_CONFIG, get_model_and_optimizer

torch.backends.cudnn.benchmark = True

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s][%(asctime)s] %(message)s",
    datefmt="%H:%M:%S",
)


# ==================== TRAINING FUNCTION ====================
def train_fold(train_df, valid_df, exp_save_root, fold_idx):
    fold_dir = exp_save_root / f"fold_{fold_idx}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'='*50}Fold {fold_idx} | Train: {len(train_df)} | Val: {len(valid_df)}{'='*50}")

    # Data loaders
    sampler = torch.utils.data.WeightedRandomSampler(
        weights=make_weights_for_balanced_classes(train_df.label.values),
        num_samples=len(train_df),
        replacement=True
    )

    common_args = dict(
        workers=config.NUM_WORKERS,
        batch_size=config.BATCH_SIZE,
        size_mm=config.SIZE_MM,
        size_px=config.SIZE_PX,
    )

    train_loader = dataloader.get_data_loader(
        config.DATADIR, train_df, mode=config.MODE, sampler=sampler,
        rotations=config.ROTATION, translations=config.TRANSLATION,
        use_monai_transforms="train", config=config, **common_args
    )

    valid_loader = dataloader.get_data_loader(
        config.DATADIR, valid_df, mode=config.MODE,
        rotations=None, translations=None,
        use_monai_transforms="val", **common_args
    )

    # Model setup
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if config.MODE == "2D":
        model, optimizer = get_model_and_optimizer(config.MODEL_NAME, device, config)
    else:  # 3D
        model = I3D(num_classes=1, input_channels=3, pre_trained=True, freeze_bn=True).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)

    # criterion = FocalLoss(alpha=0.9, gamma=2)
    criterion = FocalLossWithSmoothing(alpha=0.9, gamma=3, smoothing=0.1)
    # scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-7)

    # criterion = torch.nn.BCEWithLogitsLoss()

    # Training loop
    best_metric, best_epoch = -1, -1
    epochs_no_improve = 0

    for epoch in range(config.EPOCHS):
        if epochs_no_improve > config.PATIENCE:
            print(f"Early stop - no improvement for {config.PATIENCE} epochs")
            break

        # Train
        model.train()
        train_loss = 0
        for step, batch in enumerate(tqdm(train_loader, desc=f"Fold {fold_idx} Epoch {epoch+1} Train", leave=False), 1):
            inputs, labels = batch["image"].to(device), batch["label"].float().to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels.squeeze())
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / step
        # scheduler.step()

        # Validate
        model.eval()
        val_loss = 0
        y_true, y_pred = [], []

        with torch.no_grad():
            for step, batch in enumerate(tqdm(valid_loader, desc=f"Fold {fold_idx} Epoch {epoch+1} Val", leave=False), 1):
                inputs, labels = batch["image"].to(device), batch["label"].float().to(device)

                outputs = model(inputs)
                loss = criterion(outputs.squeeze(), labels.squeeze())

                val_loss += loss.item()
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(torch.sigmoid(outputs).cpu().numpy())

        avg_val_loss = val_loss / step
        y_true = np.array(y_true).reshape(-1)
        y_pred = np.array(y_pred).reshape(-1)

        # Metrics
        fpr, tpr, _ = metrics.roc_curve(y_true, y_pred)
        auc = metrics.auc(fpr, tpr)
        ap = metrics.average_precision_score(y_true, y_pred)

        y_pred_binary = (y_pred > 0.5).astype(int)
        acc = metrics.accuracy_score(y_true, y_pred_binary)
        prec = metrics.precision_score(y_true, y_pred_binary, zero_division=0)
        rec = metrics.recall_score(y_true, y_pred_binary, zero_division=0)
        f1 = metrics.f1_score(y_true, y_pred_binary, zero_division=0)

        print(f"Fold {fold_idx} | Epoch {epoch+1}/{config.EPOCHS} | "
                    f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | "
                    f"AUC: {auc:.4f} | AP: {ap:.4f} | Acc: {acc:.4f} | F1: {f1:.4f}")

        # Save best model
        if f1 > best_metric:
            best_metric, best_epoch = f1, epoch + 1
            epochs_no_improve = 0

            torch.save(model.state_dict(), fold_dir / "best_metric_model.pth")
            np.save(fold_dir / "config.npy", {
                "fold": fold_idx, "config": config, "best_ap": best_metric,
                "best_auc": auc, "best_acc": acc, "best_prec": prec,
                "best_rec": rec, "best_f1": f1, "epoch": best_epoch
            })
            print(f"✓ New best model saved (F1: {best_metric:.4f})")
        else:
            epochs_no_improve += 1

        print(
                "current epoch: {} current F1: {:.4f} best F1: {:.4f} at epoch {}".format(
                    epoch + 1, f1, best_metric, best_epoch
                )
            )
    print(f"Fold {fold_idx} complete - Best F1: {best_metric:.4f} at epoch {best_epoch}")

    return {
        'auc': auc, 'ap': best_metric, 'accuracy': acc,
        'precision': prec, 'recall': rec, 'f1': f1
    }

# ==================== CROSS VALIDATION ====================
def train_cross_validation(csv_path, exp_save_root, n_folds=5):
    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)
    random.seed(config.SEED)

    exp_save_root.mkdir(parents=True, exist_ok=True)

    # Setup file logging
    log_file = exp_save_root / "cross_validation.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter("[%(levelname)s][%(asctime)s] %(message)s"))
    logging.getLogger().addHandler(file_handler)

    print(f"Starting {n_folds}-fold cross-validation")
    print(f"Data CSV: {csv_path}")

    # Create folds
    fold_data = create_kfold_splits(csv_path, n_splits=n_folds, random_state=config.SEED)

    # Train each fold
    fold_metrics = []
    for fold_idx, (train_df, valid_df) in enumerate(fold_data):
        fold_result = train_fold(train_df, valid_df, exp_save_root, fold_idx)
        fold_metrics.append(fold_result)

    # Summarize results
    metrics_names = ['auc', 'ap', 'accuracy', 'precision', 'recall', 'f1']
    mean_metrics = {m: np.mean([f[m] for f in fold_metrics]) for m in metrics_names}
    std_metrics = {m: np.std([f[m] for f in fold_metrics]) for m in metrics_names}

    print("="*60)
    print("CROSS-VALIDATION RESULTS")
    print("="*60)
    for fold_idx, metrics_dict in enumerate(fold_metrics):
        print(f"Fold {fold_idx}: AUC={metrics_dict['auc']:.4f} | AP={metrics_dict['ap']:.4f} | "
                    f"Acc={metrics_dict['accuracy']:.4f} | F1={metrics_dict['f1']:.4f}")

    print("-"*60)
    for m in metrics_names:
        print(f"Mean {m.upper()}: {mean_metrics[m]:.4f} +/- {std_metrics[m]:.4f}")

    # Save results
    np.save(exp_save_root / "cv_results.npy", {
        "fold_metrics": fold_metrics,
        "mean_metrics": mean_metrics,
        "std_metrics": std_metrics,
        "config": config,
    })

    return mean_metrics, std_metrics
