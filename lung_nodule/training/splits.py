
import logging
import numpy as np
import torch
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from lung_nodule.config import config


def make_weights_for_balanced_classes(labels):
    n = len(labels)
    unique, counts = np.unique(labels, return_counts=True)
    frac = {cls: n / (2 * c) for cls, c in zip(unique, counts)}
    return torch.DoubleTensor([frac[l] for l in labels])

def create_kfold_splits(csv_path, n_splits=5, random_state=None):
    df = pd.read_csv(csv_path)

    # Stratified + Grouped K-Fold
    sgkf = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state
    )

    fold_dfs = []

    X = df.index
    y = df['label']          # label de stratify
    groups = df['PatientID'] # group de tranh leakage

    for fold_idx, (train_idx, val_idx) in enumerate(sgkf.split(X, y, groups)):
        train_fold = df.iloc[train_idx].reset_index(drop=True)
        val_fold = df.iloc[val_idx].reset_index(drop=True)

        train_patients = set(train_fold.PatientID)
        val_patients = set(val_fold.PatientID)

        # Check leakage
        common = train_patients.intersection(val_patients)
        if common:
            logging.warning(f"[Fold {fold_idx}] Patient leakage detected: {len(common)} patients!")

        # Print thong tin fold
        print(f"=== Fold {fold_idx} ===")
        print(f"Train: {len(train_fold)} samples from {len(train_patients)} patients")
        print(f"  Label ratio: {train_fold['label'].value_counts(normalize=True).to_dict()}")
        print(f"Val:   {len(val_fold)} samples from {len(val_patients)} patients")
        print(f"  Label ratio: {val_fold['label'].value_counts(normalize=True).to_dict()}")

        fold_dfs.append((train_fold, val_fold))

    return fold_dfs
