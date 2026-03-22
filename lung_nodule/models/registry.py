
import numpy as np
import torch

from lung_nodule.models.model_2d import (
    ResNet18, ResNet50, ResNet101, ResNet152,
    EfficientNetB3, EfficientNetB4, EfficientNetB5,
    ConvNeXtTiny, ConvNeXtSmall, ConvNeXtBase, ConvNeXtLarge,
    DenseNet121, DenseNet169,
    ViTBase, ViTLarge,
)
from lung_nodule.models.model_3d import I3D


# ==================== MODEL REGISTRY ====================
MODEL_REGISTRY = {
    "ResNet18": ResNet18, "ResNet50": ResNet50, "ResNet101": ResNet101, "ResNet152": ResNet152,
    "EfficientNetB3": EfficientNetB3, "EfficientNetB4": EfficientNetB4, "EfficientNetB5": EfficientNetB5,
    "ConvNeXtTiny": ConvNeXtTiny, "ConvNeXtSmall": ConvNeXtSmall, "ConvNeXtBase": ConvNeXtBase, "ConvNeXtLarge": ConvNeXtLarge,
    "DenseNet121": DenseNet121, "DenseNet169": DenseNet169,
    "ViTBase": ViTBase, "ViTLarge": ViTLarge,
}

MODEL_LR_CONFIG = {
    "ResNet18": {"lr": 1e-4, "optimizer": "Adam"},
    "ResNet50": {"lr": 1e-4, "optimizer": "Adam"},
    "ResNet101": {"lr": 5e-5, "optimizer": "Adam"},
    "ResNet152": {"lr": 5e-5, "optimizer": "Adam"},
    "EfficientNetB3": {"lr": 5e-5, "optimizer": "AdamW"},
    "EfficientNetB4": {"lr": 5e-5, "optimizer": "AdamW"},
    "EfficientNetB5": {"lr": 3e-5, "optimizer": "AdamW"},
    "ConvNeXtTiny": {"lr": 3e-5, "optimizer": "AdamW"},
    "ConvNeXtSmall": {"lr": 1e-5, "optimizer": "AdamW"},
    "ConvNeXtBase": {"lr": 1e-5, "optimizer": "AdamW"},
    "ConvNeXtLarge": {"lr": 5e-6, "optimizer": "AdamW"},
    "DenseNet121": {"lr": 1e-4, "optimizer": "Adam"},
    "DenseNet169": {"lr": 5e-5, "optimizer": "Adam"},
    "ViTBase": {"lr": 5e-6, "optimizer": "AdamW"},
    "ViTLarge": {"lr": 3e-6, "optimizer": "AdamW"},
}


# ==================== UTILITY FUNCTIONS ====================
def get_model_and_optimizer(model_name, device, config):
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Model {model_name} not found. Available: {list(MODEL_REGISTRY.keys())}")

    model = MODEL_REGISTRY[model_name]().to(device)
    lr_config = MODEL_LR_CONFIG.get(model_name, {"lr": config.LEARNING_RATE, "optimizer": "Adam"})

    if lr_config["optimizer"] == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr_config["lr"], weight_decay=0.01)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr_config["lr"], weight_decay=config.WEIGHT_DECAY)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {model_name} | Optimizer: {lr_config['optimizer']} | LR: {lr_config['lr']:.2e} | Params: {total_params:,}")

    return model, optimizer
