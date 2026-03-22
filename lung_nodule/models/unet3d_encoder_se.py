# models/unet3d_encoder_se.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional

class DoubleConv3d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, mid_ch: Optional[int]=None, use_bn: bool=True, dropout: float=0.0):
        super().__init__()
        if mid_ch is None:
            mid_ch = out_ch
        layers = [
            nn.Conv3d(in_ch, mid_ch, kernel_size=3, padding=1, bias=not use_bn)
        ]
        if use_bn:
            layers.append(nn.BatchNorm3d(mid_ch))
        layers.append(nn.ReLU(inplace=True))

        layers.append(nn.Conv3d(mid_ch, out_ch, kernel_size=3, padding=1, bias=not use_bn))
        if use_bn:
            layers.append(nn.BatchNorm3d(out_ch))
        layers.append(nn.ReLU(inplace=True))

        if dropout and dropout > 0:
            layers.append(nn.Dropout3d(dropout))

        self.double_conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.double_conv(x)


class Down3d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, use_bn: bool=True, dropout: float=0.0):
        super().__init__()
        self.pool_conv = nn.Sequential(
            nn.MaxPool3d(kernel_size=2, stride=2),
            DoubleConv3d(in_ch, out_ch, use_bn=use_bn, dropout=dropout)
        )

    def forward(self, x):
        return self.pool_conv(x)


# ----------------------------
# Squeeze-and-Excitation block for 3D
# ----------------------------
class SEBlock3D(nn.Module):
    """Squeeze-and-Excitation block for 3D feature maps."""
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool3d(1)  # -> (B, C, 1,1,1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, D, H, W)
        b, c, _, _, _ = x.shape
        y = self.pool(x).view(b, c)         # (B, C)
        y = self.fc(y).view(b, c, 1, 1, 1)  # (B, C, 1,1,1)
        return x * y.expand_as(x)


# ----------------------------
# scSE block for 3D (channel + spatial)
# ----------------------------
class scSEBlock3D(nn.Module):
    """
    scSE: combination of cSE (channel) and sSE (spatial).
    Output = x * cSE + x * sSE  (sum of both recalibrated maps)
    """
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        # channel SE
        self.cse_pool = nn.AdaptiveAvgPool3d(1)
        self.cse_fc = nn.Sequential(
            nn.Linear(channels, max(1, channels // reduction), bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(max(1, channels // reduction), channels, bias=False),
            nn.Sigmoid()
        )
        # spatial SE: 1x1x1 conv -> sigmoid producing single-channel map
        self.sse_conv = nn.Conv3d(channels, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, d, h, w = x.shape
        # cSE
        y = self.cse_pool(x).view(b, c)           # (B, C)
        y = self.cse_fc(y).view(b, c, 1, 1, 1)    # (B, C, 1,1,1)
        x_cse = x * y

        # sSE
        s = self.sse_conv(x)                      # (B,1,D,H,W)
        s = self.sigmoid(s)
        x_sse = x * s
        return x_cse + x_sse



# ----------------------------
# UNet3D encoder + SE + classifier head
# ----------------------------
class UNet3DEncoderClassifier(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 1,
        features: List[int] = [32, 64, 128, 256],
        use_bn: bool = True,
        dropout: float = 0.0,
        se_reduction: int = 16,
        mlp_hidden: int = 256,
        mlp_dropout: float = 0.5
    ):
        super().__init__()

        # Encoder (same as UNet encoder)
        self.inc = DoubleConv3d(in_channels, features[0], use_bn=use_bn)
        self.down1 = Down3d(features[0], features[1], use_bn=use_bn)
        self.down2 = Down3d(features[1], features[2], use_bn=use_bn)
        self.down3 = Down3d(features[2], features[3], use_bn=use_bn)

        # bottom layer (further downsample)
        bottom_ch = features[3] * 2
        self.down4 = Down3d(features[3], bottom_ch, use_bn=use_bn, dropout=dropout)

        # SE block applied to bottleneck features
        self.se = SEBlock3D(channels=bottom_ch, reduction=se_reduction)

        #scSE block applied to bottleneck features
        self.scse1 = scSEBlock3D(features[0], reduction=se_reduction)
        self.scse2 = scSEBlock3D(features[1], reduction=se_reduction)
        self.scse3 = scSEBlock3D(features[2], reduction=se_reduction)
        self.scse4 = scSEBlock3D(features[3], reduction=se_reduction)
        self.scse_bottom = scSEBlock3D(bottom_ch, reduction=se_reduction)

        # classifier MLP head
        self.pool = nn.AdaptiveAvgPool3d(1)  # (B, C, 1,1,1) -> flatten -> (B, C)
        self.mlp = nn.Sequential(
            nn.Linear(bottom_ch, mlp_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(mlp_dropout),
            nn.Linear(mlp_hidden, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)           # level0: (B, f0, ...)
        # x1 = self.scse1(x1)
        x2 = self.down1(x1)        # level1: (B, f1, ...)
        # x2 = self.scse2(x2)
        x3 = self.down2(x2)        # level2: (B, f2, ...)
        # x3 = self.scse3(x3)
        x4 = self.down3(x3)        # level3: (B, f3, ...)
        # x4 = self.scse4(x4)
        x5 = self.down4(x4)       # bottleneck (B, bottom_ch, D', H', W')
        # x5 = self.se(x5)         # apply SE attention
        x5 = self.scse_bottom(x5)   # apply scSE attention


        pooled = self.pool(x5)   # (B, bottom_ch, 1,1,1)
        vec = pooled.view(pooled.size(0), -1)  # (B, bottom_ch)
        logits = self.mlp(vec)   # (B, num_classes)
        if logits.shape[1] == 1:
            return logits.view(-1)
        return logits
