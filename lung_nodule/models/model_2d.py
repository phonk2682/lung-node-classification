
import torch
import torch.nn as nn
import torchvision.models as models
import timm


# ==================== ResNet Family ====================
class ResNet18(nn.Module):
    def __init__(self, num_classes=1, weights='IMAGENET1K_V1'):
        super(ResNet18, self).__init__()
        self.resnet18 = models.resnet18(weights=weights)
        num_features = self.resnet18.fc.in_features
        self.resnet18.fc = nn.Sequential(
            nn.Linear(num_features, num_classes)
        )

    def forward(self, x):
        return self.resnet18(x)


class ResNet50(nn.Module):
    def __init__(self, num_classes=1, weights='IMAGENET1K_V2'):
        super(ResNet50, self).__init__()
        self.resnet50 = models.resnet50(weights=weights)
        num_features = self.resnet50.fc.in_features
        self.resnet50.fc = nn.Sequential(
            nn.Linear(num_features, num_classes)
        )

    def forward(self, x):
        return self.resnet50(x)


class ResNet101(nn.Module):
    def __init__(self, num_classes=1, weights='IMAGENET1K_V2'):
        super(ResNet101, self).__init__()
        self.resnet101 = models.resnet101(weights=weights)
        num_features = self.resnet101.fc.in_features
        self.resnet101.fc = nn.Sequential(
            nn.Linear(num_features, num_classes)
        )

    def forward(self, x):
        return self.resnet101(x)

class ResNet152(nn.Module):
    def __init__(self, num_classes=1, weights='IMAGENET1K_V2'):
        super(ResNet152, self).__init__()
        self.resnet152 = models.resnet152(weights=weights)
        num_features = self.resnet152.fc.in_features
        self.resnet152.fc = nn.Sequential(
            nn.Linear(num_features, num_classes)
        )

    def forward(self, x):
        return self.resnet152(x)

# ==================== EfficientNet Family ====================
class EfficientNetB3(nn.Module):
    def __init__(self, num_classes=1, pretrained=True):
        super(EfficientNetB3, self).__init__()
        # timm - efficientnet_b3.ra2_in1k (RA augmented, ImageNet-1k)
        self.efficientnet = timm.create_model(
            'efficientnet_b3.ra2_in1k',
            pretrained=pretrained,
            num_classes=num_classes
        )

    def forward(self, x):
        return self.efficientnet(x)


class EfficientNetB4(nn.Module):
    def __init__(self, num_classes=1, pretrained=True):
        super(EfficientNetB4, self).__init__()
        # timm - efficientnet_b4.ra2_in1k
        self.efficientnet = timm.create_model(
            'efficientnet_b4.ra2_in1k',
            pretrained=pretrained,
            num_classes=num_classes
        )

    def forward(self, x):
        return self.efficientnet(x)


class EfficientNetB5(nn.Module):
    def __init__(self, num_classes=1, pretrained=True):
        super(EfficientNetB5, self).__init__()
        # timm - efficientnet_b5.sw_in12k_ft_in1k (Swin pretrained on ImageNet-12k, finetuned on 1k)
        self.efficientnet = timm.create_model(
            'efficientnet_b5.sw_in12k_ft_in1k',
            pretrained=pretrained,
            num_classes=num_classes
        )

    def forward(self, x):
        return self.efficientnet(x)


# ==================== ConvNeXt Family ====================
class ConvNeXtTiny(nn.Module):
    def __init__(self, num_classes=1, pretrained=True):
        super(ConvNeXtTiny, self).__init__()
        # timm - convnext_tiny.in12k_ft_in1k (ImageNet-12k pretrained, finetuned on 1k)
        self.convnext = timm.create_model(
            'convnext_tiny.in12k_ft_in1k',
            pretrained=pretrained,
            num_classes=num_classes
        )

    def forward(self, x):
        return self.convnext(x)


class ConvNeXtSmall(nn.Module):
    def __init__(self, num_classes=1, pretrained=True):
        super(ConvNeXtSmall, self).__init__()
        # timm - convnext_small.in12k_ft_in1k
        self.convnext = timm.create_model(
            'convnext_small.in12k_ft_in1k',
            pretrained=pretrained,
            num_classes=num_classes
        )

    def forward(self, x):
        return self.convnext(x)


class ConvNeXtBase(nn.Module):
    def __init__(self, num_classes=1, pretrained=True):
        super(ConvNeXtBase, self).__init__()
        # timm - convnext_base.fb_in22k_ft_in1k (Facebook ImageNet-22k, finetuned on 1k)
        self.convnext = timm.create_model(
            'convnext_base.fb_in22k_ft_in1k',
            pretrained=pretrained,
            num_classes=num_classes
        )

    def forward(self, x):
        return self.convnext(x)


class ConvNeXtLarge(nn.Module):
    def __init__(self, num_classes=1, pretrained=True):
        super(ConvNeXtLarge, self).__init__()
        # timm - convnext_large.fb_in22k_ft_in1k
        self.convnext = timm.create_model(
            'convnext_large.fb_in22k_ft_in1k',
            pretrained=pretrained,
            num_classes=num_classes
        )

    def forward(self, x):
        return self.convnext(x)


# ==================== Swin Transformer Family ====================
class SwinTiny(nn.Module):
    def __init__(self, num_classes=1, pretrained=True):
        super(SwinTiny, self).__init__()
        # timm - swin_tiny_patch4_window7_224.ms_in22k_ft_in1k (Microsoft ImageNet-22k, finetuned on 1k)
        self.swin = timm.create_model(
            'swin_tiny_patch4_window7_224.ms_in22k_ft_in1k',
            pretrained=pretrained,
            num_classes=num_classes
        )

    def forward(self, x):
        return self.swin(x)


class SwinSmall(nn.Module):
    def __init__(self, num_classes=1, pretrained=True):
        super(SwinSmall, self).__init__()
        # timm - swin_small_patch4_window7_224.ms_in22k_ft_in1k
        self.swin = timm.create_model(
            'swin_small_patch4_window7_224.ms_in22k_ft_in1k',
            pretrained=pretrained,
            num_classes=num_classes
        )

    def forward(self, x):
        return self.swin(x)


class SwinBase(nn.Module):
    def __init__(self, num_classes=1, pretrained=True):
        super(SwinBase, self).__init__()
        # timm - swin_base_patch4_window7_224.ms_in22k_ft_in1k
        self.swin = timm.create_model(
            'swin_base_patch4_window7_224.ms_in22k_ft_in1k',
            pretrained=pretrained,
            num_classes=num_classes
        )

    def forward(self, x):
        return self.swin(x)


class SwinLarge(nn.Module):
    def __init__(self, num_classes=1, pretrained=True):
        super(SwinLarge, self).__init__()
        # timm - swin_large_patch4_window7_224.ms_in22k_ft_in1k
        self.swin = timm.create_model(
            'swin_large_patch4_window7_224.ms_in22k_ft_in1k',
            pretrained=pretrained,
            num_classes=num_classes
        )

    def forward(self, x):
        return self.swin(x)


# ==================== DenseNet Family ====================
class DenseNet121(nn.Module):
    def __init__(self, num_classes=1, weights='IMAGENET1K_V1'):
        super(DenseNet121, self).__init__()
        self.densenet = models.densenet121(weights=weights)
        num_features = self.densenet.classifier.in_features
        self.densenet.classifier = nn.Sequential(
            nn.Linear(num_features, num_classes)
        )

    def forward(self, x):
        return self.densenet(x)


class DenseNet169(nn.Module):
    def __init__(self, num_classes=1, weights='IMAGENET1K_V1'):
        super(DenseNet169, self).__init__()
        self.densenet = models.densenet169(weights=weights)
        num_features = self.densenet.classifier.in_features
        self.densenet.classifier = nn.Sequential(
            nn.Linear(num_features, num_classes)
        )

    def forward(self, x):
        return self.densenet(x)


# ==================== Vision Transformer ====================
class ViTBase(nn.Module):
    def __init__(self, num_classes=1, pretrained=True):
        super(ViTBase, self).__init__()
        # timm - vit_base_patch16_224.augreg_in21k_ft_in1k (AugReg trained on ImageNet-21k, finetuned on 1k)
        self.vit = timm.create_model(
            'vit_base_patch16_224.augreg_in21k_ft_in1k',
            pretrained=pretrained,
            num_classes=num_classes
        )

    def forward(self, x):
        return self.vit(x)


class ViTLarge(nn.Module):
    def __init__(self, num_classes=1, pretrained=True):
        super(ViTLarge, self).__init__()
        # timm - vit_large_patch16_224.augreg_in21k_ft_in1k
        self.vit = timm.create_model(
            'vit_large_patch16_224.augreg_in21k_ft_in1k',
            pretrained=pretrained,
            num_classes=num_classes
        )

    def forward(self, x):
        return self.vit(x)
