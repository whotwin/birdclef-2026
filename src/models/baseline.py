"""Model definitions for BirdCLEF 2026.

Using a pre-trained ResNet as backbone + custom classification head.
Treats mel spectrograms as single-channel "images".
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class SpectrogramClassifier(nn.Module):
    """Audio classifier using pre-trained ResNet backbone.

    Takes mel spectrograms (1, n_mels, time) and outputs class probabilities.
    """

    def __init__(
        self,
        n_classes: int,
        backbone: str = "resnet34",
        pretrained: bool = True,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.n_classes = n_classes

        # Load pre-trained backbone
        if backbone == "resnet18":
            self.backbone = models.resnet18(weights="IMAGENET1K_V1" if pretrained else None)
            in_features = self.backbone.fc.in_features
        elif backbone == "resnet34":
            self.backbone = models.resnet34(weights="IMAGENET1K_V1" if pretrained else None)
            in_features = self.backbone.fc.in_features
        elif backbone == "resnet50":
            self.backbone = models.resnet50(weights="IMAGENET1K_V1" if pretrained else None)
            in_features = self.backbone.fc.in_features
        elif backbone == "efficientnet_b0":
            self.backbone = models.efficientnet_b0(weights="IMAGENET1K_V1" if pretrained else None)
            in_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()
        else:
            raise ValueError(f"Unknown backbone: {backbone}")

        self.backbone_name = backbone

        if "efficientnet" not in backbone:
            # Replace first conv layer to accept 1 channel
            old_conv = self.backbone.conv1
            self.backbone.conv1 = nn.Conv2d(
                1, old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=False
            )
            # Copy weights for the single channel
            with torch.no_grad():
                # Average across RGB channels for pretrained weights
                self.backbone.conv1.weight = nn.Parameter(
                    old_conv.weight.mean(dim=1, keepdim=True)
                )

            # Remove original FC layer
            self.backbone.fc = nn.Identity()

            self.fc = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(in_features, 512),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(512, n_classes),
            )
        else:
            # EfficientNet
            self.fc = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(in_features, 512),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(512, n_classes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, 1, n_mels, time) mel spectrograms
        Returns:
            (batch, n_classes) logits
        """
        if self.backbone_name == "efficientnet_b0":
            # EfficientNet expects (batch, 3, H, W) - repeat channel
            x = x.repeat(1, 3, 1, 1)
            features = self.backbone(x)
            return self.fc(features)
        else:
            # ResNet: (batch, 1, H, W) -> backbone handles it
            features = self.backbone(x)
            return self.fc(features)


class MultiLabelClassifier(nn.Module):
    """Wrapper for multi-label classification with BCE loss."""

    def __init__(
        self,
        n_classes: int,
        backbone: str = "resnet34",
        pretrained: bool = True,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.classifier = SpectrogramClassifier(
            n_classes=n_classes,
            backbone=backbone,
            pretrained=pretrained,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns raw logits (use BCEWithLogitsLoss for training)."""
        return self.classifier(x)


def get_model(n_classes: int, backbone: str = "resnet34", pretrained: bool = True) -> nn.Module:
    """Factory function to create a model."""
    return MultiLabelClassifier(
        n_classes=n_classes,
        backbone=backbone,
        pretrained=pretrained,
    )
