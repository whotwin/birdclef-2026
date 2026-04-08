"""
SimCLR Encoder: backbone + projection head.
After contrastive pretraining, use the backbone for downstream tasks.
"""
import torch
import torch.nn as nn
import timm


class ProjectionHead(nn.Module):
    """MLP projection head for contrastive learning."""

    def __init__(self, in_features, hidden_dim=512, out_features=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_features),
        )

    def forward(self, x):
        return self.net(x)


class SimCLREncoder(nn.Module):
    """
    Encoder for SimCLR contrastive learning.

    Architecture:
      backbone (timm) → GlobalAvgPool → h (features)
      h → projection_head → z (projected, normalized)

    forward_return_h=True: returns features h (for linear eval)
    forward_return_h=False: returns normalized projections z (for contrastive loss)
    """

    def __init__(self, backbone_name='efficientnet_b0', projection_dim=128, pretrained=True):
        super().__init__()
        # Backbone: single-channel mel spectrogram input
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            in_chans=1,
            num_classes=0,
            global_pool='',
        )
        self.feature_dim = self.backbone.num_features
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # Projection head
        self.projection_head = ProjectionHead(self.feature_dim, projection_dim * 4, projection_dim)

    def forward(self, x, return_h=False):
        """
        Args:
            x: [B, 1, 128, T] mel spectrogram
            return_h: if True, return features before projection head
        Returns:
            z: [B, projection_dim] normalized projections
            (or h: [B, feature_dim] if return_h=True)
        """
        h = self.backbone(x)
        h = self.global_pool(h)
        h = h.flatten(1)  # [B, feature_dim]

        if return_h:
            return h

        z = self.projection_head(h)  # [B, projection_dim]
        z = nn.functional.normalize(z, dim=1)  # L2 normalize
        return z


class LinearEvalHead(nn.Module):
    """Linear evaluation head: freeze backbone, train only this."""

    def __init__(self, encoder, num_classes=234, freeze_backbone=True):
        super().__init__()
        self.encoder = encoder
        if freeze_backbone:
            for param in self.encoder.parameters():
                param.requires_grad = False
        self.linear_head = nn.Linear(encoder.feature_dim, num_classes)

    def forward(self, x):
        h = self.encoder(x, return_h=True)
        return self.linear_head(h)
