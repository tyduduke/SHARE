#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class SLAMErrorPredictor(nn.Module):
    """
    Lightweight SLAM error predictor (paper §4.2.1).

    Input
    -----
    image          : (B, 3, 224, 224) — normalised RGB keyframe
    psd            : (B, 3, H, W)     — Point Spatial Distribution matrix
    scalar_features: (B, 8)           — temporal/geometric scalar features

    Output
    ------
    (B, 1)  — predicted SLAM tracking error (cm)
    """

    def __init__(self, num_scalar_features: int = 8):
        super().__init__()

        # ----------------------------------------------------------
        # Visual branch: MobileNetV2 backbone → 64-dim
        # ----------------------------------------------------------
        backbone = models.mobilenet_v2(
            weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        # Remove the classifier head; keep feature layers only
        self.image_backbone = nn.Sequential(*list(backbone.children())[:-1])
        # Paper: "1 fully connected layer and layer normalization → 64-dim"
        self.image_fc = nn.Linear(1280, 64)
        # BatchNorm was tried here first; it performed comparably on the training
        # set but degraded at inference time on single-sample batches (B=1),
        # where batch statistics are undefined. LayerNorm normalises per-sample
        # and is batch-size agnostic, which is required for live keyframe inference.
        # self.image_bn = nn.BatchNorm1d(64)
        self.image_ln = nn.LayerNorm(64)

        # ----------------------------------------------------------
        # PSD branch: two Conv layers → 16-dim
        # Paper: "8 filters with 3×3 kernels and stride 2,
        #         followed by 16 filters with identical parameters"
        # ----------------------------------------------------------
        self.psd_conv1 = nn.Conv2d(3, 8,  kernel_size=3, stride=2, padding=1)
        self.psd_conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1)

        # ----------------------------------------------------------
        # Scalar branch: FC → 16-dim
        # Paper: "8-element input vector through a fully connected layer
        #         with ReLU activation to produce 16-dimensional representations"
        # ----------------------------------------------------------
        self.scalar_fc = nn.Linear(num_scalar_features, 16)

        # ----------------------------------------------------------
        # Fusion: [64 + 16 + 16 = 96] → 64 → 1
        # Paper: "two-layer fully connected network with ReLU activation"
        # ----------------------------------------------------------
        self.fusion_fc = nn.Sequential(
            nn.Linear(64 + 16 + 16, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, image: torch.Tensor,
                psd: torch.Tensor,
                scalar_features: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        image           : (B, 3, 224, 224)
        psd             : (B, 3, H, W)
        scalar_features : (B, 8)

        Returns
        -------
        (B, 1)
        """
        # Visual branch
        img_feat = self.image_backbone(image)          # (B, 1280, 7, 7)
        img_feat = img_feat.mean(dim=[2, 3])           # Global Average Pooling → (B, 1280)
        img_feat = self.image_ln(F.relu(self.image_fc(img_feat)))  # → (B, 64)

        # PSD branch
        psd_feat = F.relu(self.psd_conv1(psd))         # (B, 8, H/2, W/2)
        psd_feat = F.relu(self.psd_conv2(psd_feat))    # (B, 16, H/4, W/4)
        psd_feat = F.adaptive_avg_pool2d(psd_feat, (1, 1)).view(psd_feat.size(0), -1)  # (B, 16)

        # Scalar branch
        scalar_feat = F.relu(self.scalar_fc(scalar_features))      # (B, 16)

        # Fusion
        fused = torch.cat([img_feat, psd_feat, scalar_feat], dim=1)  # (B, 96)
        out   = self.fusion_fc(fused)                                # (B, 1)
        return out
