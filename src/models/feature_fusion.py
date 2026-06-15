"""
feature_fusion.py

Step 2B

Multi-Scale Feature Fusion
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScaleFeatureFusion(nn.Module):

    def __init__(
        self,
        output_dim=768
    ):
        super().__init__()

        self.stage1_proj = nn.Conv2d(
            96,
            output_dim,
            kernel_size=1
        )

        self.stage2_proj = nn.Conv2d(
            192,
            output_dim,
            kernel_size=1
        )

        self.stage3_proj = nn.Conv2d(
            384,
            output_dim,
            kernel_size=1
        )

        self.stage4_proj = nn.Conv2d(
            768,
            output_dim,
            kernel_size=1
        )

        self.global_pool = (
            nn.AdaptiveAvgPool2d(1)
        )

    def forward(self, features):

        s1 = features["stage1"]
        s2 = features["stage2"]
        s3 = features["stage3"]
        s4 = features["stage4"]

        # Swin outputs:
        # [B,H,W,C]
        # convert to:
        # [B,C,H,W]

        s1 = s1.permute(0,3,1,2)
        s2 = s2.permute(0,3,1,2)
        s3 = s3.permute(0,3,1,2)
        s4 = s4.permute(0,3,1,2)

        target_size = s4.shape[-2:]

        s1 = F.interpolate(
            self.stage1_proj(s1),
            size=target_size,
            mode="bilinear",
            align_corners=False
        )

        s2 = F.interpolate(
            self.stage2_proj(s2),
            size=target_size,
            mode="bilinear",
            align_corners=False
        )

        s3 = F.interpolate(
            self.stage3_proj(s3),
            size=target_size,
            mode="bilinear",
            align_corners=False
        )

        s4 = self.stage4_proj(s4)

        fused = (
            s1 +
            s2 +
            s3 +
            s4
        )

        image_vector = (
            self.global_pool(fused)
            .flatten(1)
        )

        return {
            "fused_feature_map": fused,
            "image_feature_vector": image_vector
        }
