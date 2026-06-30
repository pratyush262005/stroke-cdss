"""
swin_encoder.py

Swin Transformer Encoder for Stroke CDSS

Input:
    CT Image + Symmetry Map

Output:
    Multi-scale feature maps

Author: Stroke CDSS Project
"""

import torch
import torch.nn as nn
import timm


# ============================================================
# Swin Encoder
# ============================================================

class SwinEncoder(nn.Module):

    def __init__(
        self,
        model_name="swin_tiny_patch4_window7_224",
        pretrained=True
    ):

        super().__init__()

        # ----------------------------------------------------
        # Swin Transformer Backbone
        # ----------------------------------------------------

        self.backbone = timm.create_model(

            model_name,

            pretrained=pretrained,

            in_chans=2,          # CT + Symmetry

            features_only=True   # Return multi-scale features
        )

    # ========================================================
    # Forward
    # ========================================================

    def forward(
        self,
        image,
        symmetry
    ):

        # ----------------------------------------------------
        # Concatenate inputs
        # ----------------------------------------------------

        x = torch.cat(
            [image, symmetry],
            dim=1
        )

        # ----------------------------------------------------
        # Swin Features
        # ----------------------------------------------------

        features = self.backbone(
            x
        )

        # timm returns:
        #
        # f1 -> [B, 56, 56, 96]
        # f2 -> [B, 28, 28, 192]
        # f3 -> [B, 14, 14, 384]
        # f4 -> [B, 7, 7, 768]

        processed_features = []

        for feat in features:

            # Convert:
            # [B,H,W,C] -> [B,C,H,W]

            feat = feat.permute(
                0, 3, 1, 2
            )

            processed_features.append(
                feat
            )

        return processed_features


# ============================================================
# Unit Test
# ============================================================

if __name__ == "__main__":

    DEVICE = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model = SwinEncoder().to(
        DEVICE
    )

    image = torch.randn(
        2,
        1,
        224,
        224
    ).to(DEVICE)

    symmetry = torch.randn(
        2,
        1,
        224,
        224
    ).to(DEVICE)

    features = model(
        image,
        symmetry
    )

    print("\nFeature Shapes")
    print("-" * 50)

    for i, feat in enumerate(features):

        print(
            f"f{i+1}: {feat.shape}"
        )