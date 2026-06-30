"""
multitask_model.py

Multitask Stroke CDSS Model

Tasks:
1. Stroke Classification
2. Lesion Segmentation

Architecture:
CT + Symmetry
        ↓
Swin Transformer Encoder
        ↓
 ┌───────────────┬──────────────┐
 ↓                              ↓
Classification Head      U-Net Decoder

Author: Stroke CDSS Project
"""

import torch
import torch.nn as nn

from models.swin_encoder import SwinEncoder
from models.classification_head import (
    ClassificationHead
)
from models.unet_decoder import (
    UNetDecoder
)


# ============================================================
# Multitask Model
# ============================================================

class StrokeCDSS(nn.Module):

    def __init__(
        self,
        num_classes=3
    ):

        super().__init__()

        # ----------------------------------------------------
        # Encoder
        # ----------------------------------------------------

        self.encoder = SwinEncoder()

        # ----------------------------------------------------
        # Classification Branch
        # ----------------------------------------------------

        self.classifier = ClassificationHead(
            in_features=768,
            num_classes=num_classes
        )

        # ----------------------------------------------------
        # Segmentation Branch
        # ----------------------------------------------------

        self.decoder = UNetDecoder()

    # ========================================================
    # Forward
    # ========================================================

    def forward(
        self,
        image,
        symmetry
    ):

        # ----------------------------------------
        # Encoder Features
        # ----------------------------------------

        features = self.encoder(
            image,
            symmetry
        )

        # Features:
        # f1 -> [B, 96, 56, 56]
        # f2 -> [B, 192, 28, 28]
        # f3 -> [B, 384, 14, 14]
        # f4 -> [B, 768, 7, 7]

        f1, f2, f3, f4 = features

        # ----------------------------------------
        # Classification
        # ----------------------------------------

        logits = self.classifier(
            f4
        )

        # ----------------------------------------
        # Segmentation
        # ----------------------------------------

        mask = self.decoder(
            features
        )

        return {

            "logits": logits,

            "mask": mask
        }


# ============================================================
# Unit Test
# ============================================================

if __name__ == "__main__":

    DEVICE = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model = StrokeCDSS().to(
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

    outputs = model(
        image,
        symmetry
    )

    print("\nOutputs")
    print("-" * 50)

    print(
        "Logits Shape:",
        outputs["logits"].shape
    )

    print(
        "Mask Shape:",
        outputs["mask"].shape
    )