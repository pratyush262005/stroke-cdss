"""
swin_encoder.py

Step 2A

Swin Transformer Encoder
"""

import torch
import torch.nn as nn
import timm


class SwinEncoder(nn.Module):

    def __init__(
        self,
        model_name="swin_tiny_patch4_window7_224",
        pretrained=True
    ):
        super().__init__()

        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=True,
            in_chans=2
        )

        self.feature_info = (
            self.backbone.feature_info
        )

    def forward(self, image, symmetry):

        x = torch.cat(
            [
                image,
                symmetry
            ],
            dim=1
        )

        features = self.backbone(x)

        return {

            "stage1": features[0],
            "stage2": features[1],
            "stage3": features[2],
            "stage4": features[3]
        }
