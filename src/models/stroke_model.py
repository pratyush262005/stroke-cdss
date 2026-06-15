
import torch
import torch.nn as nn

from models.swin_encoder import SwinEncoder
from models.feature_fusion import MultiScaleFeatureFusion
from models.classification_head import StrokeClassificationHead


class StrokeModel(nn.Module):

    def __init__(self):

        super().__init__()

        self.encoder = SwinEncoder(
            pretrained=False
        )

        self.fusion = MultiScaleFeatureFusion()

        self.classifier = (
            StrokeClassificationHead()
        )

    def forward(
        self,
        image,
        symmetry
    ):

        features = self.encoder(
            image,
            symmetry
        )

        fusion_output = (
            self.fusion(features)
        )

        logits = self.classifier(
            fusion_output[
                "image_feature_vector"
            ]
        )

        return {
            "logits": logits,
            "features": features,
            "fusion_output": fusion_output
        }
