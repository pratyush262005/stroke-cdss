"""
classification_head.py

Step 3

Multi-Class Stroke Classification
"""

import torch
import torch.nn as nn


class StrokeClassificationHead(nn.Module):

    def __init__(
        self,
        input_dim=768,
        hidden_dim=512,
        num_classes=3,
        dropout=0.3
    ):
        super().__init__()

        self.classifier = nn.Sequential(

            nn.Linear(
                input_dim,
                hidden_dim
            ),

            nn.ReLU(),

            nn.Dropout(
                dropout
            ),

            nn.Linear(
                hidden_dim,
                num_classes
            )
        )

    def forward(
        self,
        image_feature_vector
    ):

        logits = self.classifier(
            image_feature_vector
        )

        return logits
