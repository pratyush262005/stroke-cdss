"""
classification_head.py

Classification Head for Stroke CDSS

Classes:
0 -> Normal
1 -> Bleeding
2 -> Ischemia

Author: Stroke CDSS Project
"""

import torch
import torch.nn as nn


# ============================================================
# Classification Head
# ============================================================

class ClassificationHead(nn.Module):

    def __init__(
        self,
        in_features=768,
        num_classes=3,
        dropout=0.3
    ):

        super().__init__()

        self.pool = nn.AdaptiveAvgPool2d(
            (1, 1)
        )

        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Linear(
                in_features,
                512
            ),

            nn.ReLU(inplace=True),

            nn.Dropout(
                dropout
            ),

            nn.Linear(
                512,
                128
            ),

            nn.ReLU(inplace=True),

            nn.Dropout(
                dropout
            ),

            nn.Linear(
                128,
                num_classes
            )
        )

    # ========================================================
    # Forward
    # ========================================================

    def forward(
        self,
        x
    ):

        # x shape:
        # [B, 768, 7, 7]

        x = self.pool(
            x
        )

        logits = self.classifier(
            x
        )

        return logits


# ============================================================
# Unit Test
# ============================================================

if __name__ == "__main__":

    DEVICE = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model = ClassificationHead().to(
        DEVICE
    )

    x = torch.randn(
        2,
        768,
        7,
        7
    ).to(DEVICE)

    logits = model(x)

    print(
        "Logits Shape:",
        logits.shape
    )