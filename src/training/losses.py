"""
losses.py

Multitask Loss Functions

Author: Stroke CDSS Project
"""

import torch
import torch.nn as nn


# ============================================================
# Dice Loss
# ============================================================

class DiceLoss(nn.Module):

    def __init__(
        self,
        smooth=1e-6
    ):

        super().__init__()

        self.smooth = smooth

    def forward(
        self,
        pred,
        target
    ):

        pred = torch.sigmoid(
            pred
        )

        pred = pred.view(-1)

        target = target.view(-1)

        intersection = (
            pred * target
        ).sum()

        dice = (
            2.0 * intersection
            + self.smooth
        ) / (
            pred.sum()
            + target.sum()
            + self.smooth
        )

        return 1 - dice


# ============================================================
# Multitask Loss
# ============================================================

class MultitaskLoss(nn.Module):

    def __init__(
        self,
        cls_weight=1.0,
        seg_weight=2.0
    ):

        super().__init__()

        # ----------------------------------------
        # Classification Class Weights
        # ----------------------------------------

        class_weights = torch.tensor(
            [
                1.50,   # Normal
                5.88,   # Bleeding
                6.08    # Ischemia
            ],
            dtype=torch.float32
        )

        self.register_buffer(
            "class_weights",
            class_weights
        )

        # ----------------------------------------
        # Losses
        # ----------------------------------------

        self.cls_loss = (
            nn.CrossEntropyLoss(
                weight=self.class_weights
            )
        )

        self.dice_loss = DiceLoss()

        self.bce_loss = (
            nn.BCEWithLogitsLoss()
        )

        self.cls_weight = cls_weight

        self.seg_weight = seg_weight

    # ========================================================
    # Forward
    # ========================================================

    def forward(
        self,
        pred_logits,
        true_labels,
        pred_mask,
        true_mask
    ):

        self.cls_loss.weight = (
            self.class_weights.to(
                pred_logits.device
            )
        )

        # ----------------------------------------
        # Classification Loss
        # ----------------------------------------

        classification_loss = (
            self.cls_loss(
                pred_logits,
                true_labels
            )
        )

        # ----------------------------------------
        # Segmentation Loss
        # ----------------------------------------

        dice = self.dice_loss(
            pred_mask,
            true_mask
        )

        bce = self.bce_loss(
            pred_mask,
            true_mask
        )

        segmentation_loss = (
            dice + bce
        )

        # ----------------------------------------
        # Total Loss
        # ----------------------------------------

        total_loss = (

            self.cls_weight
            * classification_loss

            +

            self.seg_weight
            * segmentation_loss
        )

        return {

            "total_loss":
                total_loss,

            "classification_loss":
                classification_loss,

            "segmentation_loss":
                segmentation_loss
        }


# ============================================================
# Unit Test
# ============================================================

if __name__ == "__main__":

    criterion = MultitaskLoss()

    pred_logits = torch.randn(
        4,
        3
    )

    true_labels = torch.tensor(
        [0, 1, 2, 1]
    )

    pred_mask = torch.randn(
        4,
        1,
        224,
        224
    )

    true_mask = torch.randint(
        0,
        2,
        (4, 1, 224, 224)
    ).float()

    losses = criterion(
        pred_logits,
        true_labels,
        pred_mask,
        true_mask
    )

    print("\nLosses")
    print("-" * 50)

    for k, v in losses.items():

        print(
            f"{k}: {v.item():.4f}"
        )