"""
validate.py

Validation utilities for Stroke CDSS

Author: Stroke CDSS Project
"""

import torch

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# ============================================================
# Dice Score
# ============================================================

def dice_score(
    pred,
    target,
    smooth=1e-6
):

    pred = torch.sigmoid(pred)

    pred = (pred > 0.5).float()

    intersection = (
        pred * target
    ).sum()

    dice = (
        2 * intersection + smooth
    ) / (
        pred.sum()
        + target.sum()
        + smooth
    )

    return dice.item()


# ============================================================
# Validation
# ============================================================

def validate(
    model,
    loader,
    criterion,
    device
):

    model.eval()

    running_loss = 0

    all_preds = []
    all_labels = []

    total_dice = 0

    with torch.no_grad():

        for batch in loader:

            image = batch["image"].to(
                device
            )

            symmetry = batch["symmetry"].to(
                device
            )

            mask = batch["mask"].to(
                device
            )

            label = batch["label"].to(
                device
            )

            outputs = model(
                image,
                symmetry
            )

            # -----------------------------------
            # Dice Score
            # -----------------------------------

            dice = dice_score(
                outputs["mask"],
                mask
            )

            total_dice += dice

            # -----------------------------------
            # Loss
            # -----------------------------------

            losses = criterion(
                outputs["logits"],
                label,
                outputs["mask"],
                mask
            )

            running_loss += (
                losses["total_loss"].item()
            )

            # -----------------------------------
            # Classification Predictions
            # -----------------------------------

            preds = torch.argmax(
                outputs["logits"],
                dim=1
            )

            all_preds.extend(
                preds.cpu().numpy()
            )

            all_labels.extend(
                label.cpu().numpy()
            )

    # ========================================================
    # Metrics
    # ========================================================

    avg_loss = (
        running_loss /
        len(loader)
    )

    avg_dice = (
        total_dice /
        len(loader)
    )

    accuracy = accuracy_score(
        all_labels,
        all_preds
    )

    precision = precision_score(
        all_labels,
        all_preds,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        all_labels,
        all_preds,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        all_labels,
        all_preds,
        average="weighted",
        zero_division=0
    )

    cm = confusion_matrix(
        all_labels,
        all_preds
    )

    report = classification_report(
        all_labels,
        all_preds,
        target_names=[
            "Normal",
            "Bleeding",
            "Ischemia"
        ]
    )

    return {

        "loss": avg_loss,

        "accuracy": accuracy,

        "precision": precision,

        "recall": recall,

        "f1": f1,

        "dice": avg_dice,

        "confusion_matrix": cm,

        "report": report
    }