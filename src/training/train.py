"""
train.py

Training Script for Multitask Stroke CDSS

Author: Stroke CDSS Project
"""

import sys

sys.path.append(
    r"D:\vscode\SWIM\stroke-cdss\src"
)

import torch
from torch.utils.data import (
    DataLoader,
    random_split
)
from tqdm import tqdm

from dataset.segmentation_dataset import (
    StrokeSegmentationDataset
)

from models.multitask_model import (
    StrokeCDSS
)

from training.losses import (
    MultitaskLoss
)

from training.validate import (
    validate
)


# ============================================================
# CONFIG
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

BATCH_SIZE = 8
EPOCHS = 1
LEARNING_RATE = 1e-4

DATASET_PATH = (
    r"D:\vscode\SWIM\Brain_Stroke_CT_Dataset"
)

MODEL_SAVE_PATH = (
    r"D:\vscode\SWIM\stroke-cdss\checkpoints"
    r"\best_stroke_cdss.pth"
)


# ============================================================
# DATASET
# ============================================================

dataset = StrokeSegmentationDataset(
    DATASET_PATH
)

train_size = int(
    0.8 * len(dataset)
)

val_size = (
    len(dataset)
    - train_size
)

train_dataset, val_dataset = random_split(
    dataset,
    [train_size, val_size]
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

print(
    f"\nTraining Samples: {len(train_dataset)}"
)

print(
    f"Validation Samples: {len(val_dataset)}"
)


# ============================================================
# MODEL
# ============================================================

model = StrokeCDSS().to(
    DEVICE
)

criterion = MultitaskLoss().to(
    DEVICE
)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE
)

scheduler = (
    torch.optim.lr_scheduler.
    ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2
    )
)

best_f1 = 0


# ============================================================
# TRAINING LOOP
# ============================================================

for epoch in range(EPOCHS):

    model.train()

    running_loss = 0

    progress = tqdm(
        train_loader,
        desc=f"Epoch {epoch+1}/{EPOCHS}"
    )

    for batch in progress:

        image = batch[
            "image"
        ].to(DEVICE)

        symmetry = batch[
            "symmetry"
        ].to(DEVICE)

        mask = batch[
            "mask"
        ].to(DEVICE)

        label = batch[
            "label"
        ].to(DEVICE)

        optimizer.zero_grad()

        outputs = model(
            image,
            symmetry
        )

        losses = criterion(
            outputs["logits"],
            label,
            outputs["mask"],
            mask
        )

        loss = losses[
            "total_loss"
        ]

        loss.backward()

        optimizer.step()

        running_loss += (
            loss.item()
        )

        progress.set_postfix(
            loss=f"{loss.item():.4f}"
        )

    # ========================================================
    # Epoch Results
    # ========================================================

    epoch_loss = (
        running_loss /
        len(train_loader)
    )

    print(
        f"\nEpoch {epoch+1} Loss: "
        f"{epoch_loss:.4f}"
    )

    # ========================================================
    # VALIDATION
    # ========================================================

    metrics = validate(
        model,
        val_loader,
        criterion,
        DEVICE
    )

    scheduler.step(
        metrics["loss"]
    )

    print("\nValidation Metrics")
    print("-" * 50)

    print(
        f"Val Loss      : "
        f"{metrics['loss']:.4f}"
    )

    print(
        f"Accuracy      : "
        f"{metrics['accuracy']:.4f}"
    )

    print(
        f"Precision     : "
        f"{metrics['precision']:.4f}"
    )

    print(
        f"Recall        : "
        f"{metrics['recall']:.4f}"
    )

    print(
        f"F1 Score      : "
        f"{metrics['f1']:.4f}"
    )

    print(
        f"Dice Score    : "
        f"{metrics['dice']:.4f}"
    )

    print(
        "\nConfusion Matrix"
    )

    print(
        metrics[
            "confusion_matrix"
        ]
    )

    print(
        "\nClassification Report"
    )

    print(
        metrics["report"]
    )

    print(
        f"\nCurrent LR: "
        f"{optimizer.param_groups[0]['lr']:.6f}"
    )

    # ========================================================
    # SAVE BEST MODEL
    # ========================================================

    if metrics["f1"] > best_f1:

        best_f1 = metrics["f1"]

        torch.save(
            model.state_dict(),
            MODEL_SAVE_PATH
        )

        print(
            "\nBest model saved!"
        )

print(
    "\nTraining Complete."
)