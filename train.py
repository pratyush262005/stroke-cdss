
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

sys.path.append(r"D:\vscode\SWIM\stroke-cdss\src")

from imaging.dataset import StrokeDataset
from models.stroke_model import StrokeModel
from training.trainer import Trainer


# ==========================================================
# CONFIG
# ==========================================================

DATASET_ROOT =r"D:\vscode\SWIM\Brain_Stroke_CT_Dataset"


BATCH_SIZE = 8
NUM_EPOCHS = 20
LEARNING_RATE = 1e-4

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

CHECKPOINT_DIR = Path(
    r"D:\vscode\SWIM\stroke-cdss\checkpoints"
)

CHECKPOINT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ==========================================================
# DATASET
# ==========================================================

print("Loading dataset...")

dataset = StrokeDataset(
    DATASET_ROOT
)

print(
    f"Total Samples: {len(dataset)}"
)

train_size = int(
    0.8 * len(dataset)
)

val_size = (
    len(dataset)
    - train_size
)

train_dataset, val_dataset = (
    random_split(
        dataset,
        [train_size, val_size]
    )
)

print(
    f"Train: {len(train_dataset)}"
)

print(
    f"Validation: {len(val_dataset)}"
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


# ==========================================================
# MODEL
# ==========================================================

print("Creating model...")

model = StrokeModel().to(
    DEVICE
)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE
)


# ==========================================================
# TRAINER
# ==========================================================

trainer = Trainer(
    model=model,
    optimizer=optimizer,
    criterion=criterion,
    device=DEVICE
)

best_val_acc = 0.0


# ==========================================================
# TRAIN LOOP
# ==========================================================

for epoch in range(
    NUM_EPOCHS
):

    print(
        f"\nEpoch "
        f"{epoch+1}/{NUM_EPOCHS}"
    )

    train_metrics = (
        trainer.train_one_epoch(
            train_loader
        )
    )

    val_metrics = (
        trainer.validate(
            val_loader
        )
    )

    print(
        f"Train Loss: "
        f"{train_metrics['loss']:.4f}"
    )

    print(
        f"Train Acc: "
        f"{train_metrics['accuracy']:.4f}"
    )

    print(
        f"Val Loss: "
        f"{val_metrics['loss']:.4f}"
    )

    print(
        f"Val Acc: "
        f"{val_metrics['accuracy']:.4f}"
    )

    if (
        val_metrics["accuracy"]
        >
        best_val_acc
    ):

        best_val_acc = (
            val_metrics["accuracy"]
        )

        checkpoint_path = (
            CHECKPOINT_DIR
            /
            "best_model.pth"
        )

        torch.save(

            {
                "epoch": epoch,
                "model_state_dict":
                    model.state_dict(),

                "optimizer_state_dict":
                    optimizer.state_dict(),

                "val_accuracy":
                    best_val_acc
            },

            checkpoint_path
        )

        print(
            f"Saved: "
            f"{checkpoint_path}"
        )


print(
    "\nTraining Complete."
)

print(
    f"Best Validation Accuracy: "
    f"{best_val_acc:.4f}"
)
