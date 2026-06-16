import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split

from sklearn.metrics import (
    classification_report,
    confusion_matrix
)

sys.path.append(
    r"D:\vscode\SWIM\stroke-cdss\src"
)

from imaging.dataset import StrokeDataset
from models.stroke_model import StrokeModel


# ==========================================
# CONFIG
# ==========================================

DATASET_ROOT = (
    r"D:\vscode\SWIM\Brain_Stroke_CT_Dataset"
)

CHECKPOINT_PATH = (
    r"D:\vscode\SWIM\stroke-cdss"
    r"\checkpoints\best_model.pth"
)

BATCH_SIZE = 8

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ==========================================
# DATASET
# ==========================================

dataset = StrokeDataset(
    DATASET_ROOT
)

train_size = int(
    0.8 * len(dataset)
)

val_size = (
    len(dataset)
    - train_size
)

_, val_dataset = random_split(
    dataset,
    [train_size, val_size]
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# ==========================================
# MODEL
# ==========================================

model = StrokeModel().to(
    DEVICE
)

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=DEVICE
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()


# ==========================================
# EVALUATION
# ==========================================

all_preds = []
all_labels = []

with torch.no_grad():

    for batch in val_loader:

        image = batch["image"].to(
            DEVICE
        )

        symmetry = batch[
            "symmetry_map"
        ].to(
            DEVICE
        )

        labels = batch[
            "label"
        ].to(
            DEVICE
        )

        outputs = model(
            image,
            symmetry
        )

        logits = outputs[
            "logits"
        ]

        preds = logits.argmax(
            dim=1
        )

        all_preds.extend(
            preds.cpu().numpy()
        )

        all_labels.extend(
            labels.cpu().numpy()
        )


# ==========================================
# RESULTS
# ==========================================

print("\nClassification Report\n")

print(

    classification_report(

        all_labels,
        all_preds,

        target_names=[

            "Normal",
            "Bleeding",
            "Ischemia"

        ]
    )
)

print("\nConfusion Matrix\n")

print(

    confusion_matrix(

        all_labels,
        all_preds
    )
)